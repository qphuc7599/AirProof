/** Sepolia-only feasibility sample. Credentials are never written to artifacts. */
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { Contract, ContractFactory, JsonRpcProvider, Wallet, formatEther, id } from "ethers";

const EXPECTED_CHAIN = 11155111n;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function loadCredentials(filename) {
  const contents = fs.readFileSync(filename, "utf8");
  const urls = [...new Set(contents.match(/https:\/\/[^\s<>`"')\]]+/g) || [])];
  const keys = [...new Set((contents.match(/(?<![a-fA-F0-9])(?:0x)?[a-fA-F0-9]{64}(?![a-fA-F0-9])/g) || [])
    .map((value) => value.startsWith("0x") ? value : `0x${value}`))];
  const addresses = [...new Set((contents.match(/0x[a-fA-F0-9]{40}(?![a-fA-F0-9])/g) || []).map((v) => v.toLowerCase()))];
  if (urls.length !== 1 || keys.length !== 1 || addresses.length !== 1) {
    throw new Error("CREDENTIAL_FORMAT_REQUIRES_ONE_HTTPS_RPC_KEY_AND_ADDRESS");
  }
  const wallet = new Wallet(keys[0]);
  if (wallet.address.toLowerCase() !== addresses[0]) throw new Error("WALLET_ADDRESS_MISMATCH");
  return { url: urls[0], wallet, rpcHost: new URL(urls[0]).hostname };
}

export async function makePreflight(credentials, anchors = 30) {
  let requestId = 0;
  const rpc = async (method, params = []) => {
    const response = await fetch(credentials.url, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ jsonrpc: "2.0", id: ++requestId, method, params }),
      signal: AbortSignal.timeout(25000),
    });
    if (!response.ok) throw new Error(`RPC_HTTP_${response.status}`);
    const data = await response.json();
    if (data.error || data.result === undefined) throw new Error(`RPC_FAILED_${method}`);
    return data.result;
  };
  const chain = BigInt(await rpc("eth_chainId"));
  if (chain !== EXPECTED_CHAIN) throw new Error("REFUSING_NON_SEPOLIA_CHAIN");
  const [balanceHex, block, gasPriceHex] = await Promise.all([
    rpc("eth_getBalance", [credentials.wallet.address, "latest"]),
    rpc("eth_getBlockByNumber", ["latest", false]), rpc("eth_gasPrice"),
  ]);
  const balance = BigInt(balanceHex);
  const latestTimestamp = Number(BigInt(block.timestamp));
  if (Math.abs(Date.now() / 1000 - latestTimestamp) > 7200) throw new Error("RPC_CHAIN_HEAD_STALE");
  const reportedGasPrice = BigInt(gasPriceHex);
  // Small bounded testnet budget. An unexpected fee spike pauses the campaign.
  const maximumFeePerGas = reportedGasPrice * 3n > 100000000n ? reportedGasPrice * 3n : 100000000n;
  if (maximumFeePerGas > 10000000000n) throw new Error("SEPOLIA_FEE_EXCEEDS_10_GWEI_CAP");
  const gasAllowance = 1000000n + 100000n + BigInt(anchors) * 300000n;
  const requiredBalance = gasAllowance * maximumFeePerGas;
  return {
    chain_id: Number(chain), network: "sepolia", rpc_host: credentials.rpcHost,
    wallet_address: credentials.wallet.address, balance_test_eth: formatEther(balance),
    latest_block: Number(BigInt(block.number)), latest_block_timestamp: latestTimestamp,
    gas_price_wei: reportedGasPrice.toString(), max_fee_per_gas_wei: maximumFeePerGas.toString(),
    reserved_gas_allowance: gasAllowance.toString(),
    conservative_required_test_eth: formatEther(requiredBalance),
    funded_for_sample: balance >= requiredBalance, anchors, checked_at: new Date().toISOString(),
  };
}

async function benchmark(credentials, args, preflight) {
  if (!preflight.funded_for_sample) throw new Error("INSUFFICIENT_TESTNET_ETH_FOR_SAMPLE");
  const output = path.resolve(args.output);
  fs.mkdirSync(output, { recursive: false }); // never overwrite an existing campaign
  const provider = new JsonRpcProvider(credentials.url, EXPECTED_CHAIN, { staticNetwork: true });
  provider.pollingInterval = 2500;
  const signer = credentials.wallet.connect(provider);
  const contractDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const artifact = JSON.parse(fs.readFileSync(path.join(contractDirectory, "artifacts/src/AirProofAnchor.sol/AirProofAnchor.json"), "utf8"));
  const maxFee = BigInt(preflight.max_fee_per_gas_wei);
  const fees = { maxFeePerGas: maxFee, maxPriorityFeePerGas: maxFee / 3n };
  let nonce = await provider.getTransactionCount(signer.address, "pending");
  const eventsPath = path.join(output, "events.jsonl");
  const log = (row) => fs.appendFileSync(eventsPath, `${JSON.stringify({ time: new Date().toISOString(), ...row })}\n`);
  fs.writeFileSync(path.join(output, "manifest.json"), JSON.stringify({
    ...preflight, protocol: "sepolia-feasibility-three-time-blocks-v1", anchors_per_block: args.perBlock,
    time_blocks: 3, block_spacing_seconds: args.spacing, confirmations: args.confirmations,
    contract_source_sha256: crypto.createHash("sha256").update(fs.readFileSync(path.join(contractDirectory, "src/AirProofAnchor.sol"))).digest("hex"),
    bytecode_sha256: crypto.createHash("sha256").update(artifact.bytecode).digest("hex"),
    started_at: new Date().toISOString(),
  }, null, 2));

  async function transact(unsigned, label, extra = {}) {
    const transaction = { ...unsigned, ...fees, chainId: EXPECTED_CHAIN, nonce: nonce++, type: 2 };
    const signed = await signer.signTransaction(transaction);
    const hash = crypto.createHash("sha256"); // only used for a non-secret payload digest below
    hash.update(unsigned.data || "");
    const { keccak256 } = await import("ethers");
    const transactionHash = keccak256(signed);
    const started = performance.now();
    const base = { label, ...extra, transaction_hash: transactionHash, nonce: transaction.nonce };
    // Record the deterministic hash BEFORE broadcast; an uncertain response must
    // never trigger a newly signed transaction with a different nonce.
    log({ event: "signed", ...base, data_sha256: hash.digest("hex") });
    try {
      await provider.broadcastTransaction(signed);
    } catch (error) {
      log({ event: "rpc_failure", stage: "broadcast", ...base, code: safeCode(error) });
      // The original transaction may have reached the node. Observe that exact
      // hash, never blindly resend or submit a replacement transaction.
    }
    const receipt = await provider.waitForTransaction(transactionHash, 1, 300000);
    if (!receipt) throw new Error("TRANSACTION_RECEIPT_TIMEOUT_CHECK_JOURNAL_HASH");
    const receiptSeconds = (performance.now() - started) / 1000;
    const confirmed = await provider.waitForTransaction(transactionHash, args.confirmations, 300000);
    if (!confirmed) throw new Error("TRANSACTION_CONFIRMATION_TIMEOUT");
    const row = {
      ...base, status: receipt.status, reverted: receipt.status !== 1,
      block_number: receipt.blockNumber, block_hash: receipt.blockHash,
      gas_used: Number(receipt.gasUsed), gas_price_wei: receipt.gasPrice.toString(),
      fee_test_eth: formatEther(receipt.fee), time_to_receipt_seconds: receiptSeconds,
      time_to_confirmation_seconds: (performance.now() - started) / 1000,
      confirmations: args.confirmations, contract_address: receipt.contractAddress,
    };
    log({ event: "receipt", ...row });
    console.log(JSON.stringify({ event: "receipt", label, ...extra, status: row.status,
      gas_used: row.gas_used, time_to_receipt_seconds: row.time_to_receipt_seconds }));
    if (receipt.status !== 1) throw new Error("TRANSACTION_REVERTED");
    return row;
  }

  try {
    const factory = new ContractFactory(artifact.abi, artifact.bytecode, signer);
    const deployment = await transact({ ...(await factory.getDeployTransaction()), gasLimit: 1000000n }, "deployment");
    const contract = new Contract(deployment.contract_address, artifact.abi, signer);
    const run = `${Date.now()}`;
    const schema = id("airproof:v4:public-audit-schema:v1");
    const policy = id(`airproof:v4:sepolia:${run}`);
    await transact({ ...(await contract.registerPolicy.populateTransaction(policy, schema)), gasLimit: 100000n }, "register_policy");
    const startedBlocks = Date.now();
    const anchors = [];
    for (let block = 0; block < 3; block++) {
      const target = startedBlocks + block * args.spacing * 1000;
      while (Date.now() < target) await sleep(Math.min(30000, target - Date.now()));
      log({ event: "time_block_started", time_block: block });
      console.log(JSON.stringify({ event: "time_block_started", time_block: block }));
      for (let index = 0; index < args.perBlock; index++) {
        const replicate = block * args.perBlock + index;
        const head = await provider.getBlock("latest");
        const epoch = Math.floor(head.timestamp / 3600);
        // Public synthetic commitments only: no citizen data, credentials or RPC secrets.
        const city = id(`airproof:synthetic-city:${run}:${replicate}`);
        const root = id(`airproof:synthetic-root:${run}:${replicate}`);
        const unsigned = await contract.anchorBatch.populateTransaction(
          city, epoch, root, 512, schema, policy, id(`synthetic-cid:${run}:${replicate}`),
        );
        const row = await transact({ ...unsigned, gasLimit: 300000n }, "anchor", { time_block: block, replicate });
        const stored = await contract.getAnchor(city, epoch);
        if (stored.root !== root || Number(stored.count) !== 512) throw new Error("ONCHAIN_ANCHOR_READBACK_MISMATCH");
        anchors.push({ ...row, readback_verified: true });
        fs.writeFileSync(path.join(output, "anchors.json"), JSON.stringify(anchors, null, 2));
      }
    }
    // Confirmation count is measured above. Query consensus-finalized status
    // separately; never equate two confirmations with consensus finality.
    const finalized = await provider.getBlock("finalized");
    const summary = {
      environment: "ethereum-sepolia-public-testnet", contract_address: deployment.contract_address,
      anchors: anchors.length, successful_anchors: anchors.filter((r) => r.status === 1).length,
      readback_verified: anchors.every((r) => r.readback_verified),
      time_blocks: 3, confirmations: args.confirmations,
      finalized_block_at_end: finalized?.number ?? null,
      anchors_finalized_at_end: finalized ? anchors.filter((r) => r.block_number <= finalized.number).length : null,
      finished_at: new Date().toISOString(),
      rpc_failures: fs.readFileSync(eventsPath, "utf8").split("\n").filter((line) => line.includes('"event":"rpc_failure"')).length,
      reverted_transactions: anchors.filter((r) => r.reverted).length,
    };
    fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(summary, null, 2));
    console.log(JSON.stringify({ event: "complete", ...summary }));
  } catch (error) {
    const failure = { event: "campaign_failed", code: safeCode(error), at: new Date().toISOString() };
    log(failure);
    fs.writeFileSync(path.join(output, "failure.json"), JSON.stringify(failure, null, 2));
    throw error;
  } finally {
    provider.destroy();
  }
}

function safeCode(error) {
  // Never log an ethers exception object/message: it can contain an RPC URL.
  if (typeof error?.code === "string" && /^[A-Z_0-9]+$/.test(error.code)) return error.code;
  if (typeof error?.message === "string" && /^[A-Z_0-9]+$/.test(error.message)) return error.message;
  return "REDACTED_RUNTIME_ERROR";
}

async function main() {
  const { values } = parseArgs({ options: {
    credentials: { type: "string" }, output: { type: "string" }, preflight: { type: "boolean", default: false },
    "per-block": { type: "string", default: "10" }, "spacing-seconds": { type: "string", default: "1800" },
    confirmations: { type: "string", default: "2" },
  } });
  if (!values.credentials || !values.output) throw new Error("CREDENTIAL_PATH_AND_OUTPUT_REQUIRED");
  const args = { output: values.output, perBlock: Number(values["per-block"]),
    spacing: Number(values["spacing-seconds"]), confirmations: Number(values.confirmations) };
  if (!Number.isInteger(args.perBlock) || args.perBlock < 10 || args.perBlock > 20
      || !Number.isInteger(args.spacing) || args.spacing < 300
      || !Number.isInteger(args.confirmations) || args.confirmations < 2) throw new Error("INVALID_SAMPLE_PROTOCOL");
  const credentials = loadCredentials(values.credentials);
  const preflight = await makePreflight(credentials, args.perBlock * 3);
  if (values.preflight) {
    fs.mkdirSync(path.dirname(path.resolve(values.output)), { recursive: true });
    fs.writeFileSync(values.output, JSON.stringify(preflight, null, 2), { flag: "wx" });
    console.log(JSON.stringify(preflight));
  } else {
    await benchmark(credentials, args, preflight);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(JSON.stringify({ error: safeCode(error) })); process.exitCode = 1; });
}
