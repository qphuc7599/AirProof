/** Sepolia finality campaign bounded to less than three hours. */
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { Contract, ContractFactory, JsonRpcProvider, Wallet, formatEther, id, keccak256 } from "ethers";

const CHAIN_ID = 11155111n;
const EXPECTED_WALLET = "0xc6d80cb8eef27cfdbf8ecaf208e5c62ed0a0023d";
const RPC = process.env.SEPOLIA_RPC_URL;
const PRIVATE_KEY = process.env.SEPOLIA_PRIVATE_KEY;
const OUTPUT = process.env.SEPOLIA_OUTPUT;
const BLOCKS = 3;
const PER_BLOCK = 10;
const SPACING_MS = 30 * 60 * 1000;
const FINALITY_POLL_MS = 12 * 1000;
const FINALITY_WAIT_MS = 45 * 60 * 1000;
const CAMPAIGN_LIMIT_MS = 2 * 60 * 60 * 1000;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const sha256 = bytes => crypto.createHash("sha256").update(bytes).digest("hex");

function safeCode(error) {
  if (typeof error?.code === "string" && /^[A-Z_0-9]+$/.test(error.code)) return error.code;
  if (typeof error?.message === "string" && /^[A-Z_0-9]+$/.test(error.message)) return error.message;
  return "REDACTED_RUNTIME_ERROR";
}

function requireInputs() {
  if (!RPC?.startsWith("https://") || !PRIVATE_KEY || !OUTPUT) throw new Error("SECURE_ENV_INPUTS_REQUIRED");
  const wallet = new Wallet(PRIVATE_KEY);
  if (wallet.address.toLowerCase() !== EXPECTED_WALLET) throw new Error("WALLET_ADDRESS_MISMATCH");
  return wallet;
}

async function main() {
  const wallet = requireInputs();
  const output = path.resolve(OUTPUT);
  fs.mkdirSync(output, { recursive: false });
  const events = path.join(output, "events.jsonl");
  const log = row => fs.appendFileSync(events, `${JSON.stringify({ observed_at: new Date().toISOString(), ...row })}\n`);
  const provider = new JsonRpcProvider(RPC, CHAIN_ID, { staticNetwork: true });
  provider.pollingInterval = 2500;
  const signer = wallet.connect(provider);
  const startedAt = Date.now();
  let nonce;
  const anchors = [];
  let monitoring = true;
  const artifactPath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../artifacts/src/AirProofAnchor.sol/AirProofAnchor.json");
  const sourcePath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../src/AirProofAnchor.sol");
  const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));

  async function refreshFinality() {
    try {
      const finalized = await provider.getBlock("finalized");
      const now = new Date().toISOString();
      let changed = false;
      for (const row of anchors) {
        if (!row.finalized_observed_at && row.block_number <= finalized.number) {
          row.finalized_observed_at = now;
          row.time_to_observed_finality_seconds = (Date.parse(now) - Date.parse(row.signed_at)) / 1000;
          row.finalized_head_number_when_observed = finalized.number;
          changed = true;
          log({ event: "anchor_finalized", transaction_hash: row.transaction_hash,
            replicate: row.replicate, time_to_observed_finality_seconds: row.time_to_observed_finality_seconds,
            finalized_head_number: finalized.number });
        }
      }
      if (changed) fs.writeFileSync(path.join(output, "anchors.json"), JSON.stringify(anchors, null, 2));
      return finalized;
    } catch (error) {
      log({ event: "rpc_failure", stage: "finality_poll", code: safeCode(error) });
      return null;
    }
  }

  const monitor = (async () => {
    while (monitoring) { await refreshFinality(); await sleep(FINALITY_POLL_MS); }
  })();

  async function transact(unsigned, label, extra = {}) {
    const fee = await provider.getFeeData();
    const gasPrice = fee.gasPrice ?? 2_000_000_000n;
    const maxFeePerGas = fee.maxFeePerGas ?? gasPrice * 2n;
    if (maxFeePerGas > 10_000_000_000n) throw new Error("SEPOLIA_FEE_EXCEEDS_10_GWEI_CAP");
    const tx = { ...unsigned, maxFeePerGas, maxPriorityFeePerGas: fee.maxPriorityFeePerGas ?? gasPrice / 3n,
      chainId: CHAIN_ID, nonce: nonce++, type: 2 };
    const signedAt = new Date().toISOString();
    const signed = await signer.signTransaction(tx);
    const transactionHash = keccak256(signed);
    const base = { label, ...extra, signed_at: signedAt, transaction_hash: transactionHash, nonce: tx.nonce };
    log({ event: "signed", ...base, data_sha256: sha256(unsigned.data || "") });
    try { await provider.broadcastTransaction(signed); }
    catch (error) { log({ event: "rpc_failure", stage: "broadcast", ...base, code: safeCode(error) }); }
    const receipt = await provider.waitForTransaction(transactionHash, 1, 300000);
    if (!receipt) throw new Error("TRANSACTION_RECEIPT_TIMEOUT_CHECK_JOURNAL_HASH");
    const confirmed = await provider.waitForTransaction(transactionHash, 2, 300000);
    if (!confirmed) throw new Error("TRANSACTION_CONFIRMATION_TIMEOUT");
    const result = { ...base, status: receipt.status, reverted: receipt.status !== 1,
      block_number: receipt.blockNumber, block_hash: receipt.blockHash,
      gas_used: Number(receipt.gasUsed), gas_price_wei: receipt.gasPrice.toString(),
      fee_test_eth: formatEther(receipt.fee), receipt_observed_at: new Date().toISOString(), confirmations: 2,
      contract_address: receipt.contractAddress };
    result.time_to_receipt_seconds = (Date.parse(result.receipt_observed_at) - Date.parse(signedAt)) / 1000;
    log({ event: "receipt", ...result });
    if (receipt.status !== 1) throw new Error("TRANSACTION_REVERTED");
    return result;
  }

  try {
    const [network, latest, balance, pendingNonce, fee] = await Promise.all([
      provider.getNetwork(), provider.getBlock("latest"), provider.getBalance(wallet.address),
      provider.getTransactionCount(wallet.address, "pending"), provider.getFeeData()]);
    if (network.chainId !== CHAIN_ID) throw new Error("REFUSING_NON_SEPOLIA_CHAIN");
    if (Date.now() / 1000 - latest.timestamp > 120) throw new Error("RPC_CHAIN_HEAD_STALE");
    if (balance < 40_000_000_000_000_000n) throw new Error("INSUFFICIENT_TESTNET_ETH_FOR_CAMPAIGN");
    nonce = pendingNonce;
    const manifest = { protocol: "sepolia-finality-three-blocks-v1", network: "ethereum-sepolia-public-testnet",
      chain_id: Number(CHAIN_ID), wallet_address: wallet.address, rpc_host: new URL(RPC).hostname,
      starting_balance_test_eth: formatEther(balance), latest_block: latest.number,
      latest_block_age_seconds: Date.now()/1000-latest.timestamp, gas_price_wei: fee.gasPrice?.toString() ?? null,
      time_blocks: BLOCKS, anchors_per_block: PER_BLOCK, spacing_seconds: SPACING_MS/1000,
      confirmations: 2, finality_poll_seconds: FINALITY_POLL_MS/1000,
      finality_wait_after_last_anchor_seconds: FINALITY_WAIT_MS/1000,
      hard_campaign_limit_seconds: CAMPAIGN_LIMIT_MS/1000,
      contract_source_sha256: sha256(fs.readFileSync(sourcePath)), bytecode_sha256: sha256(artifact.bytecode),
      started_at: new Date(startedAt).toISOString() };
    fs.writeFileSync(path.join(output, "manifest.json"), JSON.stringify(manifest, null, 2), { flag: "wx" });

    const factory = new ContractFactory(artifact.abi, artifact.bytecode, signer);
    const deployment = await transact({ ...(await factory.getDeployTransaction()), gasLimit: 1_000_000n }, "deployment");
    const contract = new Contract(deployment.contract_address, artifact.abi, signer);
    const run = `${startedAt}`;
    const schema = id("airproof:public-finality-schema:v1");
    const policy = id(`airproof:sepolia-finality:${run}`);
    await transact({ ...(await contract.registerPolicy.populateTransaction(policy, schema)), gasLimit: 100_000n }, "register_policy");
    const blockZero = Date.now();
    for (let block = 0; block < BLOCKS; block++) {
      const target = blockZero + block * SPACING_MS;
      while (Date.now() < target) await sleep(Math.min(30000, target-Date.now()));
      log({ event: "time_block_started", time_block: block });
      for (let index = 0; index < PER_BLOCK; index++) {
        if (Date.now()-startedAt > CAMPAIGN_LIMIT_MS) throw new Error("CAMPAIGN_TIME_LIMIT_EXCEEDED");
        const replicate = block*PER_BLOCK+index;
        const head = await provider.getBlock("latest");
        const epoch = Math.floor(head.timestamp/3600);
        const city = id(`airproof:finality-city:${run}:${replicate}`);
        const root = id(`airproof:finality-root:${run}:${replicate}`);
        const unsigned = await contract.anchorBatch.populateTransaction(city,epoch,root,512,schema,policy,id(`public-cid:${run}:${replicate}`));
        const row = await transact({ ...unsigned, gasLimit: 300_000n }, "anchor", { time_block:block, replicate });
        const stored = await contract.getAnchor(city,epoch);
        if (stored.root !== root || Number(stored.count) !== 512) throw new Error("ONCHAIN_ANCHOR_READBACK_MISMATCH");
        anchors.push({ ...row, readback_verified:true, finalized_observed_at:null,
          time_to_observed_finality_seconds:null, finalized_head_number_when_observed:null });
        fs.writeFileSync(path.join(output,"anchors.json"),JSON.stringify(anchors,null,2));
        await refreshFinality();
      }
    }
    const waitDeadline = Math.min(startedAt+CAMPAIGN_LIMIT_MS,Date.now()+FINALITY_WAIT_MS);
    while (anchors.some(row=>!row.finalized_observed_at) && Date.now()<waitDeadline) await sleep(FINALITY_POLL_MS);
    await refreshFinality();
    const endBalance = await provider.getBalance(wallet.address);
    const finalized = anchors.filter(row=>row.finalized_observed_at);
    const latencies = finalized.map(row=>row.time_to_observed_finality_seconds).sort((a,b)=>a-b);
    const quantile = q => latencies.length ? latencies[Math.min(latencies.length-1,Math.floor(q*(latencies.length-1)))] : null;
    const summary = { environment:"ethereum-sepolia-public-testnet", contract_address:deployment.contract_address,
      anchors:anchors.length, successful_anchors:anchors.filter(row=>row.status===1).length,
      readback_verified:anchors.every(row=>row.readback_verified), finalized_observed:finalized.length,
      right_censored_at_end:anchors.length-finalized.length, finality_latency_seconds:{min:quantile(0),median:quantile(.5),p95:quantile(.95),max:quantile(1)},
      rpc_failures:fs.readFileSync(events,"utf8").split("\n").filter(line=>line.includes('"event":"rpc_failure"')).length,
      reverted_transactions:anchors.filter(row=>row.reverted).length,
      ending_balance_test_eth:formatEther(endBalance), elapsed_seconds:(Date.now()-startedAt)/1000,
      finished_at:new Date().toISOString(), scope:"Public Sepolia finality observation; not Ethereum mainnet performance." };
    fs.writeFileSync(path.join(output,"summary.json"),JSON.stringify(summary,null,2));
    console.log(JSON.stringify({ event:"complete", ...summary }));
  } catch (error) {
    const failure={event:"campaign_failed",code:safeCode(error),at:new Date().toISOString()};
    log(failure);fs.writeFileSync(path.join(output,"failure.json"),JSON.stringify(failure,null,2));
    throw error;
  } finally {
    monitoring=false;await monitor;provider.destroy();
  }
}

main().catch(error=>{console.error(JSON.stringify({error:safeCode(error)}));process.exitCode=1;});
