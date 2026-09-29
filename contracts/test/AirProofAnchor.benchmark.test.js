import { expect } from "chai";
import { network } from "hardhat";
import fs from "node:fs";
import path from "node:path";

const { ethers } = await network.create();

describe("AirProofAnchor 30-repetition local benchmark", function () {
  it("records deployment and anchor gas/latency", async function () {
    const [owner, batcher] = await ethers.getSigners();
    const factory = await ethers.getContractFactory("AirProofAnchor");
    const deployStarted = performance.now();
    const contract = await factory.deploy();
    const deployment = await contract.deploymentTransaction().wait();
    await contract.waitForDeployment();
    const deployLatencyMs = performance.now() - deployStarted;
    const schema = ethers.id("schema:fullpaper30h:v1");
    const policy = ethers.id("policy:fullpaper30h:v1");
    await contract.connect(owner).registerPolicy(policy, schema);
    await contract.connect(owner).setBatcher(batcher.address, true);
    const block = await ethers.provider.getBlock("latest");
    const epoch = Math.floor(Number(block.timestamp) / 3600);
    const anchors = [];
    for (let replicate = 0; replicate < 30; replicate += 1) {
      const started = performance.now();
      const transaction = await contract.connect(batcher).anchorBatch(
        ethers.id(`city:${replicate}`),
        epoch,
        ethers.id(`root:${replicate}`),
        512,
        schema,
        policy,
        ethers.id(`cid:${replicate}`),
      );
      const receipt = await transaction.wait();
      anchors.push({
        replicate,
        gas_used: Number(receipt.gasUsed),
        latency_ms: performance.now() - started,
        transaction_hash: receipt.hash,
      });
    }
    expect(anchors).to.have.length(30);
    expect(anchors.every((row) => row.gas_used > 0)).to.equal(true);
    const report = {
      environment: "hardhat-local-evm",
      repetitions: anchors.length,
      deploy_gas: Number(deployment.gasUsed),
      deploy_latency_ms: deployLatencyMs,
      compiler: "0.8.28",
      optimizer_runs: 200,
      contract_address: await contract.getAddress(),
      anchors,
    };
    const output = process.env.AIRPROOF_BENCHMARK_OUTPUT;
    if (output) {
      fs.mkdirSync(path.dirname(output), { recursive: true });
      fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    }
    console.log(JSON.stringify({ benchmark: "anchorBatch30", ...report }));
  });
});
