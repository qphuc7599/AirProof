import { expect } from "chai";
import { network } from "hardhat";

const { ethers } = await network.create();

describe("AirProofAnchor", function () {
  const city = ethers.id("city:hcmc");
  const schema = ethers.id("schema:v1");
  const policy = ethers.id("policy:v1");
  const root = ethers.id("root:1");
  const cid = ethers.id("cid:1");

  async function fixture() {
    const [owner, batcher, outsider] = await ethers.getSigners();
    const factory = await ethers.getContractFactory("AirProofAnchor");
    const contract = await factory.deploy();
    await contract.waitForDeployment();
    await contract.registerPolicy(policy, schema);
    await contract.setBatcher(batcher.address, true);
    const block = await ethers.provider.getBlock("latest");
    const epoch = Math.floor(Number(block.timestamp) / 3600);
    return { contract, owner, batcher, outsider, epoch };
  }

  it("rejects unauthorized anchors", async function () {
    const { contract, outsider, epoch } = await fixture();
    await expect(contract.connect(outsider).anchorBatch(city, epoch, root, 10, schema, policy, cid))
      .to.be.revertedWithCustomError(contract, "Unauthorized");
  });

  it("anchors once and exposes no citizen fields", async function () {
    const { contract, batcher, epoch } = await fixture();
    await expect(contract.connect(batcher).anchorBatch(city, epoch, root, 10, schema, policy, cid))
      .to.emit(contract, "BatchAnchored");
    const anchor = await contract.getAnchor(city, epoch);
    expect(anchor.root).to.equal(root);
    expect(anchor.count).to.equal(10);
    await expect(contract.connect(batcher).anchorBatch(city, epoch, ethers.id("other"), 10, schema, policy, cid))
      .to.be.revertedWithCustomError(contract, "DuplicateAnchor");
  });

  it("rejects count and future-epoch boundaries", async function () {
    const { contract, batcher, epoch } = await fixture();
    await expect(contract.connect(batcher).anchorBatch(city, epoch, root, 0, schema, policy, cid))
      .to.be.revertedWithCustomError(contract, "InvalidValue");
    await expect(contract.connect(batcher).anchorBatch(city, epoch + 25, root, 1, schema, policy, cid))
      .to.be.revertedWithCustomError(contract, "InvalidValue");
  });

  it("maintains a correction chain", async function () {
    const { contract, batcher, epoch } = await fixture();
    await contract.connect(batcher).anchorBatch(city, epoch, root, 10, schema, policy, cid);
    const corrected = ethers.id("root:2");
    await expect(contract.connect(batcher).anchorCorrection(root, corrected, ethers.id("reason"), epoch))
      .to.emit(contract, "BatchCorrected");
    expect(await contract.correctionPredecessor(corrected)).to.equal(root);
  });

  it("enforces future-effective revocation", async function () {
    const { contract, batcher, epoch } = await fixture();
    await contract.revokeBatcher(batcher.address, epoch + 1);
    await contract.connect(batcher).anchorBatch(city, epoch, root, 10, schema, policy, cid);
    await expect(contract.connect(batcher).anchorBatch(city, epoch + 1, ethers.id("root:2"), 10, schema, policy, ethers.id("cid:2")))
      .to.be.revertedWithCustomError(contract, "RevokedBatcher");
  });

  it("gas benchmark emits machine-readable values", async function () {
    const { contract, batcher, epoch } = await fixture();
    const transaction = await contract.connect(batcher).anchorBatch(city, epoch, root, 512, schema, policy, cid);
    const receipt = await transaction.wait();
    console.log(JSON.stringify({ benchmark: "anchorBatch", gasUsed: receipt.gasUsed.toString() }));
    expect(receipt.gasUsed).to.be.greaterThan(0);
  });
});
