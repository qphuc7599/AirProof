// Local in-process EVM only. Uses the unchanged cached AirProofAnchor artifact.
import { expect } from 'chai';
import { network } from 'hardhat';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const { ethers } = await network.create();
const digest = data => crypto.createHash('sha256').update(data).digest('hex');

describe('v5 original 400-stream local EVM adapter', function () {
  this.timeout(120000);
  it('writes each original root once, reads it back and rejects conflicting writes', async function () {
    const input = path.resolve('../reports/v5/audit_supplement/streams.json');
    const inputBytes = fs.readFileSync(input);
    const streams = JSON.parse(inputBytes);
    expect(streams).to.have.length(400);
    expect(new Set(streams.map(s => `${s.batch_size}:${s.trial}`)).size).to.equal(400);
    const factory = await ethers.getContractFactory('AirProofAnchor');
    const deploymentStarted = performance.now();
    const contract = await factory.deploy();
    const deployment = await contract.deploymentTransaction().wait();
    const deployMs = performance.now()-deploymentStarted;
    const schema = ethers.id('schema:v5-audit-common');
    const policy = ethers.id('policy:v5-common');
    const registered = await (await contract.registerPolicy(policy,schema)).wait();
    const rows = [];
    for (const stream of streams) {
      // The symbolic original city is mapped to bytes32; epoch/version remain 0.
      const city = ethers.id(stream.log_id);
      const root = `0x${stream.root_hex}`;
      const fork = `0x${stream.fork_root_hex}`;
      const cid = `0x${stream.stream_sha256}`;
      const calldata = contract.interface.encodeFunctionData('anchorBatch',[city,0,root,stream.batch_size,schema,policy,cid]);
      const before = await contract.getAnchor(city,0);
      expect(before.root).to.equal(ethers.ZeroHash);
      const started = performance.now();
      const tx = await contract.anchorBatch(city,0,root,stream.batch_size,schema,policy,cid);
      const receipt = await tx.wait();
      const writeMs = performance.now()-started;
      const readStarted = performance.now();
      const anchor = await contract.getAnchor(city,0);
      const readMs = performance.now()-readStarted;
      expect(anchor.root).to.equal(root);
      expect(Number(anchor.count)).to.equal(stream.batch_size);
      expect(anchor.cidDigest).to.equal(cid);
      const conflictStarted = performance.now();
      // eth_call executes the rejection path; this is not a mined failed tx.
      await expect(contract.anchorBatch.staticCall(city,0,fork,stream.batch_size,schema,policy,cid))
        .to.be.revertedWithCustomError(contract,'DuplicateAnchor');
      const conflictMs = performance.now()-conflictStarted;
      expect((await contract.getAnchor(city,0)).root).to.equal(root);
      rows.push({...stream,city_id:city,root_readback:anchor.root,first_write_valid:true,
        conflicting_write_rejected:true,conflict_execution:'local eth_call, not mined transaction',
        gas_used:Number(receipt.gasUsed),write_ms:writeMs,read_ms:readMs,conflict_check_ms:conflictMs,
        calldata_bytes:(calldata.length-2)/2,read_call_bytes:68,read_return_abi_bytes:224,
        transaction_hash:receipt.hash,block_number:receipt.blockNumber});
    }
    const artifactPath=path.resolve('artifacts/src/AirProofAnchor.sol/AirProofAnchor.json');
    const artifact=JSON.parse(fs.readFileSync(artifactPath));
    const buildInfoPath=path.resolve(`artifacts/build-info/${artifact.buildInfoId}.json`);
    const buildInfo=JSON.parse(fs.readFileSync(buildInfoPath));
    expect(buildInfo.input.sources[artifact.inputSourceName].content).to.equal(fs.readFileSync('src/AirProofAnchor.sol','utf8'));
    const report={environment:'hardhat-in-process-local-evm',network_chain_id:Number((await ethers.provider.getNetwork()).chainId),
      streams_sha256:digest(inputBytes),contract_source_sha256:digest(fs.readFileSync('src/AirProofAnchor.sol')),
      cached_artifact_sha256:digest(fs.readFileSync(artifactPath)),build_info_sha256:digest(fs.readFileSync(buildInfoPath)),cached_build_source_matches:true,
      cached_bytecode_template_sha256:digest(Buffer.from(artifact.deployedBytecode.slice(2),'hex')),
      deployed_runtime_sha256:digest(Buffer.from((await ethers.provider.getCode(await contract.getAddress())).slice(2),'hex')),
      contract_address:await contract.getAddress(),deployment_gas:Number(deployment.gasUsed),deployment_ms:deployMs,
      policy_registration_gas:Number(registered.gasUsed),rows,
      limitations:['local automining timings, not public-chain latency or finality','original version-zero roots only; no correction-chain workload',
        'contract stores roots and does not execute Python Merkle/receipt verification','application ABI byte counts exclude transaction envelopes and RPC transport',
        'conflicting writes executed with eth_call; no failed transaction gas charge measured','no public RPC or Sepolia transactions']};
    fs.writeFileSync(path.resolve('../reports/v5/audit_supplement/local_evm.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({streams:rows.length,readbacks:rows.filter(r=>r.first_write_valid).length,
      conflicting_rejections:rows.filter(r=>r.conflicting_write_rejected).length,mean_gas:rows.reduce((a,r)=>a+r.gas_used,0)/rows.length}));
  });
});
