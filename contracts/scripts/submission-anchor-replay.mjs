import fs from 'node:fs';
import {network} from 'hardhat';
const inputPath='../reports/submission_checks_20260927/anchoring/returned_checkpoints.json';
const input=JSON.parse(fs.readFileSync(inputPath,'utf8'));
const {ethers}=await network.create();
const contract=await (await ethers.getContractFactory('AirProofAnchor')).deploy();
await contract.waitForDeployment();
const schema=ethers.id('AirProof shared checkpoint replay schema');
const policy=ethers.id('AirProof shared checkpoint replay policy');
await (await contract.registerPolicy(policy,schema)).wait();
const base=Number((await ethers.provider.getBlock('latest')).timestamp)+1;
let last=base-1;
const roots=new Map();
const rows=[];
for(const row of input.verified_receipts){
  if(!row.proof_verified)throw Error('Unverified receipt');
  let anchor=roots.get(row.root);
  if(!anchor){
    const stamp=Math.max(last+1,base+row.available_epoch*3600);
    await ethers.provider.send('evm_setNextBlockTimestamp',[stamp]);
    const city=ethers.id(`submission-replay-${input.seed}-${row.root}`);
    const epoch=Math.floor(stamp/3600);
    const root='0x'+row.root;
    const started=performance.now();
    const tx=await contract.anchorBatch(city,epoch,root,row.count,schema,policy,ethers.id(row.receipt_id));
    const receipt=await tx.wait();
    const readback=await contract.getAnchor(city,epoch);
    if(readback.root!==root || Number(readback.count)!==row.count)throw Error('Root readback mismatch');
    const block=await ethers.provider.getBlock(receipt.blockNumber);
    last=Number(block.timestamp);
    anchor={root:row.root,logical_epoch:(last-base)/3600,gas:Number(receipt.gasUsed),
      execution_ms:performance.now()-started,transaction_hash:receipt.hash,
      transaction_data_bytes:(tx.data.length-2)/2,readback_verified:true};
    roots.set(row.root,anchor);
  }
  rows.push({...row,anchor,delay_from_acceptance_epochs:anchor.logical_epoch-row.receipt_accepted_epoch,
    deadline_excess_epochs:anchor.logical_epoch-row.receipt_deadline_epoch});
}
const report={scope:'Local EVM causal replay after actual finite reverse delivery; not public-network finality',
  input_seed:input.seed,issued_receipts:input.transport_metrics.receipt_issued,
  anchored_roots:roots.size,verified_receipts:rows.length,
  verified_before_deadline:rows.filter(r=>r.deadline_excess_epochs<=0).length,
  reverse_payload_bytes:input.transport_metrics.receipt_payload_bytes,
  reverse_control_bytes:input.transport_metrics.audit_return_control_bytes,
  chain_transaction_data_bytes:[...roots.values()].reduce((a,x)=>a+x.transaction_data_bytes,0),
  gas_total:[...roots.values()].reduce((a,x)=>a+x.gas,0),rows};
fs.writeFileSync('../reports/submission_checks_20260927/anchoring/evm_replay.json',JSON.stringify(report,null,2));
console.log(JSON.stringify({...report,rows:undefined}));
