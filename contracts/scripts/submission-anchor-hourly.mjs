// Hourly checkpoint batches after finite receipt/terminal return.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {network} from 'hardhat';
const inputPath='../reports/submission_checks_20260927/anchoring/returned_checkpoints.json';
const bytes=fs.readFileSync(inputPath);
const input=JSON.parse(bytes);
const hash=x=>crypto.createHash('sha256').update(x).digest();
const leaf=x=>hash(Buffer.concat([Buffer.from([0]),x]));
const node=(a,b)=>hash(Buffer.concat([Buffer.from([1]),a,b]));
function split(n){let k=1;while(2*k<n)k*=2;return k;}
function root(xs){if(xs.length===1)return leaf(xs[0]);const k=split(xs.length);return node(root(xs.slice(0,k)),root(xs.slice(k)));}
function proof(xs,i){if(xs.length===1)return [];const k=split(xs.length);return i<k?[...proof(xs.slice(0,k),i),{side:'right',hash:root(xs.slice(k)).toString('hex')}]:[...proof(xs.slice(k),i-k),{side:'left',hash:root(xs.slice(0,k)).toString('hex')}];}
function verify(x,p,r){let v=leaf(x);for(const s of p){const h=Buffer.from(s.hash,'hex');v=s.side==='left'?node(h,v):node(v,h);}return v.equals(r);}
const groups=new Map();
for(const row of input.verified_receipts){
 if(!row.proof_verified)throw Error('Invalid source proof');
 const epoch=row.available_epoch+1; // close the public hourly batch
 if(!groups.has(epoch))groups.set(epoch,[]);
 groups.get(epoch).push(row);
}
const {ethers}=await network.create();
const contract=await(await ethers.getContractFactory('AirProofAnchor')).deploy();
await contract.waitForDeployment();
const schema=ethers.id('SHA256 RFC6962 over checkpoint root,count,timestamp JSON tuples');
const policy=ethers.id('AirProof public hourly returned-checkpoint batch');
const city=ethers.id(`AirProof-shared-city-${input.seed}`);
await(await contract.registerPolicy(policy,schema)).wait();
const base=Math.ceil((Number((await ethers.provider.getBlock('latest')).timestamp)+1)/3600)*3600;
const rows=[], batches=[];
for(const [epoch,items] of [...groups].sort((a,b)=>a[0]-b[0])){
 items.sort((a,b)=>a.receipt_id.localeCompare(b.receipt_id));
 const payloads=items.map(x=>Buffer.from(JSON.stringify([x.root,x.count,x.checkpoint_epoch])));
 const r=root(payloads);const anchoredRoot='0x'+r.toString('hex');
 await ethers.provider.send('evm_setNextBlockTimestamp',[base+epoch*3600]);
 const contractEpoch=base/3600+epoch;
 const tx=await contract.anchorBatch(city,contractEpoch,anchoredRoot,items.length,schema,policy,'0x'+hash(Buffer.concat(payloads)).toString('hex'));
 const receipt=await tx.wait();const readback=await contract.getAnchor(city,contractEpoch);
 if(readback.root!==anchoredRoot||Number(readback.count)!==items.length)throw Error('Readback mismatch');
 const onchainRoot=Buffer.from(readback.root.slice(2),'hex');
 for(let i=0;i<items.length;i++){
  const p=proof(payloads,i);
  if(!verify(payloads[i],p,onchainRoot))throw Error('Checkpoint inclusion failed');
  const changed=Buffer.from(payloads[i]);changed[0]^=1;
  if(verify(changed,p,onchainRoot))throw Error('Tampering accepted');
  rows.push({...items[i],anchor_epoch:epoch,anchor_root:readback.root,checkpoint_leaf:payloads[i].toString(),checkpoint_proof:p,
   onchain_inclusion_verified:true,tampered_leaf_rejected:true,
   delay_from_acceptance_epochs:epoch-items[i].receipt_accepted_epoch,
   deadline_excess_epochs:epoch-items[i].receipt_deadline_epoch});
 }
 batches.push({epoch,root:readback.root,count:items.length,gas:Number(receipt.gasUsed),
  transaction_data_bytes:(tx.data.length-2)/2,transaction_hash:receipt.hash});
}
const report={scope:'Hourly local-EVM batches of checkpoints actually delivered over the finite reverse path; no public-chain finality or chain-network simulation',
 input_sha256:hash(bytes).toString('hex'),runner_sha256:hash(fs.readFileSync(new URL(import.meta.url))).toString('hex'),
 issued_receipts:input.transport_metrics.receipt_issued,verified_receipts:rows.length,
 timely_receipts:rows.filter(x=>x.deadline_excess_epochs<=0).length,
 anchored_batches:batches.length,reverse_payload_bytes:input.transport_metrics.receipt_payload_bytes,
 reverse_control_bytes:input.transport_metrics.audit_return_control_bytes,
 chain_transaction_data_bytes:batches.reduce((a,x)=>a+x.transaction_data_bytes,0),
 gas_total:batches.reduce((a,x)=>a+x.gas,0),batches,rows};
fs.writeFileSync('../reports/submission_checks_20260927/anchoring/hourly_anchoring.json',JSON.stringify(report,null,2));
console.log(JSON.stringify({...report,batches:undefined,rows:undefined}));
