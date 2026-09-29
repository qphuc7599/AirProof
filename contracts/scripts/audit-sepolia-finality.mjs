// Read-only chain verification and consistent quantiles; no wallet or signing.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {Contract,JsonRpcProvider,Interface} from 'ethers';
const dir='../reports/submission_checks_20260927/sepolia_finality_20260927';
const read=name=>JSON.parse(fs.readFileSync(`${dir}/${name}`,'utf8'));
const rows=read('anchors.json'),manifest=read('manifest.json'),summary=read('summary.json');
const events=fs.readFileSync(`${dir}/events.jsonl`,'utf8').trim().split('\n').map(JSON.parse);
const assert=(value,message)=>{if(!value)throw Error(message);};
assert(rows.length===30&&new Set(rows.map(x=>x.transaction_hash)).size===30,'Incomplete or duplicate matrix');
assert([0,1,2].every(b=>rows.filter(x=>x.time_block===b).length===10),'Incomplete time blocks');
const p=new JsonRpcProvider('https://ethereum-sepolia-rpc.publicnode.com',11155111,{staticNetwork:true});
assert(BigInt(await p.send('eth_chainId',[]))===11155111n,'Wrong chain');
const artifact=JSON.parse(fs.readFileSync('artifacts/src/AirProofAnchor.sol/AirProofAnchor.json','utf8'));
const abi=new Interface(artifact.abi),contract=new Contract(summary.contract_address,artifact.abi,p);
const finalized=await p.getBlock('finalized');
const checked=[];
for(const row of rows){
 const [receipt,tx]=await Promise.all([p.getTransactionReceipt(row.transaction_hash),p.getTransaction(row.transaction_hash)]);
 assert(receipt&&tx&&receipt.status===1&&row.status===1,'Receipt failure');
 assert(receipt.blockHash===row.block_hash&&receipt.blockNumber===row.block_number,'Canonical block mismatch');
 assert(receipt.blockNumber<=finalized.number,'Not finalized');
 assert(tx.to.toLowerCase()===summary.contract_address.toLowerCase(),'Wrong contract');
 const decoded=abi.parseTransaction({data:tx.data,value:tx.value});
 assert(decoded.name==='anchorBatch','Wrong method');
 const [city,epoch,root,count,schema,policy,cid]=decoded.args;
 const stored=await contract.getAnchor(city,epoch);
 assert(stored.root===root&&stored.count===count&&stored.schemaHash===schema&&stored.policyHash===policy&&stored.cidDigest===cid,'Readback mismatch');
 assert(Number(receipt.gasUsed)===row.gas_used,'Gas mismatch');
 const observed=(Date.parse(row.finalized_observed_at)-Date.parse(row.signed_at))/1000;
 assert(Math.abs(observed-row.time_to_observed_finality_seconds)<1e-8,'Timestamp mismatch');
 assert(events.some(e=>e.event==='anchor_finalized'&&e.transaction_hash===row.transaction_hash
   &&Math.abs(e.time_to_observed_finality_seconds-observed)<1e-8
   &&Date.parse(e.observed_at)>=Date.parse(row.finalized_observed_at)
   &&Date.parse(e.observed_at)-Date.parse(row.finalized_observed_at)<1000),'Missing finality event');
 checked.push({transaction_hash:row.transaction_hash,canonical_receipt:true,contract_readback:true,finalized:true});
}
const quantiles=values=>{
 const x=[...values].sort((a,b)=>a-b);
 const q=p=>{const index=(x.length-1)*p,lo=Math.floor(index),f=index-lo;return x[lo]+f*(x[Math.min(lo+1,x.length-1)]-x[lo]);};
 return {minimum:q(0),median:q(.5),p95:q(.95),maximum:q(1)};
};
const report={verified_at:new Date().toISOString(),anchors:rows.length,chain_id:11155111,
 finalized_head:finalized.number,checked,quantile_convention:'linear interpolation at (n-1)*p; median averages the middle two for n=30',
 observed_finality_seconds:quantiles(rows.map(x=>x.time_to_observed_finality_seconds)),
 two_confirmation_observation_seconds:quantiles(rows.map(x=>x.time_to_receipt_seconds)),
 gas:quantiles(rows.map(x=>x.gas_used)),anchor_fees_test_eth:rows.reduce((a,x)=>a+Number(x.fee_test_eth),0),
 elapsed_seconds:summary.elapsed_seconds,rpc_failures:events.filter(x=>x.event==='rpc_failure').length,
 reverted_anchors:rows.filter(x=>x.status!==1).length,right_censored:rows.filter(x=>!x.finalized_observed_at).length,
 clock_note:'signed_at precedes signing/broadcast; time_to_receipt_seconds was recorded after two-confirmation wait and is not first-inclusion latency. Finality is first RPC observation with nominal 12 s polling, not exact consensus transition time.',
 source_sha256:Object.fromEntries(['manifest.json','anchors.json','events.jsonl','summary.json'].map(n=>[n,crypto.createHash('sha256').update(fs.readFileSync(`${dir}/${n}`)).digest('hex')]))};
fs.writeFileSync(`${dir}/verified_analysis.json`,JSON.stringify(report,null,2));
console.log(JSON.stringify({...report,checked:undefined,source_sha256:undefined}));
p.destroy();
