# AirProof contract

`src/AirProofAnchor.sol` commits only city/epoch batch roots, counts, schema/policy hashes and a
high-entropy content digest. Events deliberately contain no pseudonym, nullifier, exact
location or raw observation. The contract proves post-anchor byte integrity only; it does
not claim measurement truth, completeness, availability or timely inclusion.

Local build and deterministic tests:

```powershell
npm install
npm run compile
npm test
```

Public-testnet deployment is intentionally not automatic. A deployment must explicitly
supply a testnet RPC endpoint and funded test account, record chain ID/address/transaction
hash/bytecode hash, and reject Ethereum mainnet (`chainId=1`).
