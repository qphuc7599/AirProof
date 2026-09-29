"""Live loopback-HTTP fault matrix: 2/4/8 stateless frontends, protected shared DB."""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from airproof.experiment import source_tree_digest
from airproof.ingestion import IngestionAuthority, sign_ingress
from airproof.records import Observation, canonical_json, raw_nullifier


def run_case(output, frontends, fault, trials):
    key = Ed25519PrivateKey.generate()
    secret = bytes(range(32))
    database = output / f"frontends-{frontends}-{fault}.sqlite"
    authorities = [IngestionAuthority(database, city="synthetic", policy_id="concurrency-v4",
                    verification_keys={0: key.public_key()}, credential_secrets={0: secret})
                   for _ in range(frontends)]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 16384:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(size))
                index = int(self.headers["X-Frontend"]) % frontends
                accepted = authorities[index].submit(request)
                authorities[index].store.apply_pending()
                body = json.dumps({"accepted": accepted}).encode()
                self.send_response(200)
            except Exception:
                body = b'{"rejected":true}'
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    url = f"http://127.0.0.1:{server.server_port}/ingest"
    requests = []
    for epoch in range(trials):
        nullifier = raw_nullifier(secret, city="synthetic", policy_id="concurrency-v4",
                                  group=0, epoch=epoch, interval=epoch)
        observation = Observation(0, epoch, 0, 0, 12 + epoch / 100, 2, 1, 512,
                                  nullifier, epoch, epoch)
        requests.append(sign_ingress(observation, key, city="synthetic", policy_id="concurrency-v4"))
    barrier = threading.Barrier(frontends)

    def frontend(index):
        rng = random.Random(1000 + index)
        accepted, rejected, attempted = 0, 0, 0

        def post(request):
            nonlocal accepted, rejected, attempted
            attempted += 1
            try:
                message = Request(url, data=json.dumps(request).encode(), headers={"X-Frontend": str(index),
                                  "Content-Type": "application/json"})
                with urlopen(message, timeout=30) as response:
                    accepted += int(json.load(response).get("accepted", False))
            except HTTPError as error:
                assert error.code == 400
                rejected += 1

        for request in requests:
            barrier.wait(timeout=30)
            if fault != "simultaneous":
                time.sleep(rng.uniform(0, .015))
            post(request)
            if fault == "malicious_frontend" and index == frontends - 1:
                for _ in range(4):
                    post(request)
                payload = json.loads(base64.b64decode(request["payload"]))
                payload["observation"]["value"] += 1000
                post({**request, "payload": base64.b64encode(canonical_json(payload)).decode()})
        return {"accepted": accepted, "rejected": rejected, "attempted": attempted}

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=frontends) as pool:
            client_results = list(pool.map(frontend, range(frontends)))
        metrics = authorities[0].store.journal_metrics()
        snapshot = authorities[0].store.information_snapshot()
        # Each trial is its own (epoch,cell): directly check numerical multiplicity
        # per record, so missing effects cannot cancel duplicates in a total count.
        duplicate_influence = sum(max(0, row[6] - 1) for row in snapshot)
        numerical_mismatches = sum(
            abs(row[4] - .25) > 1e-12 or abs(row[5] - .25 * (12 + row[2] / 100)) > 1e-10
            for row in snapshot
        )
        result = {"frontends": frontends, "fault": fault, "trials": trials, **metrics,
                  "duplicate_influence_count": duplicate_influence,
                  "numerical_effect_mismatches": numerical_mismatches,
                  "accepted_responses": sum(r["accepted"] for r in client_results),
                  "rejected_forgeries": sum(r["rejected"] for r in client_results),
                  "requests": sum(r["attempted"] for r in client_results),
                  "elapsed_seconds": time.perf_counter() - started}
        result["pass"] = (len(snapshot) == trials and metrics["pending"] == 0
            and metrics["accepted_unique"] == trials and metrics["applied_effects"] == trials
            and result["accepted_responses"] == trials and duplicate_influence == 0
            and numerical_mismatches == 0
            and result["rejected_forgeries"] == (trials if fault == "malicious_frontend" else 0))
        return result
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(5)
        for authority in authorities:
            authority.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for frontends in (2, 4, 8):
        for fault in ("simultaneous", "network_jitter", "malicious_frontend"):
            row = run_case(args.output, frontends, fault, args.trials)
            rows.append(row)
            (args.output / "results.json").write_text(json.dumps({
                "source_hash": source_tree_digest(),
                "scope": "loopback-http-stateless-frontends-shared-protected-single-host-SQLite-authority",
                "storage_fault_tolerance": "transaction rollback/recovery, not BFT database compromise",
                "rows": rows, "all_pass": len(rows) == 9 and all(r["pass"] for r in rows),
            }, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
            if not row["pass"]:
                raise RuntimeError("concurrency case failed")


if __name__ == "__main__":
    main()
