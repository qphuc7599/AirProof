import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import multiprocessing
import os
import sqlite3

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from airproof.ingestion import IngestionAuthority, sign_ingress
from airproof.integrity import AtomicNullifierStore
from airproof.records import Observation, canonical_json, raw_nullifier


def example(epoch=0):
    nullifier = raw_nullifier(bytes(range(32)), city="city", policy_id="policy",
                              group=0, epoch=epoch, interval=epoch)
    return Observation(1, epoch, 0, 0, 12 + epoch, 2, 1, 512, nullifier, epoch, epoch)


@pytest.mark.parametrize("frontends", [2, 4, 8])
def test_separate_connections_share_exactly_once_numerical_effects(tmp_path, frontends):
    key = Ed25519PrivateKey.generate()
    database = tmp_path / "authority.sqlite"
    authorities = [IngestionAuthority(database, city="city", policy_id="policy",
                    verification_keys={1: key.public_key()}, credential_secrets={1: bytes(range(32))})
                   for _ in range(frontends)]
    request = sign_ingress(example(), key, city="city", policy_id="policy")

    def client(index):
        authority = authorities[index % frontends]
        won = authority.submit(request)
        authority.store.apply_pending()
        return won

    with ThreadPoolExecutor(max_workers=frontends) as pool:
        accepted = list(pool.map(client, range(frontends * 20)))
    assert sum(accepted) == 1
    assert authorities[0].store.journal_metrics() == {
        "accepted_unique": 1, "pending": 0, "applied_effects": 1,
        "duplicate_influence_count": 0, "precision": .25, "weighted_sum": 3.,
    }
    assert authorities[0].store.information_snapshot() == [("policy", "raw", 0, 0, .25, 3., 1)]
    for authority in authorities:
        authority.close()


def test_frontend_cannot_forge_value_domain_or_nullifier(tmp_path):
    key = Ed25519PrivateKey.generate()
    authority = IngestionAuthority(tmp_path / "authority.sqlite", city="city", policy_id="policy",
                                   verification_keys={1: key.public_key()},
                                   credential_secrets={1: bytes(range(32))})
    request = sign_ingress(example(), key, city="city", policy_id="policy")
    payload = json.loads(base64.b64decode(request["payload"]))
    payload["observation"]["value"] = 1e6
    forged = {**request, "payload": base64.b64encode(canonical_json(payload)).decode()}
    with pytest.raises(InvalidSignature):
        authority.submit(forged)
    with pytest.raises(ValueError, match="nullifier"):
        authority.submit(sign_ingress(replace(example(), nullifier="nonce-chosen-by-sender"),
                                      key, city="city", policy_id="policy"))
    with pytest.raises(ValueError, match="domain"):
        authority.submit(sign_ingress(example(), key, city="other-city", policy_id="policy"))
    assert authority.store.journal_metrics()["accepted_unique"] == 0
    authority.close()


def test_acceptance_and_payload_roll_back_together(tmp_path):
    store = AtomicNullifierStore(tmp_path / "atomic.sqlite")
    store.enable_journal()
    store._connection.execute("CREATE TEMP TRIGGER interrupt_payload BEFORE INSERT ON accepted_payloads "
                              "BEGIN SELECT RAISE(ABORT, 'simulated failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.accept_payload(example().nullifier, canonical_json(asdict(example())), policy_id="policy")
    assert store._connection.execute("SELECT COUNT(*) FROM accepted_v2").fetchone()[0] == 0
    store._connection.execute("DROP TRIGGER interrupt_payload")
    assert store.accept_payload(example().nullifier, canonical_json(asdict(example())), policy_id="policy")
    store.close()
    recovered = AtomicNullifierStore(tmp_path / "atomic.sqlite")
    recovered.enable_journal()
    assert recovered.apply_pending() == 1
    assert recovered.apply_pending() == 0
    recovered.close()


def crash_during_uncommitted_effect(database):
    connection = sqlite3.connect(database, isolation_level=None)
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("INSERT INTO information_state VALUES('policy','raw',0,0,.25,3,1)")
    os._exit(17)


def test_process_crash_cannot_commit_an_effect_without_its_marker(tmp_path):
    database = tmp_path / "crash.sqlite"
    store = AtomicNullifierStore(database)
    store.enable_journal()
    store.accept_payload(example().nullifier, canonical_json(asdict(example())), policy_id="policy")
    store.close()
    process = multiprocessing.get_context("spawn").Process(target=crash_during_uncommitted_effect,
                                                           args=(str(database),))
    process.start()
    process.join(20)
    assert process.exitcode == 17
    recovered = AtomicNullifierStore(database)
    recovered.enable_journal()
    assert recovered.information_snapshot() == []
    assert recovered.apply_pending() == 1
    assert recovered.journal_metrics()["duplicate_influence_count"] == 0
    assert recovered.journal_metrics()["applied_effects"] == 1
    recovered.close()
