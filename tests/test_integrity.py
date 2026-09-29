from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from airproof.integrity import EMPTY_ROOT, AtomicNullifierStore, MerkleTree, verify_proof


def test_merkle_proofs_and_tamper_detection():
    tree = MerkleTree([b"c", b"a", b"b", b"d", b"e"])
    for index, payload in enumerate(tree.payloads):
        assert verify_proof(payload, tree.proof(index), tree.root)
        assert not verify_proof(payload + b"!", tree.proof(index), tree.root)


def test_merkle_proof_direction_and_cardinality_are_bound():
    tree = MerkleTree([b"a", b"b", b"c", b"d", b"e"])
    proof = tree.proof(1)
    first_side, first_hash = proof.siblings[0]
    forged_siblings = (("left" if first_side == "right" else "right", first_hash), *proof.siblings[1:])
    assert not verify_proof(tree.payloads[1], replace(proof, siblings=forged_siblings), tree.root)
    assert not verify_proof(tree.payloads[1], replace(proof, size=1), tree.root)


def test_empty_tree_has_stable_root():
    assert MerkleTree([]).root == EMPTY_ROOT


def test_atomic_duplicate_rejection_under_concurrency():
    store = AtomicNullifierStore()
    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(lambda _: store.accept_once("n"), range(100)))
    store.close()
    assert sum(outcomes) == 1
