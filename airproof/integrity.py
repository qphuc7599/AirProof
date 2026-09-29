from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def leaf_hash(payload: bytes) -> bytes:
    return _hash(b"\x00" + payload)


def node_hash(left: bytes, right: bytes) -> bytes:
    return _hash(b"\x01" + left + right)


EMPTY_ROOT = _hash(b"")


@dataclass(frozen=True)
class MerkleProof:
    index: int
    size: int
    siblings: tuple[tuple[str, str], ...]


class MerkleTree:
    """Deterministic, domain-separated binary Merkle tree with explicit cardinality."""

    def __init__(self, payloads: Iterable[bytes]):
        ordered = sorted(payloads)
        self.payloads = tuple(ordered)
        self.levels: list[list[bytes]] = [[leaf_hash(item) for item in ordered]]
        while self.levels and len(self.levels[-1]) > 1:
            current = self.levels[-1]
            nxt: list[bytes] = []
            for index in range(0, len(current), 2):
                left = current[index]
                right = current[index + 1] if index + 1 < len(current) else left
                nxt.append(node_hash(left, right))
            self.levels.append(nxt)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0] if self.levels and self.levels[0] else EMPTY_ROOT

    @property
    def root_hex(self) -> str:
        return self.root.hex()

    def proof(self, index: int) -> MerkleProof:
        if not 0 <= index < len(self.payloads):
            raise IndexError(index)
        siblings: list[tuple[str, str]] = []
        cursor = index
        for level in self.levels[:-1]:
            sibling_index = cursor ^ 1
            if sibling_index >= len(level):
                sibling_index = cursor
            side = "left" if sibling_index < cursor else "right"
            siblings.append((side, level[sibling_index].hex()))
            cursor //= 2
        return MerkleProof(index=index, size=len(self.payloads), siblings=tuple(siblings))


def verify_proof(payload: bytes, proof: MerkleProof, expected_root: bytes) -> bool:
    if proof.size <= 0 or not 0 <= proof.index < proof.size:
        return False
    expected_sides: list[str] = []
    index = proof.index
    size = proof.size
    while size > 1:
        sibling_index = index ^ 1
        if sibling_index >= size:
            sibling_index = index
        expected_sides.append("left" if sibling_index < index else "right")
        index //= 2
        size = (size + 1) // 2
    if [side for side, _ in proof.siblings] != expected_sides:
        return False
    value = leaf_hash(payload)
    for side, sibling_hex in proof.siblings:
        sibling = bytes.fromhex(sibling_hex)
        value = node_hash(sibling, value) if side == "left" else node_hash(value, sibling)
    return value == expected_root


class AtomicNullifierStore:
    """SQLite UNIQUE constraint supplies process-safe first-writer-wins semantics.

    Separate instances must use the SAME local file for shared acceptance. The
    default in-memory store is deliberately process-local, not distributed dedup.
    Only the protected acceptance authority may write the database. Frontend
    compromise does not imply compromise of this authority or SQLite storage.
    """

    def __init__(self, path: str | Path = ":memory:"):
        self._connection = sqlite3.connect(str(path), timeout=30, check_same_thread=False, isolation_level=None)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS accepted_v2 (
                policy_id TEXT NOT NULL,
                release_domain TEXT NOT NULL,
                nullifier TEXT NOT NULL,
                PRIMARY KEY(policy_id, release_domain, nullifier)
            )
            """
        )
        self._lock = threading.Lock()

    def enable_journal(self) -> None:
        """Durable accepted-input journal plus exactly-once information effects.

        An in-memory numerical twin is reconstructed from unique journal entries;
        it must never assimilate a frontend's uncommitted acceptance response.
        """
        with self._lock:
            self._connection.executescript("""
                CREATE TABLE IF NOT EXISTS accepted_payloads (
                    policy_id TEXT NOT NULL, release_domain TEXT NOT NULL,
                    nullifier TEXT NOT NULL, payload BLOB NOT NULL,
                    applied INTEGER NOT NULL DEFAULT 0 CHECK(applied IN (0,1)),
                    PRIMARY KEY(policy_id, release_domain, nullifier),
                    FOREIGN KEY(policy_id, release_domain, nullifier)
                      REFERENCES accepted_v2(policy_id, release_domain, nullifier)
                );
                CREATE TABLE IF NOT EXISTS information_state (
                    policy_id TEXT NOT NULL, release_domain TEXT NOT NULL,
                    epoch INTEGER NOT NULL, cell INTEGER NOT NULL,
                    precision REAL NOT NULL, weighted_sum REAL NOT NULL,
                    contribution_count INTEGER NOT NULL,
                    PRIMARY KEY(policy_id, release_domain, epoch, cell)
                );
            """)

    def accept_payload(self, nullifier: str, payload: bytes, *, policy_id: str,
                       release_domain: str = "raw") -> bool:
        """Commit acceptance AND its authenticated payload, or neither.

        The caller is the signature/credential-verifying authority. Frontends may
        not call this trusted-storage API directly. Replays cannot rewrite payloads.
        """
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._connection.execute(
                    "INSERT OR IGNORE INTO accepted_v2 VALUES (?, ?, ?)",
                    (policy_id, release_domain, nullifier),
                )
                if cursor.rowcount == 1:
                    self._connection.execute(
                        "INSERT INTO accepted_payloads(policy_id,release_domain,nullifier,payload) VALUES(?,?,?,?)",
                        (policy_id, release_domain, nullifier, payload),
                    )
                self._connection.execute("COMMIT")
                return cursor.rowcount == 1
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def apply_pending(self, limit: int = 1000) -> int:
        """Apply additive information contributions atomically with their markers.

        This is the information-form numerical-effect witness, not a claim that
        an arbitrary external callback becomes exactly-once. A graph solve reads
        this committed state or replays the unique journal; it does not process
        transport duplicates. A crash before COMMIT leaves both effects pending.
        """
        applied = 0
        for _ in range(limit):
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    row = self._connection.execute(
                        "SELECT policy_id,release_domain,nullifier,payload FROM accepted_payloads "
                        "WHERE applied=0 ORDER BY rowid LIMIT 1"
                    ).fetchone()
                    if row is None:
                        self._connection.execute("COMMIT")
                        return applied
                    policy, domain, nullifier, payload = row
                    item = json.loads(payload)
                    value, quality, sigma = float(item["value"]), float(item["quality"]), float(item["sigma"])
                    if not all(math.isfinite(v) for v in (value, quality, sigma)) or sigma <= 0 or quality < 0:
                        raise ValueError("invalid authenticated journal observation")
                    precision = quality / sigma**2
                    if not math.isfinite(precision) or not math.isfinite(precision * value):
                        raise ValueError("non-finite information contribution")
                    self._connection.execute(
                        "INSERT INTO information_state VALUES(?,?,?,?,?,?,1) "
                        "ON CONFLICT(policy_id,release_domain,epoch,cell) DO UPDATE SET "
                        "precision=precision+excluded.precision, weighted_sum=weighted_sum+excluded.weighted_sum, "
                        "contribution_count=contribution_count+1",
                        (policy, domain, int(item["epoch"]), int(item["cell"]), precision, precision * value),
                    )
                    self._connection.execute(
                        "UPDATE accepted_payloads SET applied=1 WHERE policy_id=? AND release_domain=? AND nullifier=?",
                        (policy, domain, nullifier),
                    )
                    self._connection.execute("COMMIT")
                    applied += 1
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
        return applied

    def journal_metrics(self) -> dict[str, int | float]:
        with self._lock:
            self._connection.execute("BEGIN")
            try:
                accepted, pending = self._connection.execute(
                    "SELECT COUNT(*),COALESCE(SUM(1-applied),0) FROM accepted_payloads"
                ).fetchone()
                effects, precision, weighted_sum = self._connection.execute(
                    "SELECT COALESCE(SUM(contribution_count),0),COALESCE(SUM(precision),0),"
                    "COALESCE(SUM(weighted_sum),0) FROM information_state"
                ).fetchone()
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return {"accepted_unique": accepted, "pending": pending, "applied_effects": effects,
                "duplicate_influence_count": max(0, effects - accepted),
                "precision": precision, "weighted_sum": weighted_sum}

    def information_snapshot(self) -> list[tuple]:
        with self._lock:
            return self._connection.execute(
                "SELECT policy_id,release_domain,epoch,cell,precision,weighted_sum,contribution_count "
                "FROM information_state ORDER BY policy_id,release_domain,epoch,cell"
            ).fetchall()

    def accept_once(
        self,
        nullifier: str,
        *,
        policy_id: str = "default-policy",
        release_domain: str = "raw",
    ) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO accepted_v2(policy_id, release_domain, nullifier)
                VALUES (?, ?, ?)
                """,
                (policy_id, release_domain, nullifier),
            )
            return cursor.rowcount == 1

    def close(self) -> None:
        self._connection.close()
