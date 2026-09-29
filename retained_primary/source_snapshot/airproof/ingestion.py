"""Protected ingestion authority behind stateless, potentially faulty frontends."""
from __future__ import annotations

import base64
from dataclasses import asdict
import json
import math

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .integrity import AtomicNullifierStore
from .records import Observation, canonical_json, raw_nullifier


def sign_ingress(observation: Observation, key: Ed25519PrivateKey, *, city: str,
                 policy_id: str, domain: str = "raw") -> dict[str, str]:
    payload = canonical_json({"city": city, "policy_id": policy_id, "domain": domain,
                              "observation": asdict(observation)})
    return {"payload": base64.b64encode(payload).decode("ascii"),
            "signature": base64.b64encode(key.sign(payload)).decode("ascii")}


class IngestionAuthority:
    """Only this trusted service, not frontends, holds verification/state access.

    The credential authority supplies a per-user nullifier secret and verification
    key. Requests bind the city, policy, domain and observation under one signature.
    SQLite is a single-host serializable acceptance authority, NOT BFT consensus.
    """
    def __init__(self, database, *, city: str, policy_id: str,
                 verification_keys: dict[int, Ed25519PublicKey], credential_secrets: dict[int, bytes]):
        self.city, self.policy_id = city, policy_id
        self.keys, self.secrets = verification_keys, credential_secrets
        self.store = AtomicNullifierStore(database)
        self.store.enable_journal()

    def submit(self, request: dict[str, str]) -> bool:
        payload = base64.b64decode(request["payload"], validate=True)
        signature = base64.b64decode(request["signature"], validate=True)
        data = json.loads(payload)
        observation = Observation(**data["observation"])
        key = self.keys[observation.user_id]
        key.verify(signature, payload)
        if data["city"] != self.city or data["policy_id"] != self.policy_id or data["domain"] != "raw":
            raise ValueError("wrong signed ingestion domain")
        if (observation.source_class != "citizen" or observation.epoch < 0 or observation.cell < 0
                or observation.group < 0 or not 0 <= observation.quality <= 1
                or not 0 < observation.sigma < math.inf or not math.isfinite(observation.value)):
            raise ValueError("invalid signed observation")
        expected = raw_nullifier(self.secrets[observation.user_id], city=self.city,
                                 policy_id=self.policy_id, group=observation.group,
                                 epoch=observation.epoch, interval=observation.epoch)
        if observation.nullifier != expected:
            raise ValueError("nullifier is not bound to the credential/domain")
        return self.store.accept_payload(expected, canonical_json(asdict(observation)),
                                         policy_id=self.policy_id, release_domain="raw")

    def close(self):
        self.store.close()
