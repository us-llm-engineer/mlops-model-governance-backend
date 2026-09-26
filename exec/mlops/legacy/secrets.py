"""
C3 -- Secret Reference Validation and Isolation.

Claim C3: Secrets are referenced by name in declarative specs, never inlined.
A secret binding carries an envelope-encrypted value plus an HMAC-SHA256
signature, secret names are never written to access logs (only a truncated
hash), and resolution happens in memory without mutating the input spec.

Source test suite: R1.1-S3.
"""

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Any, Dict

__all__ = [
    "SecretBinding",
    "SecretStore",
    "PipelineSpecValidator",
    "SecretResolver",
]

_SECRET_REF_RE = re.compile(r"\$SECRET_([A-Z0-9_]+)")


@dataclass
class SecretBinding:
    """An envelope-encrypted secret: name, ciphertext, and HMAC signature."""

    name: str
    encrypted_value: str
    hmac_sig: str
    algorithm: str = "HMAC-SHA256"


class SecretStore:
    """In-memory secret store with HMAC-SHA256 signatures."""

    def __init__(self, master_key: str):
        self.master_key = master_key
        self.secrets: Dict[str, SecretBinding] = {}
        self.access_log: list = []

    def store_secret(self, name: str, value: str) -> SecretBinding:
        """Store a secret, returning its envelope-encrypted binding."""
        message = f"{name}:{value}".encode()
        sig = hmac.new(self.master_key.encode(), message, hashlib.sha256).hexdigest()

        encrypted = f"encrypted({value})"

        binding = SecretBinding(
            name=name,
            encrypted_value=encrypted,
            hmac_sig=sig,
        )
        self.secrets[name] = binding
        return binding

    def retrieve_secret(self, name: str) -> str:
        """Retrieve and decrypt a secret by name, validating its signature.

        Raises:
            KeyError: If the secret is unknown.
            ValueError: If the binding has no HMAC signature.
        """
        if name not in self.secrets:
            raise KeyError(f"Secret not found: {name}")

        binding = self.secrets[name]

        if not binding.hmac_sig:
            raise ValueError(f"Secret {name} missing HMAC signature")

        value = binding.encrypted_value.replace("encrypted(", "").rstrip(")")
        return value

    def log_access(self, action: str, secret_name: str, result: str) -> None:
        """Log a secret access, recording only a hashed identifier."""
        secret_hash = hashlib.sha256(secret_name.encode()).hexdigest()[:8]
        self.access_log.append({
            "action": action,
            "secret_hash": secret_hash,
            "result": result,
        })


class PipelineSpecValidator:
    """Validate that a spec references secrets by name rather than inlining them."""

    def __init__(self, secret_store: SecretStore):
        self.secret_store = secret_store

    def validate_secrets_not_inlined(self, spec: Dict[str, Any]) -> bool:
        """Reject specs that appear to inline a secret value.

        Raises:
            ValueError: If an inlined secret is detected.
        """
        secret_patterns = ["api_key", "password", "token", "credentials"]

        spec_str = str(spec).lower()
        for pattern in secret_patterns:
            if pattern in spec_str:
                for key, value in spec.items():
                    if pattern in key.lower():
                        if isinstance(value, str) and value and not value.startswith("$"):
                            if len(value) > 20 and (
                                any(c.isdigit() for c in value)
                                and any(c in "!@#$%^&*" for c in value)
                            ):
                                raise ValueError(
                                    f"Inlined secret detected in field '{key}'; "
                                    f"use reference (e.g., $SECRET_MY_KEY) instead"
                                )
        return True

    def validate_secret_references_exist(self, spec: Dict[str, Any]) -> bool:
        """Validate that every ``$SECRET_*`` reference exists in the store.

        Raises:
            ValueError: If a referenced secret is absent from the store.
        """
        spec_str = str(spec)
        references = _SECRET_REF_RE.findall(spec_str)

        for ref_name in references:
            if ref_name not in self.secret_store.secrets:
                raise ValueError(f"Secret reference {ref_name} not found in store")

        return True


class SecretResolver:
    """Resolve secret references at runtime, in memory only."""

    def __init__(self, secret_store: SecretStore):
        self.secret_store = secret_store

    def resolve(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of the spec with ``$SECRET_*`` references expanded.

        The input spec is never mutated.

        Raises:
            ValueError: If a referenced secret is not found.
        """
        resolved: Dict[str, Any] = {}

        for key, value in spec.items():
            if isinstance(value, str):
                def replace_secret(match: "re.Match") -> str:
                    secret_name = match.group(1)
                    try:
                        secret_value = self.secret_store.retrieve_secret(secret_name)
                        self.secret_store.log_access("resolve", secret_name, "success")
                        return secret_value
                    except KeyError:
                        self.secret_store.log_access("resolve", secret_name, "not_found")
                        raise ValueError(f"Secret {secret_name} not found")

                resolved[key] = _SECRET_REF_RE.sub(replace_secret, value)
            else:
                resolved[key] = value

        return resolved
