"""
Settings management for MLOps control plane.

Provides configuration loading from JSON files and environment variables,
with strict validation and immutability.
"""

import json
import os
import stat
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    field_validator,
)
from pydantic import StrictBool, StrictInt
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import EnvSettingsSource

from mlops.kernel import ValidationFailed, content_id

MIB = 1024 * 1024
MAX_FILE_SIZE = MIB
MAX_SECRET_LEN = 1024
MAX_DB_PATH_LEN = 4096


class TokenSpec(BaseModel):
    """Specification for an API token."""

    name: str = Field(..., min_length=1, max_length=64)
    role: Literal["admin", "operator", "viewer"]

    model_config = {"extra": "forbid", "frozen": True}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate token name: 1..64 chars, no control chars."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty and not whitespace-only")
        # Check for control characters (0x00-0x1f, 0x7f)
        if any(ord(c) < 0x20 or ord(c) == 0x7f for c in v):
            raise ValueError("name must not contain control characters")
        return v


class Settings(BaseSettings):
    """MLOps control plane settings."""

    audit_secret: SecretStr = Field(..., min_length=16, max_length=MAX_SECRET_LEN, repr=False, exclude=True)
    db_path: str = Field(default=":memory:", max_length=MAX_DB_PATH_LEN)
    tokens: dict[str, TokenSpec] = Field(..., min_length=1, max_length=100, repr=False, exclude=True)
    max_body_bytes: StrictInt = Field(default=1_048_576, ge=1, le=64 * MIB)
    idempotency_ttl_s: StrictInt = Field(default=3600, ge=1)
    env_name: Literal["dev", "staging", "prod"] = "dev"
    # Off by default (matches existing tested behavior: /metrics returns 401 with no token).
    # A real Prometheus scrape config (e.g. a Helm-managed kube-prometheus-stack ServiceMonitor)
    # typically has no bearer token wired in; set MLOPS_METRICS_PUBLIC=true to let it scrape
    # unauthenticated. Never affects any other route.
    # Plain bool (not StrictBool): pydantic-settings' own env-string parsing ("true"/"false")
    # needs to run for MLOPS_METRICS_PUBLIC, which strict mode would block.
    metrics_public: bool = False
    # On by default: build_default_wiring gives ModelRegistry the active policy as its
    # decision point, so a transition to production (and a rollback) is denied unless the
    # active policy allows it; with no active policy it is denied (fail-closed). Set
    # MLOPS_POLICY_ENFORCE_TRANSITIONS=false to restore ungated transitions.
    policy_enforce_transitions: bool = True
    # Periodic worker cadences used by build_default_wiring() (svc/app.py) when the
    # service actually runs -- plain float (not StrictFloat) so the manual env-var
    # whitelist below can pre-convert an MLOPS_* string with ordinary float(), the
    # same way idempotency_ttl_s pre-converts with int() rather than relying on
    # pydantic-settings' own (Strict-blocked) string coercion.
    incident_escalation_interval_s: float = Field(default=60.0, gt=0, le=3600)
    drift_evaluation_interval_s: float = Field(default=300.0, gt=0, le=3600)
    # Off by default (None): model registration/transition never mirrors to MLflow
    # unless explicitly given an absolute sqlite:/// URI. MlflowBridge itself
    # validates the URI shape; this setting only decides whether the mirror runs
    # at all. The domain tracker/registry remain authoritative either way -- a
    # mirror failure never blocks or changes the real registration/transition.
    mlflow_tracking_uri: Optional[str] = Field(default=None, max_length=MAX_DB_PATH_LEN)

    model_config = SettingsConfigDict(
        env_prefix="MLOPS_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v: str) -> str:
        """Validate db_path: no control chars."""
        if any(ord(c) < 0x20 or ord(c) == 0x7f for c in v):
            raise ValueError("db_path must not contain control characters")
        return v

    @field_validator("tokens", mode="after")
    @classmethod
    def validate_tokens(cls, v: dict[str, TokenSpec]) -> dict[str, TokenSpec]:
        """Validate token keys are properly formatted."""
        if not v:
            raise ValueError("at least one token is required")
        if len(v) > 100:
            raise ValueError("at most 100 tokens allowed")

        for token_str in v.keys():
            # Check length: 8..128 chars
            if not (8 <= len(token_str) <= 128):
                raise ValueError(f"token length invalid")
            # Check for whitespace (space, tab, newline, etc.)
            if any(c.isspace() for c in token_str):
                raise ValueError("token has whitespace")
            # Check for control characters
            if any(ord(c) < 32 or ord(c) == 127 for c in token_str):
                raise ValueError("token has control characters")

        return v

    def __repr__(self) -> str:
        """Custom repr that excludes secrets and tokens."""
        return (
            f"Settings("
            f"db_path={self.db_path!r}, "
            f"max_body_bytes={self.max_body_bytes!r}, "
            f"idempotency_ttl_s={self.idempotency_ttl_s!r}, "
            f"env_name={self.env_name!r}"
            f")"
        )

    def __str__(self) -> str:
        """Custom str that excludes secrets and tokens."""
        return self.__repr__()

    def model_dump_json(self, **kwargs: Any) -> str:
        """Override model_dump_json to exclude secrets and tokens."""
        # Use exclude to remove sensitive fields
        if "exclude" not in kwargs:
            kwargs["exclude"] = {"audit_secret", "tokens"}
        return super().model_dump_json(**kwargs)


def load_settings(
    file: str | os.PathLike | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """
    Load settings from a JSON file and environment variables.

    Args:
        file: Path to JSON settings file (optional).
        env: Environment variable mapping. If None, uses os.environ.
             If an explicit mapping (even empty), it replaces os.environ.

    Returns:
        Loaded Settings object (frozen/immutable).

    Raises:
        ValidationFailed: On any validation error. Never includes secrets
            or token strings in error messages.
    """
    # Use os.environ if env is not provided, otherwise use the explicit mapping
    if env is None:
        env_source = dict(os.environ)
    else:
        env_source = dict(env)

    # Start with env vars only
    env_config = {}

    # Parse environment variables into a config dict
    for key, value in env_source.items():
        if key.startswith("MLOPS_"):
            # Remove prefix and lowercase the rest for field lookup
            field_name = key[6:].lower()

            # Handle fields
            if field_name == "audit_secret":
                env_config["audit_secret"] = value
            elif field_name == "db_path":
                env_config["db_path"] = value
            elif field_name == "max_body_bytes":
                try:
                    env_config["max_body_bytes"] = int(value)
                except ValueError:
                    raise ValidationFailed(
                        "invalid settings: max_body_bytes (value_error)"
                    )
            elif field_name == "idempotency_ttl_s":
                try:
                    env_config["idempotency_ttl_s"] = int(value)
                except ValueError:
                    raise ValidationFailed(
                        "invalid settings: idempotency_ttl_s (value_error)"
                    )
            elif field_name == "env_name":
                env_config["env_name"] = value
            elif field_name == "incident_escalation_interval_s":
                try:
                    env_config["incident_escalation_interval_s"] = float(value)
                except ValueError:
                    raise ValidationFailed(
                        "invalid settings: incident_escalation_interval_s (value_error)"
                    )
            elif field_name == "drift_evaluation_interval_s":
                try:
                    env_config["drift_evaluation_interval_s"] = float(value)
                except ValueError:
                    raise ValidationFailed(
                        "invalid settings: drift_evaluation_interval_s (value_error)"
                    )
            elif field_name == "mlflow_tracking_uri":
                env_config["mlflow_tracking_uri"] = value
            elif field_name == "metrics_public":
                # load_settings disables pydantic-settings' own env reading, so any
                # MLOPS_* field missing from this whitelist is silently ignored.
                lowered = value.strip().lower()
                if lowered in ("true", "1", "yes"):
                    env_config["metrics_public"] = True
                elif lowered in ("false", "0", "no"):
                    env_config["metrics_public"] = False
                else:
                    raise ValidationFailed("invalid settings: metrics_public (value_error)")
            elif field_name == "policy_enforce_transitions":
                lowered = value.strip().lower()
                if lowered in ("true", "1", "yes"):
                    env_config["policy_enforce_transitions"] = True
                elif lowered in ("false", "0", "no"):
                    env_config["policy_enforce_transitions"] = False
                else:
                    raise ValidationFailed("invalid settings: policy_enforce_transitions (value_error)")
            elif field_name == "tokens":
                try:
                    env_config["tokens"] = json.loads(value)
                except json.JSONDecodeError:
                    raise ValidationFailed(
                        "invalid settings: tokens (json_error)"
                    )

    # Load file config if provided
    file_config = {}
    if file is not None:
        try:
            file_path = Path(file)

            # Get file stats and check it's a regular file
            try:
                file_stat = file_path.stat()
            except OSError:
                raise ValidationFailed("configuration file not found")

            # Ensure it's a regular file (not device, symlink, etc.)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValidationFailed("configuration file is not a regular file")

            # Check file size before reading
            if file_stat.st_size > MAX_FILE_SIZE:
                raise ValidationFailed("configuration file exceeds 1 MiB")

            # Read file with bounded size check
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read(MAX_FILE_SIZE + 1)
            except OSError:
                raise ValidationFailed("cannot read configuration file")

            # Verify we didn't exceed the limit (read more than MAX_FILE_SIZE)
            if len(file_bytes) > MAX_FILE_SIZE:
                raise ValidationFailed("configuration file exceeds 1 MiB")

            # Decode UTF-8
            try:
                file_contents = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raise ValidationFailed("configuration file is not valid UTF-8")

            # Parse JSON
            try:
                data = json.loads(file_contents)
            except json.JSONDecodeError:
                raise ValidationFailed("configuration file is not valid JSON")

            # Ensure it's a dict (object), not an array or primitive
            if not isinstance(data, dict):
                raise ValidationFailed("configuration file must be a JSON object")

            file_config = data

        except FileNotFoundError:
            raise ValidationFailed("configuration file not found")

    # Merge: defaults < file < env (env wins)
    merged = {}
    merged.update(file_config)  # file overrides defaults
    merged.update(env_config)   # env overrides file

    # Validate and create Settings with merged config
    # Fallback: override settings_customise_sources to use only init_settings
    try:
        # Save the original settings_customise_sources
        original_sources = Settings.settings_customise_sources

        # Override to use only init_settings (no env reading)
        @classmethod
        def _init_only_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
            return (init_settings,)

        Settings.settings_customise_sources = _init_only_sources

        try:
            # Now when we call __init__, it will only use init_settings, not os.environ
            settings = Settings(**merged, _env_file=None)
        finally:
            # Restore the original
            Settings.settings_customise_sources = original_sources

    except Exception as e:
        # Build a safe error message from pydantic errors
        error_msg = _build_safe_error_message(e)
        raise ValidationFailed(error_msg)

    return settings


def _build_safe_error_message(exc: Exception) -> str:
    """
    Build a safe error message from a pydantic ValidationError.

    Replaces token strings with "<token>" and avoids echoing values.
    """
    # Try to extract errors from pydantic ValidationError
    if hasattr(exc, "errors") and callable(exc.errors):
        try:
            errors = exc.errors()
            if errors:
                field_issues = []
                for err in errors:
                    loc = err.get("loc", ())
                    error_type = err.get("type", "error")

                    # Build field path, masking token strings
                    path_parts = []
                    for i, loc_part in enumerate(loc):
                        # If first element is "tokens", next element is the token string
                        if i == 0 and loc_part == "tokens":
                            path_parts.append("tokens")
                        elif i == 1 and len(loc) > i and loc[0] == "tokens":
                            # This is the token string key, mask it
                            path_parts.append("<token>")
                        else:
                            path_parts.append(str(loc_part))

                    field_path = ".".join(path_parts) if path_parts else "unknown"
                    field_issues.append(f"{field_path} ({error_type})")

                if field_issues:
                    return f"invalid settings: {', '.join(field_issues)}"
        except Exception:
            pass

    # Fallback: extract just the class name or a generic message
    error_str = str(exc)
    # Don't include the full error message as it may contain values
    if "validation" in error_str.lower():
        return "invalid settings: validation error"
    elif "value" in error_str.lower():
        return "invalid settings: value error"
    else:
        return "invalid settings"


def config_hash(settings: Settings) -> str:
    """
    Generate a stable hash of the configuration.

    The hash is computed from non-secret fields only (includes token names/roles
    but NOT token strings or the secret value). Stable under key order changes.
    Hash is prefixed with 'cfg_'.

    Args:
        settings: Settings object to hash.

    Returns:
        String hash starting with 'cfg_' followed by 16 hex characters.
    """
    # Build a normalized list of token specs (name, role pairs) sorted by name
    # to ensure stability. Do NOT include the token strings (keys).
    token_specs = sorted(
        [{"name": spec.name, "role": spec.role} for spec in settings.tokens.values()],
        key=lambda x: (x["name"], x["role"]),
    )

    # Build a normalized dict with only non-secret fields
    config_data = {
        "env_name": settings.env_name,
        "db_path": settings.db_path,
        "max_body_bytes": settings.max_body_bytes,
        "idempotency_ttl_s": settings.idempotency_ttl_s,
        "token_specs": token_specs,
        # audit_secret is NOT included
        # token strings (keys) are NOT included
    }

    # Use kernel's content_id for deterministic hashing
    hash_id = content_id("cfg", config_data)
    return hash_id
