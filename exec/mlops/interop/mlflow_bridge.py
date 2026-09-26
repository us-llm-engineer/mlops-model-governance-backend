"""Isolated MLflow mirrors for experiment logs and model-stage aliases.

The in-process experiment tracker and model registry remain the system of
record.  This adapter mirrors their state into one explicitly chosen SQLite
MLflow store so external tooling can inspect it without changing the domain
objects' ownership or lifecycle rules.

MLflow parameters are strings on disk, so each mirrored parameter also carries
private run-tag metadata naming its original supported Python type. Retrieval
uses that declaration; it never infers a type from text such as ``"001"`` or
``"false"``. Model stages use aliases because MLflow's legacy stage field is
deprecated and unreliable in the pinned client version.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode

from mlops import kernel
from mlops.experiments import LogEntry
from mlops.model_stages import Stage

__all__ = ["MlflowBridge"]

_RUN_ID_TAG = "mlops_run_id"
_PARAM_TYPE_TAG_PREFIX = "mlops.param_type."
_MAX_PAGE_SIZE = 10_000
_STAGE_ALIASES = ("staging", "production")


def _validate_text(value: Any, label: str) -> str:
    """Require a non-empty printable string before calling MLflow."""
    if not isinstance(value, str) or not value:
        raise kernel.ValidationFailed(f"{label} must be a non-empty string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise kernel.ValidationFailed(f"{label} contains a control character")
    return value


def _mlflow_error_code(exc: MlflowException) -> Any:
    """Read the stable numeric code exposed by MLflow's public exception."""
    return getattr(exc, "error_code", None)


def _has_error_code(exc: MlflowException, expected: int, expected_name: str) -> bool:
    """Match MLflow error codes across its string and protobuf representations."""
    actual = _mlflow_error_code(exc)
    if actual == expected or getattr(actual, "name", None) == expected_name:
        return True
    if isinstance(actual, str):
        label = actual.rsplit(".", 1)[-1].upper()
        if label == expected_name:
            return True
        try:
            return int(actual) == int(expected)
        except ValueError:
            return False
    return False


def _raise_kernel_error(
    exc: MlflowException,
    operation: str,
    *,
    invalid_param_is_conflict: bool = False,
) -> None:
    """Translate an MLflow boundary failure into the platform error vocabulary.

    MLflow's exception classes are not sufficiently specific for callers: a
    missing resource, a duplicate, and invalid caller data can all surface as
    ``MlflowException``.  The provider error code plus the operation context
    preserves the distinction the control plane exposes, while ensuring no
    raw MLflow exception crosses this module's public API.
    """
    # Provider exception text can contain tracking URIs, SQL fragments, tags,
    # or caller values. Public errors therefore identify only this fixed
    # operation label and never retain the provider exception as a cause.
    message = f"MLflow {operation} failed"
    if _has_error_code(exc, ErrorCode.RESOURCE_DOES_NOT_EXIST, "RESOURCE_DOES_NOT_EXIST"):
        raise kernel.NotFound(message) from None
    if (
        _has_error_code(exc, ErrorCode.RESOURCE_ALREADY_EXISTS, "RESOURCE_ALREADY_EXISTS")
        or _has_error_code(exc, ErrorCode.INVALID_STATE, "INVALID_STATE")
    ):
        raise kernel.Conflict(message) from None
    if _has_error_code(exc, ErrorCode.INVALID_PARAMETER_VALUE, "INVALID_PARAMETER_VALUE"):
        if invalid_param_is_conflict:
            raise kernel.Conflict(message) from None
        raise kernel.ValidationFailed(message) from None
    raise kernel.ValidationFailed(message) from None


def _param_type(value: Any) -> str:
    """Return the declared wire-safe type name for a supported parameter."""
    # bool subclasses int, so it must be checked first to preserve its identity.
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        if not math.isfinite(value):
            raise kernel.ValidationFailed("parameter floats must be finite")
        return "float"
    if type(value) is str:
        return "str"
    raise kernel.ValidationFailed(
        "parameter values must be bool, int, finite float, or str"
    )


def _param_text(value: Any) -> str:
    """Format a parameter exactly as MLflow's string-backed param store does."""
    return str(value)


def _type_tag_key(name: str) -> str:
    """Hash a parameter name so MLflow's tag-key length limit stays bounded."""
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return _PARAM_TYPE_TAG_PREFIX + digest


def _decode_type_metadata(tags: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Read ``hashed-tag-key -> (parameter name, declared type)`` metadata."""
    metadata: dict[str, tuple[str, str]] = {}
    for key, raw in tags.items():
        if not key.startswith(_PARAM_TYPE_TAG_PREFIX):
            continue
        try:
            item = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise kernel.ValidationFailed(
                f"corrupt mirrored parameter type metadata in tag {key}"
            ) from exc
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or item.get("type") not in ("bool", "int", "float", "str")
        ):
            raise kernel.ValidationFailed(
                f"corrupt mirrored parameter type metadata in tag {key}"
            )
        name = item["name"]
        if key != _type_tag_key(name):
            raise kernel.ValidationFailed(
                f"parameter type metadata key does not match its name: {key}"
            )
        metadata[name] = (key, item["type"])
    return metadata


def _restore_param(raw: str, declared_type: str | None) -> Any:
    """Restore only types explicitly declared by this bridge's run tag."""
    if declared_type is None or declared_type == "str":
        # Untagged third-party MLflow params are strings by provider contract;
        # their contents are deliberately not parsed or guessed.
        return raw
    try:
        if declared_type == "bool":
            if raw not in ("True", "False"):
                raise ValueError("boolean parameter is not canonical")
            return raw == "True"
        if declared_type == "int":
            value = int(raw)
            if str(value) != raw:
                raise ValueError("integer parameter is not canonical")
            return value
        if declared_type == "float":
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("float parameter is not finite")
            return value
    except (TypeError, ValueError, OverflowError) as exc:
        raise kernel.ValidationFailed(
            f"mirrored parameter cannot be restored as {declared_type}"
        ) from exc
    raise kernel.ValidationFailed(f"unknown mirrored parameter type: {declared_type}")


class MlflowBridge:
    """Mirror experiment records and model aliases into one SQLite MLflow store.

    Each instance owns a separate ``MlflowClient`` configured with the caller's
    absolute SQLite URI. It never changes MLflow's fluent/global tracking URI,
    so two bridges in one process cannot silently redirect one another. The
    tracker and registry remain authoritative; MLflow stores a read-oriented
    mirror and may therefore contain only the state explicitly sent here.

    Args:
        tracking_uri: Absolute ``sqlite:///`` URI, normally built from a test
            or application-owned database path.

    Raises:
        kernel.ValidationFailed: The URI is not an absolute SQLite URI or
            MLflow cannot construct the isolated client.
    """

    def __init__(self, tracking_uri: str):
        if (
            not isinstance(tracking_uri, str)
            or not tracking_uri.startswith("sqlite:///")
            or not tracking_uri[len("sqlite:///") :].startswith("/")
        ):
            raise kernel.ValidationFailed(
                "tracking_uri must be an absolute sqlite:/// URI"
            )
        self.tracking_uri = tracking_uri
        try:
            # Constructing a client is instance-scoped; calling
            # mlflow.set_tracking_uri here would mutate shared process state.
            self._client = MlflowClient(tracking_uri=tracking_uri)
        except MlflowException as exc:
            _raise_kernel_error(exc, "client construction")

    def _call_mlflow(
        self,
        operation: str,
        method: Any,
        *args: Any,
        _invalid_param_is_conflict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call one provider operation behind the bridge's safe error boundary.

        Keep ``operation`` a source-controlled label. Provider exceptions may
        include database details or request data, so callers receive only the
        mapped kernel category and that label, with the provider cause hidden.
        This wrapper is used for reads as well as writes: a failed lookup is
        still an external boundary failure and must not leak raw MLflow errors.
        """
        try:
            return method(*args, **kwargs)
        except Exception as exc:
            if isinstance(exc, MlflowException):
                _raise_kernel_error(
                    exc,
                    operation,
                    invalid_param_is_conflict=_invalid_param_is_conflict,
                )
            raise kernel.ValidationFailed(f"MLflow {operation} failed") from None

    def _all_experiments(self) -> list[Any]:
        """Fetch experiment pages for run-id lookup across this store only."""
        experiments: list[Any] = []
        token = None
        while True:
            page = self._call_mlflow("experiment listing", self._client.search_experiments,
                max_results=_MAX_PAGE_SIZE, page_token=token
            )
            experiments.extend(page)
            token = getattr(page, "token", None)
            if not token:
                return experiments

    def _runs_for_domain_id(self, run_id: str) -> list[Any]:
        """Find exact tag matches without interpolating IDs into MLflow filters."""
        matches: list[Any] = []
        for experiment in self._all_experiments():
            token = None
            while True:
                page = self._call_mlflow("run listing", self._client.search_runs,
                    [experiment.experiment_id],
                    max_results=_MAX_PAGE_SIZE,
                    page_token=token,
                )
                matches.extend(
                    run
                    for run in page
                    if run.data.tags.get(_RUN_ID_TAG) == run_id
                )
                token = getattr(page, "token", None)
                if not token:
                    break
        return matches

    def _experiment_id(self, name: str) -> str:
        """Get or create one named experiment, recovering a create race."""
        experiment = self._call_mlflow(
            "experiment lookup", self._client.get_experiment_by_name, name
        )
        if experiment is not None:
            return experiment.experiment_id
        try:
            return self._call_mlflow(
                "experiment creation", self._client.create_experiment, name
            )
        except kernel.Conflict:
            # A duplicate-name response can mean another bridge won the
            # create race. Re-read through the same translated boundary.
            experiment = self._call_mlflow(
                "experiment lookup", self._client.get_experiment_by_name, name
            )
            if experiment is not None:
                return experiment.experiment_id
            raise

    def _validate_entries(self, entries: Any) -> list[tuple[str, str, Any, int]]:
        """Validate replayable entries before creating experiments or runs."""
        if not isinstance(entries, list):
            raise kernel.ValidationFailed("tracker_entries must be a list")
        prepared: list[tuple[str, str, Any, int]] = []
        seen_params: dict[str, tuple[str, str]] = {}
        for entry in entries:
            if not isinstance(entry, LogEntry):
                raise kernel.ValidationFailed("tracker_entries must contain LogEntry values")
            if entry.kind not in ("param", "metric"):
                # Checkpoint and backfill records belong to other parts of the
                # tracker contract and intentionally do not become MLflow data.
                continue
            name = _validate_text(entry.name, "entry name")
            if entry.kind == "param":
                type_name = _param_type(entry.value)
                signature = (type_name, _param_text(entry.value))
                previous = seen_params.get(name)
                if previous is not None and previous != signature:
                    raise kernel.Conflict(
                        f"one replay batch contains conflicting values for param {name!r}"
                    )
                seen_params[name] = signature
                prepared.append(("param", name, entry.value, 0))
                continue

            if isinstance(entry.value, bool) or not isinstance(entry.value, (int, float)):
                raise kernel.ValidationFailed("metric values must be finite numbers")
            value = float(entry.value)
            if not math.isfinite(value):
                raise kernel.ValidationFailed("metric values must be finite numbers")
            step = 0 if entry.step is None else entry.step
            if type(step) is not int or step < 0:
                raise kernel.ValidationFailed("metric step must be a non-negative integer")
            prepared.append(("metric", name, value, step))
        return prepared

    def _prepare_run(self, run_id: str, experiment_name: str) -> str:
        """Reuse the unique tagged run or create it in the named experiment."""
        matches = self._runs_for_domain_id(run_id)
        if len(matches) > 1:
            raise kernel.Conflict(f"multiple MLflow runs already mirror run_id {run_id!r}")
        if matches:
            run = matches[0]
            experiment = self._call_mlflow(
                "experiment lookup", self._client.get_experiment,
                run.info.experiment_id,
            )
            if experiment.name != experiment_name:
                raise kernel.Conflict(
                    f"run_id {run_id!r} is already mirrored in experiment {experiment.name!r}"
                )
            return run.info.run_id

        experiment_id = self._experiment_id(experiment_name)
        try:
            created = self._call_mlflow(
                "run creation", self._client.create_run,
                experiment_id, tags={_RUN_ID_TAG: run_id}
            )
        except kernel.ValidationFailed:
            raise
        except kernel.NotFound:
            raise
        except kernel.Conflict:
            raise

        # MLflow tags are indexed values, not unique constraints. Another
        # caller may have inserted the same domain run after our initial
        # lookup. Re-list from the provider after creation, retain this call's
        # row when it is present, soft-delete every competing active row, and
        # re-read before returning. Every query/delete here passes through the
        # translated boundary so a repair failure cannot leak MLflow text.
        matches = self._runs_for_domain_id(run_id)
        survivor_id = created.info.run_id
        for duplicate in matches:
            if duplicate.info.run_id != survivor_id:
                self._call_mlflow(
                    "duplicate run cleanup", self._client.delete_run,
                    duplicate.info.run_id
                )
        reconciled = self._runs_for_domain_id(run_id)
        if len(reconciled) != 1:
            raise kernel.Conflict("MLflow run reconciliation did not produce one active run")
        return reconciled[0].info.run_id

    def _replay_params(
        self, run_id: str, entries: Iterable[tuple[str, str, Any, int]]
    ) -> None:
        """Replay typed params while checking MLflow's immutable-param rule."""
        run = self._call_mlflow("run lookup", self._client.get_run, run_id)
        metadata = _decode_type_metadata(run.data.tags)
        param_entries = [item for item in entries if item[0] == "param"]

        # Check the whole param batch first: a known conflict should not leave
        # earlier params from the same call partially mirrored.
        for _, name, value, _ in param_entries:
            expected_text = _param_text(value)
            actual_text = run.data.params.get(name)
            if actual_text is not None and actual_text != expected_text:
                raise kernel.Conflict(
                    f"MLflow parameter {name!r} already has a different value"
                )
            declared = metadata.get(name)
            if declared is not None and declared[1] != _param_type(value):
                raise kernel.Conflict(
                    f"MLflow parameter {name!r} already has a different declared type"
                )

        for _, name, value, _ in param_entries:
            type_name = _param_type(value)
            if name not in run.data.params:
                self._call_mlflow(
                    "parameter logging", self._client.log_param,
                    run_id, name, value,
                    _invalid_param_is_conflict=True,
                )
            if name not in metadata:
                # Type metadata follows successful param persistence: a failed
                # param write must not leave a declaration for a nonexistent
                # value. Conversely, a metadata failure is surfaced explicitly
                # rather than returning a falsely typed value later.
                tag_value = json.dumps(
                    {"name": name, "type": type_name},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                try:
                    self._call_mlflow(
                        "parameter type metadata", self._client.set_tag,
                        run_id, _type_tag_key(name), tag_value
                    )
                except kernel.ValidationFailed:
                    raise
                metadata[name] = (_type_tag_key(name), type_name)

    def _replay_metrics(
        self, run_id: str, entries: Iterable[tuple[str, str, Any, int]]
    ) -> None:
        """Append unseen metric (name, step, value) records with exact steps."""
        known: dict[str, set[tuple[int, float]]] = {}
        for kind, name, _, _ in entries:
            if kind != "metric" or name in known:
                continue
            try:
                history = self._call_mlflow(
                    "metric history lookup", self._client.get_metric_history,
                    run_id, name
                )
            except kernel.NotFound:
                # A missing metric history means this name has not yet been
                # mirrored; other provider errors still cross the translator.
                history = []
            known[name] = {(int(item.step), float(item.value)) for item in history}

        for kind, name, value, step in entries:
            if kind != "metric":
                continue
            fingerprint = (step, float(value))
            if fingerprint in known[name]:
                continue
            try:
                self._call_mlflow(
                    "metric logging", self._client.log_metric,
                    run_id, name, float(value), step=step
                )
            except kernel.ValidationFailed:
                raise
            known[name].add(fingerprint)

    def mirror_experiment_run(
        self, tracker_entries: list[LogEntry], run_id: str, experiment_name: str
    ) -> str:
        """Mirror a run's params and metric history and return its MLflow ID.

        The domain tracker remains authoritative. A stable ``mlops_run_id`` tag
        lets repeated calls (including calls through a fresh bridge instance)
        reuse the same MLflow run. Parameters carry private type tags because
        MLflow stores them as strings; metrics retain their original step. The
        mirror skips non-param/metric log kinds and avoids duplicating already
        mirrored metric samples.

        Raises:
            kernel.ValidationFailed: Invalid input or an MLflow validation error.
            kernel.Conflict: A run/parameter conflicts with existing mirror data.
            kernel.NotFound: MLflow reports that a referenced mirror resource is
                missing.
        """
        run_id = _validate_text(run_id, "run_id")
        experiment_name = _validate_text(experiment_name, "experiment_name")
        prepared = self._validate_entries(tracker_entries)
        run_mlflow_id = self._prepare_run(run_id, experiment_name)
        self._replay_params(run_mlflow_id, prepared)
        self._replay_metrics(run_mlflow_id, prepared)
        return run_mlflow_id

    def get_mirrored_run(self, run_id: str) -> dict[str, dict[str, Any]]:
        """Return mirrored ``params``, latest ``metrics``, and run ``tags``.

        Only type metadata written by this bridge changes a parameter's Python
        type. A parameter created directly by another MLflow client, or one
        whose metadata predates this bridge, remains a string regardless of
        whether its text resembles a number or boolean.

        Raises:
            kernel.ValidationFailed: ``run_id`` is malformed or type metadata
                is corrupt.
            kernel.NotFound: No MLflow run carries the requested domain ID.
            kernel.Conflict: More than one run carries that ID.
        """
        run_id = _validate_text(run_id, "run_id")
        matches = self._runs_for_domain_id(run_id)
        if not matches:
            raise kernel.NotFound(f"no mirrored MLflow run for run_id {run_id!r}")
        if len(matches) > 1:
            raise kernel.Conflict(f"multiple MLflow runs mirror run_id {run_id!r}")
        run = matches[0]
        metadata = _decode_type_metadata(run.data.tags)
        params = {
            name: _restore_param(raw, metadata.get(name, ("", "str"))[1])
            for name, raw in run.data.params.items()
        }
        return {
            "params": params,
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
        }

    def _registered_model(self, model_name: str) -> Any | None:
        """Return a registered model or None when MLflow confirms it is absent."""
        try:
            return self._call_mlflow(
                "registered model lookup", self._client.get_registered_model,
                model_name
            )
        except kernel.NotFound:
            return None

    def _model_versions(self, model_name: str) -> list[Any]:
        """List all versions for one model to compute MLflow's next allocation."""
        versions: list[Any] = []
        token = None
        # The registry filter language treats model names as literals. Escaping
        # both backslashes and quotes prevents names from changing the query.
        literal = model_name.replace("\\", "\\\\").replace("'", "\\'")
        filter_string = f"name = '{literal}'"
        while True:
            page = self._call_mlflow("model version listing", self._client.search_model_versions,
                filter_string=filter_string,
                max_results=_MAX_PAGE_SIZE,
                page_token=token,
            )
            versions.extend(
                version for version in page if version.name == model_name
            )
            token = getattr(page, "token", None)
            if not token:
                return versions

    def _next_version(self, model_name: str) -> int:
        """Calculate the next numeric MLflow allocation before registry writes."""
        versions = self._model_versions(model_name)
        numbers: list[int] = []
        for version in versions:
            raw = str(version.version)
            if not raw.isascii() or not raw.isdigit():
                raise kernel.ValidationFailed(
                    f"MLflow returned a non-numeric version for {model_name!r}"
                )
            numbers.append(int(raw))
        return max(numbers, default=0) + 1

    def _find_version(self, model_name: str, canonical_version: str) -> Any | None:
        """Return a canonical model version, or None when it does not exist."""
        try:
            return self._call_mlflow(
                "model version lookup", self._client.get_model_version,
                model_name, canonical_version
            )
        except kernel.NotFound:
            return None

    def _ensure_version(self, model_name: str, canonical_version: str) -> Any:
        """Find or safely allocate exactly the requested MLflow version.

        MLflow assigns version numbers itself. Before creating a model or model
        version, this method checks the next real allocation. That preflight is
        the side-effect boundary: asking for version 7 on a fresh model raises
        Conflict while leaving no registered model behind. If concurrent
        writers race after the preflight, the returned allocation is checked
        again and a mismatch is rejected before any alias can move.
        """
        existing = self._find_version(model_name, canonical_version)
        if existing is not None:
            return existing

        model = self._registered_model(model_name)
        if model is None:
            # A brand-new MLflow model can only receive version 1. Rejecting a
            # skipped request here avoids even creating an empty model shell.
            if int(canonical_version) != 1:
                raise kernel.Conflict(
                    f"requested version {canonical_version} is not MLflow's next allocation 1"
                )
            try:
                self._call_mlflow(
                    "registered model creation", self._client.create_registered_model,
                    model_name
                )
            except kernel.Conflict:
                # A concurrent writer may have created the model. Re-check its
                # versions below; the preflight must reflect the state now.
                pass
            existing = self._find_version(model_name, canonical_version)
            if existing is not None:
                return existing

        next_number = self._next_version(model_name)
        if int(canonical_version) != next_number:
            raise kernel.Conflict(
                f"requested version {canonical_version} is not MLflow's next allocation {next_number}"
            )

        try:
            created = self._call_mlflow("model version creation", self._client.create_model_version,
                name=model_name,
                source="",
                run_id="",
            )
        except (kernel.ValidationFailed, kernel.NotFound, kernel.Conflict):
            raise
        actual_version = str(created.version)
        if actual_version != canonical_version:
            # Another registry writer won the allocation between preflight and
            # create. The extra provider version may remain, but it is never
            # advertised through the requested alias as though it matched.
            raise kernel.Conflict(
                f"MLflow allocated version {actual_version}; requested {canonical_version}"
            )
        confirmed = self._find_version(model_name, canonical_version)
        if confirmed is None:
            raise kernel.Conflict(
                f"MLflow did not expose newly allocated version {canonical_version}"
            )
        return confirmed

    def mirror_stage_transition(
        self, model_name: str, version: str, stage: Stage
    ) -> None:
        """Mirror a domain stage using MLflow aliases, never legacy stages.

        ``version`` must contain ASCII digits and is normalized with
        ``str(int(version))``; thus ``"007"`` means version 7. The bridge first
        checks for that exact canonical version. If absent, it compares the
        request with MLflow's next allocation before creating any registry
        state. Only after the returned version is verified may a staging or
        production alias move. On an allocation race, the bridge fails closed
        and leaves aliases untouched.

        ``REGISTERED`` and ``ARCHIVED`` both clear the staging and production
        aliases; they do not use MLflow's deprecated ``current_stage`` field.
        Repeating a transition is safe because alias assignment is idempotent
        and deleting a missing alias is treated as already clear.

        Raises:
            kernel.ValidationFailed: Invalid model, version, or stage.
            kernel.Conflict: The requested canonical version cannot be honored.
            kernel.NotFound: MLflow reports a referenced registry resource
                missing outside the expected absent-resource checks.
        """
        model_name = _validate_text(model_name, "model_name")
        if (
            not isinstance(version, str)
            or not version
            or not version.isascii()
            or not version.isdigit()
        ):
            raise kernel.ValidationFailed("version must contain ASCII digits only")
        if not isinstance(stage, Stage):
            raise kernel.ValidationFailed("stage must be a Stage value")
        try:
            canonical_version = str(int(version))
        except (ValueError, OverflowError) as exc:
            raise kernel.ValidationFailed("version is outside the supported integer range") from exc

        target = self._ensure_version(model_name, canonical_version)
        if str(target.version) != canonical_version:
            raise kernel.Conflict(
                f"resolved MLflow version {target.version} does not match {canonical_version}"
            )

        alias = {
            Stage.STAGING: "staging",
            Stage.PRODUCTION: "production",
        }.get(stage)
        if alias is not None:
            try:
                self._call_mlflow(
                    f"{alias} alias assignment", self._client.set_registered_model_alias,
                    model_name, alias, canonical_version
                )
            except (kernel.ValidationFailed, kernel.NotFound, kernel.Conflict):
                raise
            return

        # REGISTERED/ARCHIVED represent no active serving designation. Clear
        # both aliases only after the requested model/version is known to exist;
        # a bad version request must never disturb a currently serving alias.
        for active_alias in _STAGE_ALIASES:
            try:
                self._call_mlflow(
                    f"{active_alias} alias cleanup", self._client.delete_registered_model_alias,
                    model_name, active_alias
                )
            except kernel.NotFound:
                continue
            except (kernel.ValidationFailed, kernel.Conflict):
                raise
