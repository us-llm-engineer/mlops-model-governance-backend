"""CLI for MLOps control plane (S3.3)."""

import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import click
import typer

from mlops.svc.sdk import MlopsClient, MlopsApiError
from mlops.svc.cli_ops import register_ops_commands

app: typer.Typer = None


def default_factory() -> MlopsClient:
    """Create MlopsClient from environment variables.

    Reads MLOPS_URL and MLOPS_TOKEN environment variables.
    Exits with code 2 if either is missing.
    """
    url = os.environ.get("MLOPS_URL")
    token = os.environ.get("MLOPS_TOKEN")

    if not url:
        typer.echo("error: MLOPS_URL environment variable is not set", err=True)
        raise typer.Exit(code=2)

    if not token:
        typer.echo("error: MLOPS_TOKEN environment variable is not set", err=True)
        raise typer.Exit(code=2)

    return MlopsClient(base_url=url, token=token)


def _read_file_json(path: str) -> dict:
    """Read and parse a JSON file, exiting with code 2 on errors.

    Checks file size (max 5 MiB), reads content, and parses JSON.
    Rejects NaN/Infinity in JSON.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON object

    Raises:
        typer.Exit: With code 2 on any file/JSON error
    """
    file_path = Path(path)

    # Check if file exists and get size
    try:
        file_stat = file_path.stat()
    except FileNotFoundError:
        typer.echo(f"error: file not found: {path}", err=True)
        raise typer.Exit(code=2)
    except OSError as e:
        typer.echo(f"error: cannot read file: {e}", err=True)
        raise typer.Exit(code=2)

    # Check size (5 MiB)
    max_size = 5 * 1024 * 1024
    if file_stat.st_size > max_size:
        typer.echo(f"error: file too large (max {max_size} bytes)", err=True)
        raise typer.Exit(code=2)

    # Read file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        typer.echo("error: file is not valid UTF-8", err=True)
        raise typer.Exit(code=2)
    except OSError as e:
        typer.echo(f"error: cannot read file: {e}", err=True)
        raise typer.Exit(code=2)

    # Parse JSON, rejecting NaN/Infinity
    try:
        def reject_constant(x):
            raise ValueError(f"Invalid JSON constant: {x}")

        obj = json.loads(content, parse_constant=reject_constant)
        return obj
    except json.JSONDecodeError as e:
        typer.echo(f"error: invalid JSON: {e}", err=True)
        raise typer.Exit(code=2)
    except ValueError as e:
        typer.echo(f"error: invalid JSON: {e}", err=True)
        raise typer.Exit(code=2)


def _print_json(obj: dict) -> None:
    """Print object as pretty JSON to stdout."""
    output = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    typer.echo(output, nl=False)


def _handle_api_error(e: MlopsApiError) -> None:
    """Handle API error by printing error message and exiting with code 1."""
    typer.echo(f"error: {e.code}: {e.message}", err=True)
    raise typer.Exit(code=1)


def create_cli(client_factory: Callable[[], MlopsClient]) -> typer.Typer:
    """Create the CLI application.

    Args:
        client_factory: Callable that returns an MlopsClient

    Returns:
        Configured Typer application
    """
    cli_app = typer.Typer()

    # Health command (no auth)
    @cli_app.command()
    def health() -> None:
        """Check health of the API."""
        try:
            client = client_factory()
            result = client.health()
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)
        except Exception as e:
            typer.echo(f"error: {str(e)}", err=True)
            raise typer.Exit(code=1)

    # Models subcommand group
    models = typer.Typer()
    cli_app.add_typer(models, name="models")

    @models.command()
    def register(
        model_id: str,
        version_id: str,
        artifact_hash: str,
        dataset_version: Optional[str] = None,
    ) -> None:
        """Register a new model version."""
        try:
            client = client_factory()
            result = client.register_model(
                model_id=model_id,
                version_id=version_id,
                artifact_hash=artifact_hash,
                dataset_version=dataset_version,
            )
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    # Lineage subcommand group
    lineage = typer.Typer()
    cli_app.add_typer(lineage, name="lineage")

    @lineage.command("blast-radius")
    def lineage_blast_radius_cmd(model_id: str, version_id: str) -> None:
        """Nodes that would be affected by a change to this model version."""
        try:
            client = client_factory()
            result = client.lineage_blast_radius(model_id, version_id)
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    @lineage.command("ancestors")
    def lineage_ancestors_cmd(model_id: str, version_id: str) -> None:
        """Nodes this model version's current state descends from."""
        try:
            client = client_factory()
            result = client.lineage_ancestors(model_id, version_id)
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    # Datasets subcommand group
    datasets = typer.Typer()
    cli_app.add_typer(datasets, name="datasets")

    @datasets.command("register")
    def register_dataset(
        name: str,
        content_hash: str,
        rows: int,
        schema_file: str = typer.Argument(..., help="JSON file with the dataset schema"),
    ) -> None:
        """Register a new dataset version."""
        try:
            schema = _read_file_json(schema_file)
            client = client_factory()
            result = client.register_dataset(
                name=name,
                content_hash=content_hash,
                rows=rows,
                schema=schema,
            )
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    @models.command()
    def transition(
        model_id: str,
        version_id: str,
        to_stage: str = typer.Argument(..., help="Target stage: staging|production|archived"),
        context: Optional[str] = typer.Option(None, help="Context as JSON string"),
    ) -> None:
        """Transition a model to a new stage."""
        try:
            client = client_factory()

            # Parse context if provided
            context_dict = None
            if context:
                try:
                    def reject_constant(x):
                        raise ValueError(f"Invalid JSON constant: {x}")
                    context_dict = json.loads(context, parse_constant=reject_constant)
                except (json.JSONDecodeError, ValueError) as e:
                    typer.echo(f"error: invalid context JSON: {e}", err=True)
                    raise typer.Exit(code=2)

            # Validate to_stage
            if to_stage not in ("staging", "production", "archived"):
                typer.echo(f"error: invalid stage: {to_stage}", err=True)
                raise typer.Exit(code=2)

            result = client.transition(
                model_id=model_id,
                version_id=version_id,
                to_stage=to_stage,
                context=context_dict,
            )
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    @models.command()
    def get(model_id: str) -> None:
        """Get model version history."""
        try:
            client = client_factory()
            result = client.get_model(model_id=model_id)
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    # Policy subcommand group
    policy = typer.Typer()
    cli_app.add_typer(policy, name="policy")

    @policy.command()
    def publish(file: str) -> None:
        """Publish a policy from a JSON file."""
        try:
            client = client_factory()
            rules_obj = _read_file_json(file)

            # Validate shape: must be object
            if not isinstance(rules_obj, dict):
                typer.echo(f"error: policy must be a JSON object", err=True)
                raise typer.Exit(code=2)

            # Extract policy fields
            name = rules_obj.get("name")
            version = rules_obj.get("version")
            rules = rules_obj.get("rules", [])
            note = rules_obj.get("note")

            result = client.publish_policy(
                rules=rules,
                name=name,
                version=version,
                note=note,
            )
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    @policy.command()
    def activate(
        version: int,
        approved_by: Optional[str] = typer.Option(None, "--approved-by", help="Approver name"),
    ) -> None:
        """Activate a policy version."""
        try:
            client = client_factory()
            result = client.activate_policy(
                version=version,
                human_approved_by=approved_by,
            )
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    @policy.command()
    def decide(action: str, context_json: str) -> None:
        """Make a policy decision."""
        try:
            client = client_factory()

            # Parse context JSON
            try:
                def reject_constant(x):
                    raise ValueError(f"Invalid JSON constant: {x}")
                context = json.loads(context_json, parse_constant=reject_constant)
            except (json.JSONDecodeError, ValueError) as e:
                typer.echo(f"error: invalid context JSON: {e}", err=True)
                raise typer.Exit(code=2)

            # Validate context is a dict (JSON object)
            if not isinstance(context, dict):
                typer.echo(f"error: context must be a JSON object", err=True)
                raise typer.Exit(code=2)

            result = client.decide(action=action, context=context)
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    @policy.command()
    def active() -> None:
        """Get the active policy."""
        try:
            client = client_factory()
            result = client.active_policy()
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    @policy.command("differential-check")
    def differential_check(
        file: str,
        subject: list[str] = typer.Option(..., "--subject", help="Subject name to test (repeatable)"),
        seed: int = typer.Option(0, help="Deterministic seed for generated test requests"),
    ) -> None:
        """Cross-check a candidate policy bundle: casbin vs the real PolicyDecisionPoint.

        Entirely local (no server round trip, no audit entries written): builds a
        casbin enforcer from the same bundle file `policy publish` would send, and
        compares its decisions against the real PolicyDecisionPoint on a generated
        set of requests -- a pre-publish sanity check, not a claim of equivalence.
        """
        from mlops.kernel import ValidationFailed
        from mlops.policy_engine import PolicyBundle, PolicyDecisionPoint
        from mlops.interop.casbin_adapter import generate_requests, run_differential

        try:
            data = _read_file_json(file)
            bundle = PolicyBundle.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            typer.echo(f"error: invalid policy bundle: {e}", err=True)
            raise typer.Exit(code=2)

        pdp = PolicyDecisionPoint(bundle=bundle)
        try:
            requests = generate_requests(bundle, list(subject), seed)
            result = run_differential(pdp, bundle, requests)
        except ValidationFailed as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(code=2)

        _print_json({
            "total": result.total,
            "agreements": result.agreements,
            "disagreements": result.disagreements,
            "untranslatable_rule_ids": result.untranslatable_rule_ids,
        })

        if result.disagreements:
            raise typer.Exit(code=1)

    @policy.command("calibration-crosscheck")
    def calibration_crosscheck(
        bundle_file: str,
        feedback_file: str,
        metric: str = typer.Option(..., help="Context key in each feedback record to use as the score"),
        blocks_when: str = typer.Option(
            "below", "--blocks-when",
            help="below: a LOW metric should be blocked (min_metric rules); above: a HIGH one should",
        ),
    ) -> None:
        """Cross-check mlops.policy_calibration.evaluate() against sklearn calibrators.

        Entirely local: loads a policy bundle and a list of {"context": {...},
        "should_block": bool} feedback records, evaluates the bundle's real detection/
        false-alarm rates, then cross-checks the same (score, ground-truth) pairs
        against sklearn's isotonic/logistic calibrators as a demonstration/cross-check
        (never a claim of equivalence to the domain calibrator).
        """
        from mlops.kernel import ValidationFailed
        from mlops.policy_engine import PolicyBundle
        from mlops.policy_calibration import Feedback, evaluate
        from mlops.ext.calibration_sk import compare_to_calibrator

        try:
            bundle = PolicyBundle.from_dict(_read_file_json(bundle_file))
            feedback_raw = _read_file_json(feedback_file)
            if not isinstance(feedback_raw, list):
                raise ValidationFailed("feedback file must be a JSON array")
            feedback = [Feedback(context=f["context"], should_block=f["should_block"]) for f in feedback_raw]
            if blocks_when not in ("below", "above"):
                raise ValidationFailed("--blocks-when must be 'below' or 'above'")
            # sklearn's isotonic fit is increasing: a larger x must mean a larger chance
            # of label 1 (blocked). For a min_metric rule a LOW score is what blocks, so
            # negate the score; otherwise the fit collapses to the constant base rate.
            sign = -1.0 if blocks_when == "below" else 1.0
            x = [sign * float(f.context[metric]) for f in feedback]
            y_binary = [1 if f.should_block else 0 for f in feedback]
        except (KeyError, TypeError, ValueError, ValidationFailed) as e:
            typer.echo(f"error: invalid bundle/feedback: {e}", err=True)
            raise typer.Exit(code=2)

        try:
            existing_result = evaluate(bundle, feedback)
            comparison = compare_to_calibrator(existing_result, x, y_binary)
        except ValidationFailed as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(code=2)

        # compare_to_calibrator's "existing_agrees_within" is the mean absolute difference
        # between the isotonic and Platt predictions; it does not use existing_result.
        # Report it under a name that says so instead of one that implies otherwise.
        _print_json({
            "existing_result": existing_result,
            "sklearn_crosscheck": {
                "score_orientation": f"x = {'-' if blocks_when == 'below' else '+'}{metric}",
                "isotonic": comparison["isotonic"],
                "platt": comparison["platt"],
                "isotonic_vs_platt_mean_abs_diff": comparison["existing_agrees_within"],
            },
            "note": "existing_result is reported next to the sklearn fits, not compared with them.",
        })

    # Lint command
    @cli_app.command()
    def lint(file: str) -> None:
        """Lint a Kubernetes manifest file."""
        try:
            client = client_factory()
            manifest = _read_file_json(file)

            # Validate shape: must be object, array, or {"kind":"List"}
            if isinstance(manifest, dict):
                documents = [manifest]
            elif isinstance(manifest, list):
                documents = manifest
            else:
                typer.echo(f"error: manifest must be a JSON object or array", err=True)
                raise typer.Exit(code=2)

            result = client.lint(documents=documents)

            # Check for block-severity findings
            findings = result.get("findings", [])
            has_block = any(f.get("severity") == "block" for f in findings)

            _print_json(result)

            if has_block:
                raise typer.Exit(code=1)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    # Lint-manifest command: local raw-YAML manifest bundle (not the pre-parsed
    # JSON documents `lint` above sends over HTTP). Runs entirely locally --
    # load_and_lint adds amplification/byte-size/schema checks on top of the
    # same mlops.config_lint rules `lint` uses, so no server round trip is
    # needed for a file already sitting on disk.
    @cli_app.command("lint-manifest")
    def lint_manifest(file: str) -> None:
        """Lint a raw YAML Kubernetes manifest bundle (local file, no server round trip)."""
        from mlops.kernel import ValidationFailed
        from mlops.svc.manifest_io import load_and_lint

        try:
            raw = Path(file).read_text(encoding="utf-8")
        except OSError as e:
            typer.echo(f"error: cannot read file: {e}", err=True)
            raise typer.Exit(code=2)

        try:
            docs, findings = load_and_lint(raw)
        except ValidationFailed as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(code=2)

        result = {
            "documents": len(docs),
            "findings": [
                f if isinstance(f, dict) else {
                    "rule_id": f.rule_id, "category": f.category,
                    "severity": f.severity, "message": f.message,
                }
                for f in findings
            ],
        }
        _print_json(result)

        if any(f.get("severity") == "block" for f in result["findings"]):
            raise typer.Exit(code=1)

    # Drift subcommand group
    drift = typer.Typer()
    cli_app.add_typer(drift, name="drift")

    @drift.command()
    def check(reference_file: str, window_file: str) -> None:
        """Check for data drift."""
        try:
            client = client_factory()

            reference = _read_file_json(reference_file)
            window = _read_file_json(window_file)

            # Validate both are arrays
            if not isinstance(reference, list):
                typer.echo(f"error: reference must be a JSON array", err=True)
                raise typer.Exit(code=2)
            if not isinstance(window, list):
                typer.echo(f"error: window must be a JSON array", err=True)
                raise typer.Exit(code=2)

            result = client.drift_check(reference=reference, window=window)
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    # Audit subcommand group
    audit = typer.Typer()
    cli_app.add_typer(audit, name="audit")

    @audit.command()
    def head() -> None:
        """Get the audit log head."""
        try:
            client = client_factory()
            result = client.audit_head()
            _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            _handle_api_error(e)

    # Register operations commands (incidents, audit verify/tail, sbom build)
    register_ops_commands(cli_app, client_factory, audit_group=audit)

    return cli_app


# Create the module-level app
app = create_cli(default_factory)
