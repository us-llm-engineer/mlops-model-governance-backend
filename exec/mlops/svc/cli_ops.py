"""F2.5 CLI operations: incidents, audit verify/tail, and sbom build."""

import json
import re
from typing import Callable, Optional

import typer
from rich.console import Console
from rich.table import Table

from mlops.ext.cyclonedx_sbom import (
    build_cyclonedx_sbom,
    sbom_matches_requirements,
    validate_cyclonedx_sbom,
)
from mlops.kernel import ValidationFailed
from mlops.svc.sdk import MlopsClient, MlopsApiError


def register_ops_commands(
    cli_app: typer.Typer,
    client_factory: Callable[[], MlopsClient],
    *,
    audit_group: Optional[typer.Typer] = None,
) -> None:
    """Register operations commands (incidents, audit verify/tail, sbom build).

    Args:
        cli_app: The main CLI app to add sub-apps to
        client_factory: Factory function to create MlopsClient
        audit_group: Optional existing audit Typer group to add commands to
    """

    # ================================================================ incidents sub-app
    incidents = typer.Typer()
    cli_app.add_typer(incidents, name="incidents")

    @incidents.command()
    def open(title: str, signal_file: str) -> None:
        """Open a new incident with a signal file."""
        try:
            from mlops.svc.cli import _read_file_json, _handle_api_error

            client = client_factory()
            signal = _read_file_json(signal_file)
            result = client.open_incident(title, signal)
            # Print just the incident ID as plain text
            typer.echo(result.get("incident_id", ""))
        except typer.Exit:
            raise
        except MlopsApiError as e:
            from mlops.svc.cli import _handle_api_error
            _handle_api_error(e)
        except Exception as e:
            typer.echo(f"error: {str(e)}", err=True)
            raise typer.Exit(code=1)

    @incidents.command()
    def get(incident_id: str, table: bool = False) -> None:
        """Get an incident by ID."""
        try:
            from mlops.svc.cli import _print_json, _handle_api_error

            client = client_factory()
            result = client.get_incident(incident_id)

            if table:
                # Render as table
                t = Table(title="Incident")
                t.add_column("Key", style="cyan")
                t.add_column("Value", style="green")

                # Add selected fields to table
                for key in ["incident_id", "title", "severity", "status", "opened_ts"]:
                    if key in result:
                        t.add_row(key, str(result[key]))

                console = Console()
                console.print(t)
            else:
                # Print as JSON
                _print_json(result)
        except typer.Exit:
            raise
        except MlopsApiError as e:
            from mlops.svc.cli import _handle_api_error
            _handle_api_error(e)
        except Exception as e:
            typer.echo(f"error: {str(e)}", err=True)
            raise typer.Exit(code=1)

    # ================================================================ audit verify & tail
    if audit_group is not None:
        @audit_group.command()
        def verify(export_file: str, head_file: str) -> None:
            """Verify an audit export chain."""
            try:
                from mlops.svc.cli import _read_file_json, _handle_api_error

                client = client_factory()
                export = _read_file_json(export_file)
                head = _read_file_json(head_file)

                result = client.audit_verify(export, head)
                # Fail closed: only a literal JSON true counts ("yes", 1, [..] must not exit 0).
                is_valid = result.get("valid") is True

                # Always print the result as JSON
                output = {"valid": is_valid}
                if "reason" in result:
                    output["reason"] = result["reason"]
                typer.echo(json.dumps(output))

                # Exit with code 1 if invalid
                if not is_valid:
                    raise typer.Exit(code=1)
            except typer.Exit:
                raise
            except MlopsApiError as e:
                from mlops.svc.cli import _handle_api_error
                _handle_api_error(e)
            except Exception as e:
                typer.echo(f"error: {str(e)}", err=True)
                raise typer.Exit(code=1)

        @audit_group.command()
        def tail() -> None:
            """Tail the audit stream."""
            try:
                client = client_factory()
                for event in client.audit_tail():
                    typer.echo(json.dumps(event))
            except KeyboardInterrupt:
                # Clean exit on Ctrl-C
                raise typer.Exit(code=0)
            except typer.Exit:
                raise
            except MlopsApiError as e:
                from mlops.svc.cli import _handle_api_error
                _handle_api_error(e)
            except Exception as e:
                typer.echo(f"error: {str(e)}", err=True)
                raise typer.Exit(code=1)

    # ================================================================ sbom sub-app
    sbom = typer.Typer()
    cli_app.add_typer(sbom, name="sbom")

    @sbom.command()
    def build(requirements_file: str, out_file: Optional[str] = typer.Argument(None)) -> None:
        """Build a CycloneDX SBOM from a requirements file."""
        try:
            from pathlib import Path

            # Parse requirements file
            req_path = Path(requirements_file)
            try:
                req_text = req_path.read_text(encoding="utf-8-sig")
            except FileNotFoundError:
                typer.echo(f"error: file not found: {requirements_file}", err=True)
                raise typer.Exit(code=2)
            except Exception as e:
                typer.echo(f"error: cannot read file: {e}", err=True)
                raise typer.Exit(code=2)

            # Parse lines: name==version, skip blanks and comments
            components = []
            for line_num, line in enumerate(req_text.splitlines(), 1):
                # Strip whitespace
                stripped = line.strip()

                # Skip blank lines and comments
                if not stripped or stripped.startswith("#"):
                    continue

                # Inline comment: '#' preceded by whitespace ends the requirement.
                stripped = re.split(r"\s+#", stripped, maxsplit=1)[0].strip()

                # Errors name the LINE NUMBER only -- a requirements file may hold credentialed
                # URLs, and echoing a rejected line would print the secret.
                if ";" in stripped:
                    typer.echo(f"error: line {line_num}: environment markers are not supported", err=True)
                    raise typer.Exit(code=2)

                parts = stripped.split("==", 1)
                if len(parts) != 2:
                    typer.echo(f"error: invalid requirements line {line_num}", err=True)
                    raise typer.Exit(code=2)

                # name[extra1,extra2] -> name (extras do not change which distribution is installed)
                match = re.fullmatch(r"([^\[\]\s]+)\s*(?:\[[^\[\]]*\])?\s*", parts[0])
                name = match.group(1) if match else ""
                version = parts[1].strip()
                if not name or not version or re.search(r"[\s\[\];#]", version):
                    typer.echo(f"error: invalid requirements line {line_num}", err=True)
                    raise typer.Exit(code=2)
                components.append((name, version))

            # Build SBOM
            try:
                sbom_dict = build_cyclonedx_sbom(components)
            except ValidationFailed:
                # Never echo the message: a requirements line may carry credentials
                # (https://user:pw@host/pkg==1.0) that end up in the component name.
                typer.echo("error: requirements rejected (invalid or duplicate component)", err=True)
                raise typer.Exit(code=2)

            # Self-check the SBOM before ever writing it out: schema-validate it and
            # confirm it round-trips the exact components just parsed. A build that
            # silently drifted from its own input should never be handed to a user.
            try:
                validate_cyclonedx_sbom(sbom_dict)
                if not sbom_matches_requirements(sbom_dict, components):
                    raise ValidationFailed("SBOM does not match the parsed requirements")
            except ValidationFailed as e:
                typer.echo(f"error: internal SBOM self-check failed: {e}", err=True)
                raise typer.Exit(code=3)

            # Output
            output = json.dumps(sbom_dict, indent=2, sort_keys=True)

            if out_file:
                # Write to file
                try:
                    Path(out_file).write_text(output)
                except Exception as e:
                    typer.echo(f"error: cannot write file: {e}", err=True)
                    raise typer.Exit(code=2)
            else:
                # Print to stdout
                typer.echo(output, nl=False)
                if not output.endswith("\n"):
                    typer.echo()

        except typer.Exit:
            raise
        except Exception as e:
            typer.echo(f"error: {str(e)}", err=True)
            raise typer.Exit(code=2)
