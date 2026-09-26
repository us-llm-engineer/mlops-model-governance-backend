"""
F2.2 manifest IO: YAML loading with safeguards against amplification attacks,
schema validation via jsonschema, and linting integration.
"""

import yaml
import jsonschema
from typing import Any, Dict, List, Optional, Tuple

from mlops.kernel import ValidationFailed
from mlops.config_lint import lint_documents


# Minimal Kubernetes manifest schema
MANIFEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["apiVersion", "kind", "metadata"],
    "properties": {
        "apiVersion": {"type": "string"},
        "kind": {"type": "string"},
        "metadata": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"}
            }
        }
    }
}


def check_amplification(doc: Any, *, max_visits: int = 5000) -> None:
    """
    Walk a YAML-parsed structure counting total node visits.

    Detects exponential anchor/alias amplification by tracking:
    - Total node visits (not unique nodes)
    - id() of already-fully-counted container objects (to avoid double-charging
      depth on a shared branch's first visit)
    - Each subsequent alias hit counts as +1 for the container, plus +1 for each
      node it contains

    Raises ValidationFailed if max_visits is exceeded.
    """
    visits = [0]
    container_size = {}  # id() -> size (number of nodes in children of that container)

    def visit(obj: Any) -> None:
        """Recursively visit and count nodes."""
        # Count this node visit
        visits[0] += 1
        if visits[0] > max_visits:
            raise ValidationFailed(f"Node visit count exceeds {max_visits}")

        # Only containers (dict, list) have children; scalars are leaf nodes
        if not isinstance(obj, (dict, list)):
            return

        obj_id = id(obj)

        # If we've already fully explored this container, add the size of its children
        if obj_id in container_size:
            visits[0] += container_size[obj_id]
            if visits[0] > max_visits:
                raise ValidationFailed(f"Node visit count exceeds {max_visits}")
            return

        # First visit to this container: explore its children normally
        start_count = visits[0]

        if isinstance(obj, dict):
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

        # Record the size of this container's children
        size = visits[0] - start_count
        container_size[obj_id] = size

    try:
        visit(doc)
    except RecursionError:
        # Self-referential anchors (&a [*a]) or pathological nesting: the walk never bottoms
        # out, which is exactly the amplification this guard exists to refuse.
        raise ValidationFailed("Document is recursive or nested too deeply") from None


def validate_schema(doc: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    Validate a document against a JSON schema.

    Returns a list of human-readable error messages in the form "<dotted.path>: <message>",
    sorted by path, capped at 100 errors.

    Raises ValidationFailed if the schema itself is broken.
    """
    # First, validate the schema itself
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.exceptions.SchemaError as e:
        raise ValidationFailed(f"Invalid schema: {e.message}")

    # Now validate the document
    validator = jsonschema.Draft202012Validator(schema)
    errors = list(validator.iter_errors(doc))

    # Convert errors to "<dotted.path>: <message>" format
    messages = []
    for error in errors:
        # Build the dotted path from the error's path
        path_parts = list(error.absolute_path)
        if path_parts:
            dotted_path = ".".join(str(p) for p in path_parts)
        else:
            dotted_path = ""

        message = error.message

        if dotted_path:
            messages.append(f"{dotted_path}: {message}")
        else:
            messages.append(message)

    # Sort and cap at 100
    messages = sorted(messages)[:100]
    return messages


def load_manifests(
    raw: str,
    *,
    max_bytes: int = 1_000_000,
    max_documents: int = 200
) -> List[Dict[str, Any]]:
    """
    Load Kubernetes manifests from YAML.

    Args:
        raw: Raw YAML text
        max_bytes: Maximum byte size (checked BEFORE parsing)
        max_documents: Maximum number of documents

    Returns:
        List of parsed documents (dicts)

    Raises:
        ValidationFailed: If size cap exceeded, too many documents, malformed YAML,
                         non-dict documents, or amplification attack detected
    """
    if not isinstance(raw, str):
        raise ValidationFailed("Input must be a str")
    # Check byte size BEFORE parsing
    try:
        size = len(raw.encode())
    except UnicodeEncodeError:
        raise ValidationFailed("Input is not valid text") from None
    if size > max_bytes:
        raise ValidationFailed(f"Input exceeds max_bytes ({max_bytes})")

    # Parse YAML
    docs = []
    try:
        for i, doc in enumerate(yaml.safe_load_all(raw)):
            # Stop if we've exceeded max_documents
            if i >= max_documents:
                raise ValidationFailed(f"More than {max_documents} documents")

            # Skip None (blank documents)
            if doc is None:
                continue

            # Each document must be a dict
            if not isinstance(doc, dict):
                raise ValidationFailed(f"Document {i} is not a dict: {type(doc).__name__}")

            # Check for amplification attacks
            check_amplification(doc)

            docs.append(doc)
    except yaml.YAMLError:
        # Catch YAML errors and re-raise without raw input
        raise ValidationFailed("Invalid YAML syntax") from None
    except RecursionError:
        # PyYAML composes/constructs recursively: deeply nested flow sequences ([[[[...) that fit
        # under max_bytes exhaust the interpreter stack. That is bad input, not a server error.
        raise ValidationFailed("Document is nested too deeply") from None

    return docs


def load_and_lint(
    raw: str,
    *,
    image_allowlist: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], List]:
    """
    Load manifests and run linting.

    Loads via load_manifests, schema-validates each document against MANIFEST_SCHEMA,
    prepends schema-invalid findings (as plain dicts with rule_id="schema-invalid",
    severity="block") before the real lint_documents findings.

    Args:
        raw: Raw YAML text
        image_allowlist: Optional list of allowed image registries

    Returns:
        Tuple of (docs, findings) where findings is a mixed list of:
        - Schema-invalid finding dicts (prepended)
        - Real mlops.config_lint.Finding objects
    """
    # Load manifests
    docs = load_manifests(raw)

    # Collect all findings
    all_findings = []

    # Schema-validate each document and prepend findings
    schema_findings = []
    for doc in docs:
        errors = validate_schema(doc, MANIFEST_SCHEMA)
        if errors:
            for error in errors:
                # Create a Finding-shaped dict for schema errors
                schema_findings.append({
                    "rule_id": "schema-invalid",
                    "severity": "block",
                    "message": error
                })

    # Prepend schema findings
    all_findings.extend(schema_findings)

    # Run the real linter
    lint_findings = lint_documents(docs, image_allowlist)
    all_findings.extend(lint_findings)

    return docs, all_findings
