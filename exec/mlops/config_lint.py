"""
S3.1 Static Kubernetes manifest linter.

Analyzes plain-dict Kubernetes manifests for security, resource, and operational
misconfigurations. Categories and weights based on 2609.27030 (Stack Overflow
analysis of K8s misconfiguration patterns).
"""

import dataclasses
import json
import re
import sys
from typing import Any, Dict, List, Optional, Set

from mlops.kernel import ValidationFailed

__all__ = [
    "CATEGORY_WEIGHTS",
    "Finding",
    "lint_context",
    "lint_documents",
    "lint_manifest",
    "lint_score",
    "load_manifests_json",
]

# Category weights from K8s misconfiguration research (sixb-q1, 2609.27030)
CATEGORY_WEIGHTS = {
    "access_privileges": 0.238,
    "resources_probes": 0.215,
    "encryption_permissions": 0.203,
    "image_network": 0.190,
    "filesystem": 0.154,
}


@dataclasses.dataclass(frozen=True)
class Finding:
    """A single linting finding on a Kubernetes manifest.

    Frozen dataclass to prevent mutation of findings after creation.
    """
    rule_id: str
    category: str
    subcategory: str
    severity: str  # "block" or "warn"
    path: str  # JSON-pointer-like path (e.g. "spec.template.spec.containers[0].image")
    message: str
    weight: float




def _validate_depth(obj: Any, max_depth: int = 50, current_depth: int = 0) -> bool:
    """Check if object nesting depth is within bounds."""
    if current_depth > max_depth:
        return False
    if isinstance(obj, dict):
        for v in obj.values():
            if not _validate_depth(v, max_depth, current_depth + 1):
                return False
    elif isinstance(obj, list):
        for item in obj:
            if not _validate_depth(item, max_depth, current_depth + 1):
                return False
    return True


def _extract_int(val: Any) -> Optional[int]:
    """Safely extract an integer, returning None if not an int."""
    if isinstance(val, bool):
        return None  # bool is subclass of int but we reject it
    if isinstance(val, int):
        return val
    return None




def _extract_str(val: Any) -> Optional[str]:
    """Safely extract a string, returning None if not a str."""
    if isinstance(val, str):
        return val
    return None


def _extract_bool(val: Any) -> Optional[bool]:
    """Safely extract a bool, returning None if not a bool."""
    if isinstance(val, bool):
        return val
    return None


def _extract_dict(val: Any) -> Optional[Dict]:
    """Safely extract a dict, returning None if not a dict."""
    if isinstance(val, dict):
        return val
    return None




def _get_kind(doc: Any) -> Optional[str]:
    """Extract kind from document, validating basic structure."""
    if not isinstance(doc, dict):
        return None
    kind = doc.get("kind")
    if not isinstance(kind, str):
        return None
    return kind


def _get_metadata(doc: Any) -> Optional[Dict]:
    """Extract and validate metadata section."""
    if not isinstance(doc, dict):
        return None
    meta = doc.get("metadata")
    if not isinstance(meta, dict):
        return None
    return meta


def _get_namespace(meta: Dict) -> Optional[str]:
    """Extract namespace from metadata, with validation."""
    ns = meta.get("namespace")
    if ns is None:
        return None
    if not isinstance(ns, str):
        return None
    return ns


def _get_pod_spec_path(kind: str) -> Optional[str]:
    """Return the dot-path to pod spec for a given kind, or None if not a workload."""
    paths = {
        "Pod": "spec",
        "Deployment": "spec.template.spec",
        "StatefulSet": "spec.template.spec",
        "DaemonSet": "spec.template.spec",
        "Job": "spec.template.spec",
        "CronJob": "spec.jobTemplate.spec.template.spec",
    }
    return paths.get(kind)


def _navigate_path(doc: Any, path: str) -> Any:
    """Navigate a dot-separated path with array indexing. Returns None if path invalid."""
    if not isinstance(doc, dict):
        return None

    current = doc
    parts = path.split(".")
    for part in parts:
        # Handle array indexing like "containers[0]"
        if "[" in part:
            key = part[:part.index("[")]
            index_str = part[part.index("[") + 1:part.index("]")]
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if current is None:
                return None
            try:
                idx = int(index_str)
            except (ValueError, TypeError):
                return None
            if not isinstance(current, list):
                return None
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
        else:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
            if current is None:
                return None
    return current


def _extract_pod_spec(doc: Any) -> Optional[Dict]:
    """Extract the pod spec from a workload document."""
    if not isinstance(doc, dict):
        return None

    kind = _get_kind(doc)
    if kind is None:
        return None

    pod_spec_path = _get_pod_spec_path(kind)
    if pod_spec_path is None:
        return None

    spec = _navigate_path(doc, pod_spec_path)
    if not isinstance(spec, dict):
        return None
    return spec


def _validate_container_list(containers: Any) -> Optional[List[Dict]]:
    """Validate that containers is a list, returning it as-is.

    Validation of individual items happens per-item during iteration.
    This allows processing of valid items even if some are malformed.
    """
    if not isinstance(containers, list):
        return None
    return containers


def _validate_env_list(env: Any) -> Optional[List[Dict]]:
    """Validate that env is a list, returning it as-is.

    Validation of individual items happens per-item during iteration.
    This allows processing of valid items even if some are malformed.
    """
    if not isinstance(env, list):
        return None
    return env


def _validate_volumes_list(volumes: Any) -> Optional[List[Dict]]:
    """Validate that volumes is a list, returning it as-is.

    Validation of individual items happens per-item during iteration.
    This allows processing of valid items even if some are malformed.
    """
    if not isinstance(volumes, list):
        return None
    return volumes


def _matches_secret_name(name: str) -> bool:
    """Check if env var name matches secret pattern (case-insensitive).

    Matches names containing PASSWORD, SECRET, TOKEN, or API*KEY patterns.
    """
    name_upper = name.upper()
    # Check for exact patterns or patterns with underscores/prefixes
    if "PASSWORD" in name_upper:
        return True
    if "SECRET" in name_upper:
        return True
    if "TOKEN" in name_upper:
        return True
    # API_KEY or APIKEY patterns
    if "API" in name_upper and "KEY" in name_upper:
        return True
    return False


def _has_plaintext_value(env_item: Dict) -> bool:
    """Check if env item has a plaintext 'value' (not valueFrom)."""
    return "value" in env_item and "valueFrom" not in env_item


def _get_image_registry(image: str) -> str:
    """Extract registry from image string."""
    # Format: [registry/]name[:tag|@digest]
    if "/" not in image:
        return "docker.io"  # default registry
    return image.split("/")[0]


def _has_valid_tag_or_digest(image: str) -> bool:
    """Check if image has a tag (not 'latest') or digest."""
    # Remove registry
    if "/" in image:
        image = image.split("/", 1)[1]

    # Check for digest (@sha256:...)
    if "@" in image:
        return True

    # Check for tag
    if ":" in image:
        tag = image.split(":")[-1]
        return tag != "latest"

    # No tag means :latest is implied
    return False


def lint_manifest(doc: Any) -> List[Finding]:
    """Lint a single Kubernetes manifest document.

    Args:
        doc: Plain dict manifest as parsed from JSON

    Returns:
        List of Finding objects (frozen, never mutated)
    """
    findings = []

    # Basic validation
    if not isinstance(doc, dict):
        return findings

    kind = _get_kind(doc)
    if kind is None:
        return findings

    # Depth check
    if not _validate_depth(doc, max_depth=50):
        raise ValidationFailed("Manifest exceeds maximum nesting depth")

    meta = _get_metadata(doc)

    # ======================== Namespace-scoped checks ========================

    # Rule: default-namespace
    # Only check for namespaced kinds (not ClusterRoleBinding, ClusterRole, Namespace, StorageClass)
    cluster_scoped = {"ClusterRoleBinding", "ClusterRole", "Namespace", "StorageClass"}
    if kind not in cluster_scoped:
        ns = _get_namespace(meta) if meta else None
        if ns is None or ns == "default":
            findings.append(Finding(
                rule_id="default-namespace",
                category="access_privileges",
                subcategory="default-ns",
                severity="warn",
                path="metadata.namespace",
                message="Resource should not use default namespace",
                weight=CATEGORY_WEIGHTS["access_privileges"],
            ))

    # ======================== Pod Spec checks (workloads) ========================

    pod_spec = _extract_pod_spec(doc)
    if pod_spec:
        pod_spec_path = _get_pod_spec_path(kind)

        # Check automountServiceAccountToken in pod spec
        auto_mount = pod_spec.get("automountServiceAccountToken")
        if _extract_bool(auto_mount) is True:
            findings.append(Finding(
                rule_id="automount-token",
                category="access_privileges",
                subcategory="service-account",
                severity="block",
                path=f"{pod_spec_path}.automountServiceAccountToken",
                message="Service account token should not be automatically mounted",
                weight=CATEGORY_WEIGHTS["access_privileges"],
            ))

        # Check containers
        containers = _validate_container_list(pod_spec.get("containers"))
        if containers:
            for c_idx, container in enumerate(containers):
                if not isinstance(container, dict):
                    continue

                c_path = f"{pod_spec_path}.containers[{c_idx}]"

                # Rule: no-resources
                resources = container.get("resources")
                if resources is None or not isinstance(resources, dict):
                    has_requests = False
                    has_limits = False
                else:
                    requests = resources.get("requests")
                    limits = resources.get("limits")
                    has_requests = isinstance(requests, dict) and bool(requests)
                    has_limits = isinstance(limits, dict) and bool(limits)

                if not (has_requests and has_limits):
                    findings.append(Finding(
                        rule_id="no-resources",
                        category="resources_probes",
                        subcategory="resources",
                        severity="block",
                        path=f"{c_path}.resources*",
                        message="Container must have both requests and limits",
                        weight=CATEGORY_WEIGHTS["resources_probes"],
                    ))

                # Rule: no-probes (liveness or readiness missing)
                liveness = container.get("livenessProbe")
                readiness = container.get("readinessProbe")
                if liveness is None or readiness is None:
                    findings.append(Finding(
                        rule_id="no-probes",
                        category="resources_probes",
                        subcategory="probes",
                        severity="warn",
                        path=f"{c_path}*",
                        message="Container should have liveness and readiness probes",
                        weight=CATEGORY_WEIGHTS["resources_probes"],
                    ))

                # Rule: plaintext-secret-env
                env = _validate_env_list(container.get("env"))
                if env:
                    for e_idx, env_item in enumerate(env):
                        # Skip malformed items (not a dict or missing name)
                        if not isinstance(env_item, dict) or "name" not in env_item:
                            continue

                        env_name = _extract_str(env_item.get("name"))
                        if env_name and _matches_secret_name(env_name):
                            if _has_plaintext_value(env_item):
                                findings.append(Finding(
                                    rule_id="plaintext-secret-env",
                                    category="encryption_permissions",
                                    subcategory="secret-env",
                                    severity="block",
                                    path=f"{c_path}.env[{e_idx}]*",
                                    message=f"Secret-like env var '{env_name}' should not have plaintext value",
                                    weight=CATEGORY_WEIGHTS["encryption_permissions"],
                                ))

                # Rule: latest-tag
                image = _extract_str(container.get("image"))
                if image:
                    if not _has_valid_tag_or_digest(image):
                        findings.append(Finding(
                            rule_id="latest-tag",
                            category="image_network",
                            subcategory="image-tag",
                            severity="block",
                            path=f"{c_path}.image",
                            message="Image must use a specific tag or digest, not :latest",
                            weight=CATEGORY_WEIGHTS["image_network"],
                        ))

                # Rule: writable-rootfs
                sec_ctx = container.get("securityContext")
                if sec_ctx is None or not isinstance(sec_ctx, dict):
                    findings.append(Finding(
                        rule_id="writable-rootfs",
                        category="filesystem",
                        subcategory="rootfs",
                        severity="warn",
                        path=f"{c_path}.securityContext*",
                        message="Container should have readOnlyRootFilesystem set to true",
                        weight=CATEGORY_WEIGHTS["filesystem"],
                    ))
                else:
                    ro_root = sec_ctx.get("readOnlyRootFilesystem")
                    if _extract_bool(ro_root) is not True:
                        findings.append(Finding(
                            rule_id="writable-rootfs",
                            category="filesystem",
                            subcategory="rootfs",
                            severity="warn",
                            path=f"{c_path}.securityContext*",
                            message="Container should have readOnlyRootFilesystem set to true",
                            weight=CATEGORY_WEIGHTS["filesystem"],
                        ))

                    # Rule: priv-escalation
                    priv_esc = sec_ctx.get("allowPrivilegeEscalation")
                    if _extract_bool(priv_esc) is True:
                        findings.append(Finding(
                            rule_id="priv-escalation",
                            category="access_privileges",
                            subcategory="privilege",
                            severity="block",
                            path=f"{c_path}.securityContext.allowPrivilegeEscalation",
                            message="Privilege escalation should not be allowed",
                            weight=CATEGORY_WEIGHTS["access_privileges"],
                        ))

                    # Rule: privileged
                    privileged = sec_ctx.get("privileged")
                    if _extract_bool(privileged) is True:
                        findings.append(Finding(
                            rule_id="privileged",
                            category="access_privileges",
                            subcategory="privilege",
                            severity="block",
                            path=f"{c_path}.securityContext.privileged",
                            message="Container should not run in privileged mode",
                            weight=CATEGORY_WEIGHTS["access_privileges"],
                        ))

                    # Rule: cap-all
                    capabilities = sec_ctx.get("capabilities")
                    if isinstance(capabilities, dict):
                        cap_add = capabilities.get("add")
                        if isinstance(cap_add, list):
                            for cap in cap_add:
                                if cap == "ALL":
                                    findings.append(Finding(
                                        rule_id="cap-all",
                                        category="access_privileges",
                                        subcategory="capabilities",
                                        severity="block",
                                        path=f"{c_path}.securityContext.capabilities.add",
                                        message="Should not add ALL capabilities",
                                        weight=CATEGORY_WEIGHTS["access_privileges"],
                                    ))
                                    break

        # Check initContainers
        init_containers = _validate_container_list(pod_spec.get("initContainers"))
        if init_containers:
            for c_idx, container in enumerate(init_containers):
                # Skip malformed items (not a dict)
                if not isinstance(container, dict):
                    continue

                c_path = f"{pod_spec_path}.initContainers[{c_idx}]"

                # Check for privileged in init containers
                sec_ctx = container.get("securityContext")
                if isinstance(sec_ctx, dict):
                    privileged = sec_ctx.get("privileged")
                    if _extract_bool(privileged) is True:
                        findings.append(Finding(
                            rule_id="privileged",
                            category="access_privileges",
                            subcategory="privilege",
                            severity="block",
                            path=f"{c_path}.securityContext.privileged",
                            message="Init container should not run in privileged mode",
                            weight=CATEGORY_WEIGHTS["access_privileges"],
                        ))

        # Rule: hostpath (volumes)
        volumes = _validate_volumes_list(pod_spec.get("volumes"))
        if volumes:
            for v_idx, volume in enumerate(volumes):
                # Skip malformed items (not a dict)
                if not isinstance(volume, dict):
                    continue

                hostpath = volume.get("hostPath")
                if isinstance(hostpath, dict):
                    findings.append(Finding(
                        rule_id="hostpath",
                        category="filesystem",
                        subcategory="hostpath",
                        severity="block",
                        path=f"{pod_spec_path}.volumes[{v_idx}]*",
                        message="Should not mount host paths",
                        weight=CATEGORY_WEIGHTS["filesystem"],
                    ))

    # ======================== ServiceAccount checks ========================

    if kind == "ServiceAccount":
        auto_mount = doc.get("automountServiceAccountToken")
        if _extract_bool(auto_mount) is True:
            findings.append(Finding(
                rule_id="automount-token",
                category="access_privileges",
                subcategory="service-account",
                severity="block",
                path="automountServiceAccountToken",
                message="Service account token should not be automatically mounted",
                weight=CATEGORY_WEIGHTS["access_privileges"],
            ))

    # ======================== RBAC checks ========================

    if kind == "ClusterRoleBinding":
        role_ref = doc.get("roleRef")
        if isinstance(role_ref, dict):
            role_name = _extract_str(role_ref.get("name"))
            if role_name == "cluster-admin":
                findings.append(Finding(
                    rule_id="cluster-admin",
                    category="access_privileges",
                    subcategory="rbac",
                    severity="block",
                    path="roleRef.name",
                    message="Should not bind cluster-admin role",
                    weight=CATEGORY_WEIGHTS["access_privileges"],
                ))

    # ======================== Ingress checks ========================

    if kind == "Ingress":
        spec = doc.get("spec")
        if isinstance(spec, dict):
            tls = spec.get("tls")
            # TLS is required (if tls key is missing or empty, it's a violation)
            if tls is None or (isinstance(tls, list) and len(tls) == 0):
                findings.append(Finding(
                    rule_id="ingress-no-tls",
                    category="encryption_permissions",
                    subcategory="tls",
                    severity="block",
                    path="spec*",
                    message="Ingress should have TLS configured",
                    weight=CATEGORY_WEIGHTS["encryption_permissions"],
                ))

    # ======================== HPA checks ========================

    if kind == "HorizontalPodAutoscaler":
        spec = doc.get("spec")
        if isinstance(spec, dict):
            cpu_target = spec.get("targetCPUUtilizationPercentage")
            cpu_int = _extract_int(cpu_target)
            if cpu_int is not None:
                # Valid range is 20-90 (inclusive)
                if cpu_int < 20 or cpu_int > 90:
                    findings.append(Finding(
                        rule_id="hpa-threshold",
                        category="resources_probes",
                        subcategory="hpa",
                        severity="block",
                        path="spec.targetCPUUtilizationPercentage",
                        message="HPA target CPU utilization should be between 20 and 90",
                        weight=CATEGORY_WEIGHTS["resources_probes"],
                    ))

    return findings


def lint_documents(docs: Any, image_allowlist: Optional[Any] = None) -> List[Finding]:
    """Lint multiple Kubernetes documents with cross-document rules.

    Args:
        docs: List of plain dict manifests
        image_allowlist: Optional list of allowed image registries (for untrusted-registry rule)

    Returns:
        List of Finding objects, deterministically ordered
    """
    # Validate inputs
    if not isinstance(docs, list):
        raise ValidationFailed("docs must be a list")

    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            raise ValidationFailed(f"Document {i} is not a dict")

    if image_allowlist is not None:
        if not isinstance(image_allowlist, list):
            raise ValidationFailed("image_allowlist must be a list or None")
        for i, item in enumerate(image_allowlist):
            if not isinstance(item, str):
                raise ValidationFailed(f"image_allowlist[{i}] is not a string")

    # Lint each document once, store results for reuse
    doc_findings_list = []
    all_findings = []
    untrusted_registry_findings = []  # Collect untrusted-registry separately to preserve doc_idx
    for doc_idx, doc in enumerate(docs):
        doc_findings = lint_manifest(doc)
        doc_findings_list.append(doc_findings)
        all_findings.extend(doc_findings)

    # Add untrusted-registry findings if allowlist is provided
    if image_allowlist:
        for doc_idx, doc in enumerate(docs):
            if not isinstance(doc, dict):
                continue

            kind = _get_kind(doc)
            if kind is None:
                continue

            pod_spec = _extract_pod_spec(doc)
            if pod_spec:
                pod_spec_path = _get_pod_spec_path(kind)
                containers = _validate_container_list(pod_spec.get("containers"))
                if containers:
                    for c_idx, container in enumerate(containers):
                        if not isinstance(container, dict):
                            continue

                        image = _extract_str(container.get("image"))
                        if image:
                            registry = _get_image_registry(image)
                            if registry not in image_allowlist:
                                c_path = f"{pod_spec_path}.containers[{c_idx}]"
                                finding = Finding(
                                    rule_id="untrusted-registry",
                                    category="image_network",
                                    subcategory="registry",
                                    severity="warn",
                                    path=f"{c_path}.image",
                                    message=f"Image registry '{registry}' not in allowlist",
                                    weight=CATEGORY_WEIGHTS["image_network"],
                                )
                                # Store with doc_idx to distinguish findings from different documents
                                untrusted_registry_findings.append((doc_idx, finding))
                                all_findings.append(finding)

    # ======================== Cross-document rules ========================

    # Rule: no-quota
    # Find all workloads (things with pod specs) and their namespaces
    workload_kinds = {"Pod", "Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"}
    quota_kinds = {"ResourceQuota", "LimitRange"}

    namespaces_with_quota: Set[str] = set()
    workload_namespaces: Set[str] = set()

    for doc in docs:
        if not isinstance(doc, dict):
            continue

        kind = _get_kind(doc)
        if kind is None:
            continue

        # Check for quotas
        if kind in quota_kinds:
            meta = _get_metadata(doc)
            if meta:
                ns = _get_namespace(meta)
                if ns:
                    namespaces_with_quota.add(ns)

        # Check for workloads
        if kind in workload_kinds:
            meta = _get_metadata(doc)
            if meta:
                ns = _get_namespace(meta)
                if ns:
                    workload_namespaces.add(ns)

    # Flag namespaces with workloads but no quota
    for ns in workload_namespaces:
        if ns not in namespaces_with_quota:
            all_findings.append(Finding(
                rule_id="no-quota",
                category="resources_probes",
                subcategory="quota",
                severity="warn",
                path=f"metadata.namespace[{ns}]",  # Include namespace for unique deduplication
                message=f"Namespace '{ns}' has workloads but no ResourceQuota or LimitRange",
                weight=CATEGORY_WEIGHTS["resources_probes"],
            ))

    # Rule: no-pdb
    # Find Deployments/StatefulSets with replicas > 1
    pdb_kinds = {"Deployment", "StatefulSet"}
    deployments_needing_pdb: Dict[str, Dict] = {}  # (ns, labels) -> doc info

    for doc in docs:
        if not isinstance(doc, dict):
            continue

        kind = _get_kind(doc)
        if kind not in pdb_kinds:
            continue

        spec = doc.get("spec")
        if not isinstance(spec, dict):
            continue

        replicas = _extract_int(spec.get("replicas"))
        if replicas is None:
            replicas = 1  # Default replicas is 1

        # Only flag if replicas > 1
        if replicas <= 1:
            continue

        meta = _get_metadata(doc)
        if not meta:
            continue

        ns = _get_namespace(meta)
        if not ns:
            continue

        template = spec.get("template")
        if not isinstance(template, dict):
            continue

        template_meta = template.get("metadata")
        if not isinstance(template_meta, dict):
            continue

        labels = template_meta.get("labels")
        if not isinstance(labels, dict):
            continue

        deployments_needing_pdb[(ns, tuple(sorted(labels.items())))] = {
            "ns": ns,
            "labels": labels,
        }

    # Find PodDisruptionBudgets
    pdbs: List[Dict] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue

        kind = _get_kind(doc)
        if kind != "PodDisruptionBudget":
            continue

        meta = _get_metadata(doc)
        if not meta:
            continue

        ns = _get_namespace(meta)
        if not ns:
            continue

        spec = doc.get("spec")
        if not isinstance(spec, dict):
            continue

        selector = spec.get("selector")
        if not isinstance(selector, dict):
            continue

        match_labels = selector.get("matchLabels")
        if not isinstance(match_labels, dict):
            continue

        pdbs.append({
            "ns": ns,
            "match_labels": match_labels,
        })

    # Check if each deployment has a matching PDB
    for (ns, labels_tuple), deploy_info in deployments_needing_pdb.items():
        labels_dict = dict(labels_tuple)
        found_pdb = False

        for pdb in pdbs:
            if pdb["ns"] != ns:
                continue

            # Check if PDB's matchLabels is a subset of deployment's labels
            match_labels = pdb["match_labels"]
            if all(labels_dict.get(k) == v for k, v in match_labels.items()):
                found_pdb = True
                break

        if not found_pdb:
            all_findings.append(Finding(
                rule_id="no-pdb",
                category="resources_probes",
                subcategory="pdb",
                severity="warn",
                path=f"metadata.namespace[{ns}]",  # Include namespace for unique deduplication
                message=f"Deployment/StatefulSet in namespace '{ns}' should have a PodDisruptionBudget",
                weight=CATEGORY_WEIGHTS["resources_probes"],
            ))

    # Sort findings deterministically by document index, rule_id, then path
    # Track which document each finding came from (len(docs) for true cross-doc findings)
    findings_with_index = []

    # Add single-document findings (reuse stored results)
    for doc_idx, doc_findings in enumerate(doc_findings_list):
        for finding in doc_findings:
            findings_with_index.append((doc_idx, finding))

    # Add untrusted-registry findings (these belong to specific documents, not cross-doc)
    findings_with_index.extend(untrusted_registry_findings)

    # Add cross-document findings with a special index (len(docs))
    # Deduplicate by (doc_index, rule_id, path) instead of rule_id alone
    # Note: untrusted-registry is NOT in cross_doc_findings anymore since we handle it separately
    cross_doc_findings = [f for f in all_findings if f.rule_id in ("no-quota", "no-pdb")]
    added_keys = {(doc_idx, f.rule_id, f.path) for doc_idx, f in findings_with_index}

    for finding in cross_doc_findings:
        dedup_key = (len(docs), finding.rule_id, finding.path)
        if dedup_key not in added_keys:
            findings_with_index.append((len(docs), finding))
            added_keys.add(dedup_key)

    # Sort by (doc_index, rule_id, path)
    findings_with_index.sort(key=lambda x: (x[0], x[1].rule_id, x[1].path))

    return [f for _, f in findings_with_index]


def lint_score(findings: List[Finding]) -> float:
    """Compute a lint score from findings.

    Score = sum(weight * (1.0 if severity=="block" else 0.5)), clamped to [0, 1]

    Args:
        findings: List of Finding objects

    Returns:
        Float score in [0, 1]
    """
    score = 0.0
    for finding in findings:
        severity_mult = 1.0 if finding.severity == "block" else 0.5
        score += finding.weight * severity_mult
    return min(1.0, score)


def lint_context(findings: List[Finding]) -> Dict[str, Any]:
    """Extract context from findings for policy evaluation.

    Args:
        findings: List of Finding objects

    Returns:
        Dict with keys: lint_blockers (int), lint_warnings (int), lint_score (float)
    """
    blockers = sum(1 for f in findings if f.severity == "block")
    warnings = sum(1 for f in findings if f.severity == "warn")
    score = lint_score(findings)

    return {
        "lint_blockers": blockers,
        "lint_warnings": warnings,
        "lint_score": score,
    }


def load_manifests_json(text: str) -> List[Dict]:
    """Load Kubernetes manifests from JSON text.

    Accepts three formats:
    1. A single manifest object: {...}
    2. An array of manifests: [{...}, {...}]
    3. A List kind: {"kind": "List", "items": [{...}, {...}]}

    Raises ValidationFailed for invalid input, exceeding limits, or deep nesting.

    Args:
        text: JSON string

    Returns:
        List of manifest dicts
    """
    # Validate depth during parsing
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValidationFailed(f"Invalid JSON: {e}")
    except RecursionError:
        raise ValidationFailed("JSON nesting too deep")

    # Validate depth of parsed object
    if not _validate_depth(obj, max_depth=50):
        raise ValidationFailed("Manifest nesting exceeds maximum depth of 50")

    # Parse the three accepted formats
    manifests = []

    if isinstance(obj, list):
        # Format 2: Array of manifests
        if len(obj) > 5000:
            raise ValidationFailed("Too many manifests (max 5000)")
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                raise ValidationFailed(f"Item {i} is not a dict")
        manifests = obj

    elif isinstance(obj, dict):
        kind = obj.get("kind")
        if kind == "List":
            # Format 3: List kind
            items = obj.get("items")
            if not isinstance(items, list):
                raise ValidationFailed("List.items must be a list")
            if len(items) > 5000:
                raise ValidationFailed("Too many manifests (max 5000)")
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    raise ValidationFailed(f"List.items[{i}] is not a dict")
            manifests = items
        else:
            # Format 1: Single manifest object
            manifests = [obj]

    else:
        raise ValidationFailed("Input must be a JSON object, array, or List kind")

    return manifests
