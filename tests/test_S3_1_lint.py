"""S3.1 slim suite: static Kubernetes manifest linter (mlops.config_lint).
Frozen BEFORE the module exists. Plain-dict manifests only; oracle values are
hand-computed from the API contract (config_lint.py section).
A trailing '*' in an expected path means "path starts with this prefix"
(contract only fixes the exact path for a few rules).
"""
import copy
import dataclasses
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.config_lint import (
    CATEGORY_WEIGHTS,
    Finding,
    lint_context,
    lint_documents,
    lint_manifest,
    lint_score,
    load_manifests_json,
)
from mlops.kernel import ValidationFailed

ACC, RES, ENC, IMG, FS = (
    "access_privileges",
    "resources_probes",
    "encryption_permissions",
    "image_network",
    "filesystem",
)
CP = "spec.template.spec.containers[0]"
PS = "spec.template.spec"


# ----------------------------- builders -----------------------------------
def sc(**kw):
    base = {"readOnlyRootFilesystem": True, "allowPrivilegeEscalation": False}
    base.update(kw)
    return base


def container(**over):
    base = {
        "name": "app",
        "image": "registry.example/app:1.2.3",
        "resources": {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "256Mi"},
        },
        "livenessProbe": {"httpGet": {"path": "/live", "port": 8080}},
        "readinessProbe": {"httpGet": {"path": "/ready", "port": 8080}},
        "securityContext": sc(),
    }
    base.update(over)
    return base


def without(c, *keys):
    c = copy.deepcopy(c)
    for k in keys:
        c.pop(k, None)
    return c


def podspec(containers=None, **extra):
    spec = {"containers": containers if containers is not None else [container()],
            "automountServiceAccountToken": False}
    spec.update(extra)
    return spec


def meta(name="x", ns="prod", **kw):
    m = {"name": name}
    if ns is not None:
        m["namespace"] = ns
    m.update(kw)
    return m


def deployment(containers=None, ns="prod", replicas=None, labels=None, **podextra):
    spec = {"selector": {"matchLabels": labels or {"app": "web"}},
            "template": {"metadata": {"labels": labels or {"app": "web"}},
                         "spec": podspec(containers, **podextra)}}
    if replicas is not None:
        spec["replicas"] = replicas
    return {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": meta("web", ns), "spec": spec}


def pod(containers=None, ns="prod", **podextra):
    return {"apiVersion": "v1", "kind": "Pod", "metadata": meta("p", ns), "spec": podspec(containers, **podextra)}


def wrap(kind, containers=None, **podextra):
    """Workload of `kind` whose pod spec sits at the kind's real path; returns (doc, pod_spec_prefix)."""
    ps = podspec(containers, **podextra)
    md = meta("w", "prod")
    if kind == "Pod":
        return {"kind": "Pod", "metadata": md, "spec": ps}, "spec"
    if kind == "CronJob":
        return ({"kind": "CronJob", "metadata": md,
                 "spec": {"schedule": "* * * * *",
                          "jobTemplate": {"spec": {"template": {"spec": ps}}}}},
                "spec.jobTemplate.spec.template.spec")
    return {"kind": kind, "metadata": md, "spec": {"template": {"spec": ps}}}, "spec.template.spec"


def service_account(auto=None, ns="prod"):
    d = {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": meta("sa", ns)}
    if auto is not None:
        d["automountServiceAccountToken"] = auto
    return d


def crb(role="view"):
    return {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRoleBinding",
            "metadata": {"name": "b"},
            "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": role},
            "subjects": [{"kind": "User", "name": "alice"}]}


def ingress(tls=True, ns="prod"):
    spec = {"rules": [{"host": "a.example"}]}
    if tls:
        spec["tls"] = [{"hosts": ["a.example"], "secretName": "cert"}]
    return {"apiVersion": "networking.k8s.io/v1", "kind": "Ingress", "metadata": meta("ing", ns), "spec": spec}


def hpa(cpu):
    return {"apiVersion": "autoscaling/v1", "kind": "HorizontalPodAutoscaler", "metadata": meta("h"),
            "spec": {"scaleTargetRef": {"kind": "Deployment", "name": "web"},
                     "minReplicas": 1, "maxReplicas": 5, "targetCPUUtilizationPercentage": cpu}}


def secret(ns="prod"):
    return {"apiVersion": "v1", "kind": "Secret", "metadata": meta("s", ns), "type": "Opaque", "data": {}}


def namespace(name="prod"):
    return {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": name}}


def quota(ns="prod", kind="ResourceQuota"):
    spec = {"hard": {"pods": "10"}} if kind == "ResourceQuota" else {"limits": [{"type": "Container"}]}
    return {"apiVersion": "v1", "kind": kind, "metadata": meta("q", ns), "spec": spec}


def pdb(match, ns="prod"):
    return {"apiVersion": "policy/v1", "kind": "PodDisruptionBudget", "metadata": meta("pdb", ns),
            "spec": {"minAvailable": 1, "selector": {"matchLabels": match}}}


def hits(findings, rule_id):
    return [f for f in findings if f.rule_id == rule_id]


def rule_ids(findings):
    return {f.rule_id for f in findings}


def path_ok(actual, expected):
    return actual.startswith(expected[:-1]) if expected.endswith("*") else actual == expected


def with_container(**over):
    return deployment([container(**over)])


# ------------------------------ (1) clean ---------------------------------
def test_clean_deployment_has_no_findings():
    assert lint_manifest(deployment()) == []
    assert lint_manifest(pod()) == []
    assert lint_manifest(service_account(auto=False)) == []
    assert lint_manifest(ingress(tls=True)) == []
    assert lint_manifest(namespace()) == []
    assert lint_manifest(crb()) == []
    assert lint_manifest(hpa(50)) == []


# --------------------------- (2) rule table -------------------------------
RULES = [
    ("cap-all", lambda: with_container(securityContext=sc(capabilities={"add": ["ALL"]})),
     ACC, "block", CP + ".securityContext.capabilities.add"),
    ("priv-escalation", lambda: with_container(securityContext=sc(allowPrivilegeEscalation=True)),
     ACC, "block", CP + ".securityContext.allowPrivilegeEscalation"),
    ("privileged", lambda: with_container(securityContext=sc(privileged=True)),
     ACC, "block", CP + ".securityContext.privileged"),
    ("automount-token", lambda: pod(automountServiceAccountToken=True),
     ACC, "block", "spec.automountServiceAccountToken"),
    ("automount-token", lambda: service_account(auto=True),
     ACC, "block", "automountServiceAccountToken"),
    ("cluster-admin", lambda: crb("cluster-admin"), ACC, "block", "roleRef.name"),
    ("default-namespace", lambda: deployment(ns="default"), ACC, "warn", "metadata.namespace"),
    ("no-resources", lambda: with_container(resources={"requests": {"cpu": "1m"}}),
     RES, "block", CP + ".resources*"),
    ("no-probes", lambda: deployment([without(container(), "livenessProbe")]),
     RES, "warn", CP + "*"),
    ("hpa-threshold", lambda: hpa(95), RES, "block", "spec.targetCPUUtilizationPercentage"),
    ("plaintext-secret-env", lambda: with_container(env=[{"name": "PASSWORD", "value": "hunter2"}]),
     ENC, "block", CP + ".env[0]*"),
    ("ingress-no-tls", lambda: ingress(tls=False), ENC, "block", "spec*"),
    ("latest-tag", lambda: with_container(image="registry.example/app:latest"),
     IMG, "block", CP + ".image"),
    ("writable-rootfs", lambda: with_container(securityContext={"allowPrivilegeEscalation": False}),
     FS, "warn", CP + ".securityContext*"),
    ("hostpath", lambda: deployment(volumes=[{"name": "v", "hostPath": {"path": "/var/run"}}]),
     FS, "block", PS + ".volumes[0]*"),
]


@pytest.mark.parametrize("rid,make,category,severity,path", RULES,
                         ids=[f"{r[0]}-{i}" for i, r in enumerate(RULES)])
def test_rule_table(rid, make, category, severity, path):
    found = hits(lint_manifest(make()), rid)
    assert len(found) == 1, found
    f = found[0]
    assert (f.category, f.severity) == (category, severity)
    assert path_ok(f.path, path), f.path
    assert f.weight == CATEGORY_WEIGHTS[category]
    assert isinstance(f.message, str) and f.message


def test_untrusted_registry_only_with_allowlist():
    doc = with_container(image="evil.io/app:1.0")
    assert hits(lint_manifest(doc), "untrusted-registry") == []
    assert hits(lint_documents([doc]), "untrusted-registry") == []
    found = hits(lint_documents([doc], image_allowlist=["registry.example"]), "untrusted-registry")
    assert len(found) == 1
    assert (found[0].category, found[0].severity) == (IMG, "warn")
    assert path_ok(found[0].path, CP + ".image")
    ok = with_container(image="registry.example/app:1.2.3")
    assert hits(lint_documents([ok], image_allowlist=["registry.example"]), "untrusted-registry") == []


@pytest.mark.parametrize("make", [
    lambda: pod(automountServiceAccountToken=True),
    lambda: deployment(automountServiceAccountToken=True),
    lambda: service_account(auto=True),
])
def test_automount_flagged_only_when_true(make):
    assert len(hits(lint_manifest(make()), "automount-token")) == 1
    assert hits(lint_manifest(pod(automountServiceAccountToken=False)), "automount-token") == []
    assert hits(lint_manifest(service_account(auto=False)), "automount-token") == []


@pytest.mark.parametrize("make,flagged", [
    (lambda: deployment(ns="default"), True),
    (lambda: deployment(ns=None), True),
    (lambda: pod(ns=None), True),
    (lambda: service_account(ns=None), True),
    (lambda: deployment(ns="prod"), False),
    (lambda: crb(), False),
    (lambda: namespace("default"), False),
    (lambda: {"kind": "StorageClass", "metadata": {"name": "fast"}, "provisioner": "x"}, False),
])
def test_default_namespace_rule(make, flagged):
    found = hits(lint_manifest(make()), "default-namespace")
    assert len(found) == (1 if flagged else 0)
    if flagged:
        assert (found[0].category, found[0].severity) == (ACC, "warn")


@pytest.mark.parametrize("res,flagged", [
    (None, True),
    ({}, True),
    ({"requests": {"cpu": "1m"}}, True),
    ({"limits": {"cpu": "1m"}}, True),
    ({"requests": {"cpu": "1m"}, "limits": {"cpu": "2m"}}, False),
])
def test_no_resources_needs_requests_and_limits(res, flagged):
    c = without(container(), "resources") if res is None else container(resources=res)
    assert (len(hits(lint_manifest(deployment([c])), "no-resources")) == 1) is flagged


@pytest.mark.parametrize("cpu,flagged", [(95, True), (5, True), (91, True), (19, True),
                                         (50, False), (20, False), (90, False)])
def test_hpa_threshold_bounds(cpu, flagged):
    assert (len(hits(lint_manifest(hpa(cpu)), "hpa-threshold")) == 1) is flagged


@pytest.mark.parametrize("name,literal,flagged", [
    ("PASSWORD", True, True), ("DB_SECRET", True, True), ("API_KEY", True, True),
    ("apikey", True, True), ("TOKEN", True, True),
    ("PASSWORD", False, False), ("API_KEY", False, False),
    ("LOG_LEVEL", True, False),
])
def test_plaintext_secret_env(name, literal, flagged):
    var = {"name": name, "value": "abc"} if literal else \
        {"name": name, "valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}}}
    found = hits(lint_manifest(with_container(env=[var])), "plaintext-secret-env")
    assert len(found) == (1 if flagged else 0)


@pytest.mark.parametrize("image,flagged", [
    ("registry.example/app:latest", True),
    ("registry.example/app", True),
    ("localhost:5000/app", True),
    ("registry.example/app@sha256:" + "a" * 64, False),
    ("registry.example/app:1.0", False),
    ("app:1.0", False),
])
def test_latest_tag_rule(image, flagged):
    found = hits(lint_manifest(with_container(image=image)), "latest-tag")
    assert len(found) == (1 if flagged else 0)


# ------------------- (3) pod-spec extraction per kind ---------------------
@pytest.mark.parametrize("kind", ["Pod", "Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"])
def test_pod_spec_extraction_all_workload_kinds(kind):
    doc, prefix = wrap(kind, [container(), container(name="side", securityContext=sc(privileged=True))],
                       initContainers=[container(name="init", securityContext=sc(privileged=True))])
    found = hits(lint_manifest(doc), "privileged")
    paths = sorted(f.path for f in found)
    assert paths == sorted([prefix + ".containers[1].securityContext.privileged",
                            prefix + ".initContainers[0].securityContext.privileged"])
    assert lint_manifest(wrap(kind)[0]) == []


# ----------------------- (4) cross-document rules -------------------------
def test_no_quota_rule():
    d = deployment(replicas=1)
    found = hits(lint_documents([d]), "no-quota")
    assert len(found) == 1 and (found[0].category, found[0].severity) == (RES, "warn")
    assert hits(lint_documents([d, quota("prod")]), "no-quota") == []
    assert hits(lint_documents([d, quota("prod", "LimitRange")]), "no-quota") == []
    assert len(hits(lint_documents([d, quota("other")]), "no-quota")) == 1
    assert hits(lint_documents([]), "no-quota") == []


def test_no_pdb_rule():
    q = quota("prod")
    d3 = deployment(replicas=3, labels={"app": "web", "tier": "fe"})
    found = hits(lint_documents([d3, q]), "no-pdb")
    assert len(found) == 1 and (found[0].category, found[0].severity) == (RES, "warn")
    assert hits(lint_documents([d3, q, pdb({"app": "web"})]), "no-pdb") == []
    assert len(hits(lint_documents([d3, q, pdb({"app": "db"})]), "no-pdb")) == 1
    assert len(hits(lint_documents([d3, q, pdb({"app": "web"}, ns="other")]), "no-pdb")) == 1
    assert len(hits(lint_documents([deployment(replicas=2), q]), "no-pdb")) == 1
    assert hits(lint_documents([deployment(replicas=1), q]), "no-pdb") == []


def test_lint_documents_deterministic_and_ordered():
    docs = [with_container(image="a/app:latest", securityContext=sc(privileged=True)),
            pod(containers=[container(securityContext=sc(allowPrivilegeEscalation=True))]),
            quota("prod")]
    first, second = lint_documents(docs), lint_documents(copy.deepcopy(docs))
    assert first == second and first
    expected = []
    for d in docs[:2]:
        expected += sorted(lint_manifest(d), key=lambda f: (f.rule_id, f.path))
    assert first == expected


# ------------------------------ (5) score ---------------------------------
def F(category, severity, rule="r", weight=None):
    return Finding(rule_id=rule, category=category, subcategory="s", severity=severity,
                   path="p", message="m", weight=CATEGORY_WEIGHTS[category] if weight is None else weight)


def test_lint_score_hand_computed():
    assert lint_score([]) == 0.0
    assert lint_score([F(ACC, "block")]) == pytest.approx(0.238)
    assert lint_score([F(FS, "warn")]) == pytest.approx(0.077)
    assert lint_score([F(ACC, "block"), F(FS, "warn")]) == pytest.approx(0.315)
    assert lint_score([F(RES, "block", weight=0.4), F(RES, "warn", weight=0.4)]) == pytest.approx(0.6)


def test_category_weights_table():
    assert CATEGORY_WEIGHTS == {ACC: 0.238, RES: 0.215, ENC: 0.203, IMG: 0.190, FS: 0.154}
    assert sum(CATEGORY_WEIGHTS.values()) == pytest.approx(1.0)


def test_lint_context_counts():
    fs = [F(ACC, "block"), F(FS, "warn"), F(IMG, "warn")]
    ctx = lint_context(fs)
    assert set(ctx) == {"lint_blockers", "lint_warnings", "lint_score"}
    assert ctx["lint_blockers"] == 1 and ctx["lint_warnings"] == 2
    assert ctx["lint_score"] == pytest.approx(0.238 + 0.5 * (0.154 + 0.190))
    assert lint_context([]) == {"lint_blockers": 0, "lint_warnings": 0, "lint_score": 0.0}


# ------------------------- (6) load_manifests_json ------------------------
def test_load_manifests_accepts_three_shapes():
    a, b = deployment(), pod()
    assert load_manifests_json(json.dumps(a)) == [a]
    assert load_manifests_json(json.dumps([a, b])) == [a, b]
    assert load_manifests_json(json.dumps({"kind": "List", "items": [a, b]})) == [a, b]
    assert load_manifests_json("[]") == []


@pytest.mark.parametrize("text", [
    "{not json", "", "5", '"str"', "null", "true", "[{}, 1]", '[{"kind":"Pod"}, null]',
    '{"kind":"List","items":[1]}', '{"kind":"List","items":"x"}',
])
def test_load_manifests_rejects_bad_input(text):
    with pytest.raises(ValidationFailed):
        load_manifests_json(text)


def test_load_manifests_limits():
    assert len(load_manifests_json(json.dumps([{}] * 5000))) == 5000
    with pytest.raises(ValidationFailed):
        load_manifests_json(json.dumps([{}] * 5001))
    with pytest.raises(ValidationFailed):
        load_manifests_json(json.dumps({"kind": "List", "items": [{}] * 5001}))
    deep = lambda n: '{"a":' * n + "1" + "}" * n
    assert len(load_manifests_json(deep(10))) == 1
    with pytest.raises(ValidationFailed):
        load_manifests_json(deep(60))
    with pytest.raises(ValidationFailed):
        load_manifests_json("[" * 50000 + "]" * 50000)  # must not surface RecursionError
    with pytest.raises(ValidationFailed):
        load_manifests_json(deep(5000))


# --------------------------- (7) hostile inputs ---------------------------
def _pod_with(**spec):
    return {"kind": "Pod", "metadata": {"name": "p", "namespace": "prod"}, "spec": spec}


def _deep(n):
    d = {}
    for _ in range(n):
        d = {"x": d}
    return d


HOSTILE = [
    None, [], "Pod", 42, {}, {"metadata": {}}, {"kind": 5}, {"kind": None}, {"kind": ["Pod"]},
    {"kind": "Pod"}, {"kind": "Pod", "metadata": "x"}, {"kind": "Pod", "metadata": {"namespace": 7}},
    {"kind": "Pod", "spec": "x"}, {"kind": "Pod", "spec": []},
    _pod_with(containers="app"), _pod_with(containers={"name": "a"}), _pod_with(containers=[None]),
    _pod_with(containers=["x"]), _pod_with(containers=[[]]),
    _pod_with(containers=[{"env": "A=B"}]), _pod_with(containers=[{"env": [None, 5, "x"]}]),
    _pod_with(containers=[{"env": [{"name": 5, "value": 1}]}]),
    _pod_with(containers=[{"resources": "big"}]), _pod_with(containers=[{"resources": {"requests": "x", "limits": 3}}]),
    _pod_with(containers=[{"image": 5}]), _pod_with(containers=[{"image": None}]),
    _pod_with(containers=[{"securityContext": "root"}]),
    _pod_with(containers=[{"securityContext": {"capabilities": "ALL"}}]),
    _pod_with(containers=[{"securityContext": {"capabilities": {"add": "ALL"}}}]),
    _pod_with(containers=[{"securityContext": {"capabilities": {"add": [None, 3]}}}]),
    _pod_with(containers=[{"livenessProbe": "x"}]), _pod_with(initContainers="x"),
    _pod_with(volumes="x"), _pod_with(volumes=[None, 3]),
    {"kind": "Deployment", "spec": {"template": "x"}}, {"kind": "Deployment", "spec": {"template": {"spec": []}}},
    {"kind": "CronJob", "spec": {"jobTemplate": []}},
    {"kind": "ClusterRoleBinding", "roleRef": "cluster-admin"}, {"kind": "ClusterRoleBinding", "roleRef": None},
    {"kind": "Ingress", "spec": "x"}, {"kind": "HorizontalPodAutoscaler", "spec": {"targetCPUUtilizationPercentage": "hi"}},
    {"kind": "HorizontalPodAutoscaler", "spec": {"targetCPUUtilizationPercentage": True}},
    {"kind": "HorizontalPodAutoscaler", "spec": {"targetCPUUtilizationPercentage": float("nan")}},
    {"kind": "ServiceAccount", "automountServiceAccountToken": "true"},
    _pod_with(containers=[{"name": "a", "junk": _deep(3000)}]),
]


@pytest.mark.parametrize("doc", HOSTILE, ids=[str(i) for i in range(len(HOSTILE))])
def test_hostile_manifest_never_crashes(doc):
    try:
        out = lint_manifest(doc)
    except ValidationFailed:
        return
    assert isinstance(out, list) and all(isinstance(f, Finding) for f in out)


@pytest.mark.parametrize("bad", [None, "x", 5, {"kind": "Pod"}, [None], ["x"], [[]], [5]])
def test_lint_documents_rejects_bad_container(bad):
    with pytest.raises(ValidationFailed):
        lint_documents(bad)


def test_lint_documents_rejects_bad_allowlist():
    for bad in ("registry.example", 5, [None], [5]):
        with pytest.raises(ValidationFailed):
            lint_documents([deployment()], image_allowlist=bad)


# ------------------------ (8) immutability / purity -----------------------
def test_findings_are_frozen():
    f = hits(lint_manifest(with_container(image="app:latest")), "latest-tag")[0]
    assert dataclasses.is_dataclass(f)
    for field in ("severity", "path", "weight", "rule_id"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(f, field, "x")
    assert isinstance(f.weight, float) and f.subcategory


def test_inputs_never_mutated():
    bad = deployment([container(image="app:latest", securityContext=sc(privileged=True), env=[{"name": "TOKEN", "value": "t"}]),
                      {"name": "second"}], ns=None, replicas=3)
    docs = [bad, crb("cluster-admin"), ingress(tls=False), hpa(99)]
    before_bad, before_docs = copy.deepcopy(bad), copy.deepcopy(docs)
    lint_manifest(bad)
    assert bad == before_bad
    lint_documents(docs, image_allowlist=["registry.example"])
    assert docs == before_docs
    text = json.dumps(docs)
    assert load_manifests_json(text) == docs
