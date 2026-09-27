# Policy layer contract

Authority: `exec/mlops/policy_engine.py` and `exec/mlops/policy_store.py` decide everything at
runtime. casbin is a **test oracle only** (`tests/oracle/casbin_oracle.py`); the server never
imports it. Tests guarding this contract: `tests/test_policy_layer_contract.py`
(raw layer, boundary and matrix tests) and `tests/test_policy_vs_casbin_oracle.py`
(raw layer vs casbin on seeded random bundles and contexts).

## Request
A decision is over three things, the same triple casbin uses:

| casbin | here |
|---|---|
| subject | a context key named by the rule (`role_in.key`, e.g. `role`) |
| action | the `action` argument of `decide(action, context)` |
| resource | `context["resource"]` (a string) |

## Rules
Eight rule kinds, each a pure function of the context: `min_metric` (`>=`), `max_metric` (`<=`),
`flag_true`, `flag_false`, `status_equals`, `budget_ok` (`<=`), `role_in`, `no_nan`.
Numbers must be finite `int`/`float` (not bool, str, None, NaN, inf).

## Scope (casbin `keyMatch`)
Any rule may carry `action` and/or `resource` in `params`. Each defaults to `"*"`.
`"*"` matches anything; `"prefix*"` matches any string starting with `prefix`; anything else
must match exactly. A rule is **applicable** when both scopes match. An inapplicable rule is
reported (`applicable: false`), passes, and never blocks. A rule with a non-`"*"` resource scope
and no string `resource` in the context **fails closed** (the rule fails).

## Effect
Deny-override with default deny (casbin `!some(where (p.eft == deny))`):
- every applicable `block` rule must pass, otherwise the decision is deny;
- `warn` rules are reported and never block;
- an empty bundle denies unless `allow_empty=True`;
- unknown kind, missing key, invalid number, or any exception in a rule => that rule fails.
There is no separate explicit-`deny` rule kind: a failed block rule is the deny.

## Store
Append-only versions, exactly one active. Activating or rolling back to a looser bundle needs
`human_approved_by`. `is_stricter_or_equal(new, old)` is true when, for every old rule id, the
new rule exists (or the old was `warn`), has the same kind, is not downgraded `block` -> `warn`,
has a higher-or-equal `min` / lower-or-equal `max`, equal params for the other kinds, and a
scope that **covers** the old scope (widening or equal is fine; narrowing is loosening).
Decision log is bounded at 10,000; replay never writes to the audit chain.
With a real database, every publish and activation is written to the ops database before the
in-memory state changes (a failed write changes nothing), and versions, notes, the version
counter and the active version are reloaded on start; a stored row that does not parse or
validate stops startup with an integrity error. The decision log and the rollback target
are not stored, so a rollback straight after a restart has nothing to roll back to. With
`db_path=":memory:"` the store stays in memory.

## Enforcement (model transitions)
On by default (`policy_enforce_transitions`; `MLOPS_POLICY_ENFORCE_TRANSITIONS=false` turns it off).
`build_default_wiring` gives `ModelRegistry` the active policy as its decision point
(`PolicyStoreGate`), so a transition to `production` and a rollback are evaluated under action
`stage.production`; a denial is HTTP 403 `policy_denied`, the stage does not change, and the
decision is audited. No active policy => denied (fail closed): a fresh deployment must publish and
activate a policy before anything can be promoted. Transitions to `staging`/`archived` are not gated.

The server sets `role`, `actor`, `resource` (the model id) and `to_stage` from the authenticated
request and they override any client-sent `context` key of the same name. Other keys
(`accuracy`, ...) come from the client `context` and are **self-attested**: the gate does not
verify them. The rollback route accepts an optional `context` for the same reason; a metric rule
scoped to `stage.production` therefore also applies to rollback unless the caller supplies the
metric.
