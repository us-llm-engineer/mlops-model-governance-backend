# Test Suite Index — MLOps Control Plane Round 1

**Chosen Plan:** All three plans (A, B, C) selected — each addresses one distinct requirement.
- Plan A (R1.1): Auth/Tenancy + Declarative Specs
- Plan B (R1.2): Orchestration + Registry & Lineage
- Plan C (R1.3): API/CLI (Developer Self-Service)

**Total Test Suites:** 12 (frozen)
**Total Test Cases:** 52+ (4+ per suite)

---

## R1.1 — Auth/Tenancy + Declarative Pipeline/Environment Specs

### R1.1-S1: Schema Validation (Success/Boundary)
**Claim:** C1 — Pipelines and environments are declarative YAML; references immutable artifacts.
**Dimension:** success/boundary
**Case Names:**
1. `test_valid_pipeline_all_immutable_refs` — Valid refs pass validation
2. `test_valid_pipeline_with_shas` — All SHA256 refs pass
3. `test_valid_pipeline_with_version_tags` — Semantic versioning accepted (not "latest")
4. `test_reject_latest_tag_in_data_version` — Reject mutable "latest" tag
5. `test_reject_wildcard_in_data_path` — Reject wildcard paths
6. `test_reject_invalid_config_hash_format` — Reject malformed SHA256
7. `test_reject_missing_required_field` — Reject incomplete specs
8. `test_reject_develop_branch_reference` — Reject "develop" branch ref

**Run Command:**
```bash
pytest tests/R1.1-S1.py -v
```

**Mutation Targets:**
- Remove tag mismatch check (mutation: accept "latest" tags)
- Skip path wildcard validation (mutation: allow wildcards)
- Remove hash format check (mutation: accept any config_hash)
- Skip required field validation (mutation: allow missing fields)

---

### R1.1-S2: RBAC/Quota (Authorization/Privacy)
**Claim:** C2 — RBAC decisions logged; quota violations recorded before rejection.
**Dimension:** authorization/privacy
**Case Names:**
1. `test_deployer_can_submit_pipeline` — Deployer authorized for submission
2. `test_approver_can_approve_pipeline` — Approver authorized for approval
3. `test_viewer_can_list_pipelines` — Viewer authorized for read
4. `test_viewer_cannot_submit_pipeline` — Viewer denied submission
5. `test_unknown_principal_denied` — Unknown principal denied
6. `test_approver_cannot_deploy` — Approver denied deploy action
7. `test_quota_exceeded_denied_with_log` — Quota violation logged + rejected
8. `test_quota_within_limit_allowed` — Within quota allowed
9. `test_no_quota_set_allows_submission` — No limit = allow
10. `test_audit_log_contains_reason` — Audit includes reason (current/limit)

**Run Command:**
```bash
pytest tests/R1.1-S2.py -v
```

**Mutation Targets:**
- Remove role authorization check (mutation: allow all roles)
- Skip quota enforcement (mutation: always allow submission)
- Remove audit logging (mutation: skip audit on denied action)
- Skip audit on quota exceeded (mutation: silent failure)

---

### R1.1-S3: Secret Reference Validation & Isolation (Persistence/Privacy)
**Claim:** C3 — Secrets referenced by name, never inlined; isolation via envelope-encryption.
**Dimension:** persistence/privacy
**Case Names:**
1. `test_secret_reference_stored_not_inlined` — Ref stored, not plaintext
2. `test_multiple_secret_refs_handled` — Multiple refs handled correctly
3. `test_secret_not_leaked_in_error_message` — Error messages mask secrets
4. `test_secret_value_not_in_logs` — Plaintext not in stored JSON
5. `test_envelope_encryption_roundtrip` — Encrypt/decrypt works
6. `test_envelope_hmac_verification_success` — Valid HMAC passes
7. `test_envelope_hmac_verification_failure` — Invalid HMAC rejected
8. `test_envelope_ciphertext_tampering_detected` — Tampering detected
9. `test_nonexistent_secret_returns_none` — Missing secret handled
10. `test_multiple_secrets_retrieved_independently` — Multiple secrets isolated

**Run Command:**
```bash
pytest tests/R1.1-S3.py -v
```

**Mutation Targets:**
- Skip secret name masking (mutation: expose secret names in errors)
- Remove envelope encryption (mutation: store plaintext)
- Skip HMAC validation (mutation: accept unsigned envelopes)
- Return plaintext in decryption (mutation: skip decryption)

---

### R1.1-S4: YAML Round-Trip & Lineage Consistency (Persistence/Concurrency)
**Claim:** C4 — Deployed config read back, re-submitted produces identical lineage & audit.
**Dimension:** persistence/concurrency
**Case Names:**
1. `test_config_roundtrip_all_fields_preserved` — All fields survive roundtrip
2. `test_optional_fields_preserved` — Optional fields retained
3. `test_hash_consistency_across_roundtrip` — Hash unchanged after roundtrip
4. `test_config_integrity_valid` — Valid config passes integrity check
5. `test_config_integrity_detects_tampering` — Tampering detected
6. `test_audit_trail_records_submission` — Submission logged
7. `test_audit_trail_records_reads` — Reads logged
8. `test_audit_trail_preserves_config_hash` — Audit records hash
9. `test_concurrent_reads_safe` — Multiple concurrent reads safe
10. `test_concurrent_writes_serialized` — Concurrent writes serialized (no corruption)

**Run Command:**
```bash
pytest tests/R1.1-S4.py -v
```

**Mutation Targets:**
- Skip field during serialization (mutation: lose field on write)
- Modify hash after read (mutation: change hash without validation)
- Skip audit entry on read (mutation: no audit trail)
- No lock protection on concurrent writes (mutation: race condition)

---

## R1.2 — Workflow Orchestration + Artifact Registry & Lineage

### R1.2-S1: State Machine Logic (Success/Boundary)
**Claim:** C5 — Pipeline run state machine: Pending → Running → Completed/Failed; retries increment.
**Dimension:** success/boundary
**Case Names:**
1. `test_create_run_starts_in_pending` — New run in PENDING state
2. `test_pending_to_running_transition` — PENDING → RUNNING valid
3. `test_running_to_completed_on_job_success` — RUNNING → COMPLETED on success
4. `test_running_to_failed_on_job_failure` — RUNNING → FAILED on failure
5. `test_cannot_submit_job_from_completed_state` — COMPLETED → RUNNING invalid
6. `test_cannot_complete_non_running_run` — Cannot complete non-RUNNING
7. `test_cannot_retry_completed_run` — Cannot retry COMPLETED (only FAILED)
8. `test_cannot_archive_from_running` — Cannot archive RUNNING (must complete first)
9. `test_retry_increments_attempt_number` — Retry increments counter
10. `test_multiple_retries_track_attempts` — Multiple retries tracked correctly

**Run Command:**
```bash
pytest tests/R1.2-S1.py -v
```

**Mutation Targets:**
- Allow invalid state transition (mutation: skip transition guard)
- Skip state transition (mutation: keep current state)
- Reset retry counter instead of increment (mutation: counter goes to 1)
- Allow transition to invalid state (mutation: remove validation)

---

### R1.2-S2: Artifact Integrity (Persistence/Tampering)
**Claim:** C6 — Pipeline stages output immutable artifacts under content-addressed tree.
**Dimension:** persistence/tampering
**Case Names:**
1. `test_artifact_stored_under_correct_hash` — Artifact stored under SHA256 hash
2. `test_artifact_retrieved_with_correct_content` — Retrieved content matches original
3. `test_artifact_metadata_stored` — Metadata linked by hash
4. `test_cannot_overwrite_artifact_with_different_content` — Immutability enforced
5. `test_identical_content_returns_same_hash` — Content deduplication
6. `test_integrity_check_passes_for_untampered` — Integrity check passes
7. `test_integrity_check_fails_for_tampered_artifact` — Tampering detected
8. `test_missing_artifact_integrity_check_fails` — Missing artifact fails check
9. `test_retrieve_nonexistent_artifact_returns_none` — Missing returns None
10. `test_retrieve_multiple_artifacts_from_same_run` — Multiple artifacts independent

**Run Command:**
```bash
pytest tests/R1.2-S2.py -v
```

**Mutation Targets:**
- Store under wrong hash path (mutation: skip hash validation on store)
- Allow in-place modification (mutation: skip immutability check)
- Skip metadata serialization (mutation: no metadata)
- Skip hash verification (mutation: remove integrity check)

---

### R1.2-S3: Lineage DAG (Persistence/Graph-Consistency)
**Claim:** C7 — Lineage DAG immutable; query traces data→training→model→eval→promotion.
**Dimension:** persistence/graph-consistency
**Case Names:**
1. `test_create_data_node` — Create data node with immutable hash
2. `test_create_training_run_node` — Create training run node
3. `test_identical_content_produces_same_node_id` — Content deduplication
4. `test_add_edge_data_to_training` — Add edge data → training
5. `test_edge_requires_both_nodes_exist` — Nodes must exist for edge
6. `test_full_lineage_chain` — Full chain: data → training → model → eval → promo
7. `test_node_immutability_verification_passes` — Immutability check passes
8. `test_retroactive_modification_detected` — Modification detected
9. `test_lineage_traces_to_original_data` — Lineage always reaches data
10. `test_query_edges_to_node` — Query incoming edges

**Run Command:**
```bash
pytest tests/R1.2-S3.py -v
```

**Mutation Targets:**
- Skip edge insertion (mutation: edge not recorded)
- Use mutable reference (mutation: node not content-addressed)
- Skip hash validation (mutation: retroactive change not detected)
- Skip node (mutation: incomplete lineage path)

---

### R1.2-S4: Failure & Retry Semantics (Failure-Mapping/Recovery)
**Claim:** C8 — K8s failure → Failed; retries up to 3; each retry new artifacts (old preserved).
**Dimension:** failure-mapping/recovery
**Case Names:**
1. `test_timeout_is_retryable` — Timeout retryable
2. `test_oom_is_retryable` — OOM retryable
3. `test_pod_eviction_is_retryable` — Pod eviction retryable
4. `test_code_error_not_retryable` — Code error terminal
5. `test_invalid_config_not_retryable` — Invalid config terminal
6. `test_missing_data_not_retryable` — Missing data terminal
7. `test_retry_succeeds_on_attempt_1` — Retry allowed on attempt 1
8. `test_retry_succeeds_on_attempt_2` — Retry allowed on attempt 2
9. `test_no_retry_on_max_attempts` — No retry at max (terminal)
10. `test_artifacts_preserved_from_attempt_1` — Old artifacts preserved on retry
11. `test_multiple_failed_attempts_artifacts_distinct` — Each attempt artifacts distinct
12. `test_all_attempts_failed_flag` — Can query all-failed state
13. `test_retry_reason_message` — Human-readable retry reason

**Run Command:**
```bash
pytest tests/R1.2-S4.py -v
```

**Mutation Targets:**
- Failed run not recognized (mutation: skip failure check)
- Retry limit not enforced (mutation: remove attempt counter check)
- Old artifacts overwritten (mutation: reuse same hash)
- Previous artifacts lost (mutation: delete old artifacts)

---

## R1.3 — Platform API & CLI (Developer Self-Service)

### R1.3-S1: API Contract (Success/Boundary)
**Claim:** C9 — Metadata endpoints respond < 500ms (p95); JSON; pagination supported.
**Dimension:** success/boundary
**Case Names:**
1. `test_list_pipelines_returns_json` — /pipelines returns JSON
2. `test_list_models_returns_json` — /models returns JSON
3. `test_list_environments_returns_json` — /environments returns JSON
4. `test_list_audit_returns_json` — /audit returns JSON
5. `test_pagination_limit_parameter` — limit parameter works
6. `test_pagination_offset_parameter` — offset parameter works
7. `test_pagination_default_limit` — Default limit applied
8. `test_pagination_total_count` — Response includes total
9. `test_single_request_within_sla` — Single request < 500ms
10. `test_p95_latency_within_sla` — p95 latency < 500ms

**Run Command:**
```bash
pytest tests/R1.3-S1.py -v
```

**Mutation Targets:**
- Response exceeds latency SLA (mutation: remove caching)
- Latency not measured (mutation: skip timing check)
- Pagination not implemented (mutation: return all results)
- Non-JSON response (mutation: return plaintext)

---

### R1.3-S2: Error Mapping (Invalid-Request/Error-Clarity)
**Claim:** C10 — Invalid requests return HTTP 4xx with error_code + message JSON; no stack traces.
**Dimension:** invalid-request/error-clarity
**Case Names:**
1. `test_missing_pipeline_name` — Missing name field error
2. `test_missing_data_version` — Missing data_version error
3. `test_missing_model_promotion_from_env` — Missing from_env error
4. `test_invalid_uuid_format` — Invalid UUID rejected
5. `test_invalid_pagination_limit_non_integer` — Non-integer limit rejected
6. `test_invalid_pagination_offset_negative` — Negative offset rejected
7. `test_error_response_is_json` — Error response is JSON
8. `test_error_response_has_error_code` — Response includes error_code
9. `test_error_response_has_message` — Response includes message
10. `test_error_response_includes_details` — Response includes details
11. `test_no_stack_trace_in_error_response` — No stack traces in response

**Run Command:**
```bash
pytest tests/R1.3-S2.py -v
```

**Mutation Targets:**
- Missing error_code (mutation: return only message)
- Stack trace in error response (mutation: expose traceback)
- Plaintext error (mutation: return non-JSON)
- Missing field name in message (mutation: generic error)

---

### R1.3-S3: CLI Integration (Operator-Journey)
**Claim:** C11 — CLI commands map to API calls; human-readable output; error → non-zero exit.
**Dimension:** operator-journey/integration
**Case Names:**
1. `test_pipeline_list_command` — mlops pipeline list succeeds
2. `test_model_list_command` — mlops model list succeeds
3. `test_model_promote_command` — mlops model promote succeeds
4. `test_pipeline_submit_command` — mlops pipeline submit succeeds
5. `test_json_flag_outputs_json` — --json flag produces JSON
6. `test_default_format_is_human_readable` — Default is human-readable
7. `test_model_list_json_format` — model list --json works
8. `test_cli_propagates_api_error` — CLI error → non-zero exit
9. `test_cli_outputs_error_to_stderr` — Error to stderr
10. `test_cli_promotion_error_fails_fast` — Error caught immediately

**Run Command:**
```bash
pytest tests/R1.3-S3.py -v
```

**Mutation Targets:**
- CLI success but API fails (mutation: skip error check)
- API error not propagated (mutation: return 0 on error)
- Output not human-readable (mutation: return raw JSON)
- Missing --json flag (mutation: remove flag support)

---

### R1.3-S4: Concurrent Load (Concurrency/Latency)
**Claim:** C12 — End-to-end latency < 2s; concurrent CLI commands don't block.
**Dimension:** concurrency/latency
**Case Names:**
1. `test_single_promotion_within_sla` — Single promotion < 2s
2. `test_multiple_sequential_commands_within_sla` — Sequential < 2s each
3. `test_mean_latency_reasonable` — Mean latency < 500ms
4. `test_concurrent_requests_succeed` — Concurrent requests succeed
5. `test_concurrent_requests_not_serialized` — Requests execute in parallel
6. `test_high_concurrency_sustained` — High concurrency sustained
7. `test_p95_latency_measured` — p95 latency measured
8. `test_p99_latency_measured` — p99 latency measured
9. `test_latency_percentiles_ordered` — p50 ≤ p95 ≤ p99
10. `test_all_concurrent_requests_complete` — All requests complete
11. `test_no_deadlock_under_load` — No deadlock

**Run Command:**
```bash
pytest tests/R1.3-S4.py -v
```

**Mutation Targets:**
- No load testing (mutation: single request only)
- Latency not measured (mutation: skip timing)
- Lock contention (mutation: use global lock)
- Requests queue/serialize (mutation: no parallelism)

---

## Master Test Run

**Run all 12 suites:**
```bash
pytest tests/R1.1-S*.py tests/R1.2-S*.py tests/R1.3-S*.py -v
```

**Expected:**
- 52+ test cases pass
- All tests use mocks (no K8s, databases, external APIs)
- Total execution time: < 10 seconds

---

## Summary by Requirement

| Requirement | Suite | Dimension | Cases | Run Cmd |
|---|---|---|---|---|
| R1.1 | S1 | success/boundary | 8 | pytest tests/R1.1-S1.py |
| R1.1 | S2 | authorization/privacy | 10 | pytest tests/R1.1-S2.py |
| R1.1 | S3 | persistence/privacy | 10 | pytest tests/R1.1-S3.py |
| R1.1 | S4 | persistence/concurrency | 10 | pytest tests/R1.1-S4.py |
| R1.2 | S1 | success/boundary | 10 | pytest tests/R1.2-S1.py |
| R1.2 | S2 | persistence/tampering | 10 | pytest tests/R1.2-S2.py |
| R1.2 | S3 | persistence/graph-consistency | 10 | pytest tests/R1.2-S3.py |
| R1.2 | S4 | failure-mapping/recovery | 13 | pytest tests/R1.2-S4.py |
| R1.3 | S1 | success/boundary | 10 | pytest tests/R1.3-S1.py |
| R1.3 | S2 | invalid-request/error-clarity | 11 | pytest tests/R1.3-S2.py |
| R1.3 | S3 | operator-journey/integration | 10 | pytest tests/R1.3-S3.py |
| R1.3 | S4 | concurrency/latency | 11 | pytest tests/R1.3-S4.py |

**Total: 12 suites, 123 distinct test cases**
