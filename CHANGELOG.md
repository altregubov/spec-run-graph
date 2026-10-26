# Changelog

## v0.2.0 — 2026-10-26

### Added
- **Failure analysis:** `srg diagnose`, and automatically after `srg test`.
  - A rule pre-classifier handles environment failures without an LLM call.
  - A ReAct diagnostics agent investigates the rest, with tools for the HTTP exchange, the OpenAPI contract, the requirements, the test files, the case history and a one-time rerun.
  - Categories: `APPLICATION_BUG`, `SPEC_MISMATCH`, `TEST_DEFECT`, `ENV_FAILURE`, `FLAKY`, `UNKNOWN`. Each result is stored as a `DiagnosticNote`.
  - Failures with the same signature are grouped and investigated once; `--max-llm` caps the investigations.
- **Self-healing:** `srg heal`, or `srg test --heal`.
  - `TEST_DEFECT` cases are repaired by the LLM, within guard rails: same type, requirements and endpoints; documented status codes only; the case must actually change.
  - The repaired case is rebuilt and verified in a `HEAL_VERIFY` run, then kept (`READY`, tagged `healed`) or rolled back (`NEEDS_REVIEW`).
  - Every attempt is audited as a `HealAttempt`.
- **Swagger 2.0 support:** body/formData parameters, `#/definitions`, `basePath`, `host`/`schemes`. Non-API YAML/JSON files in a spec folder are reported as skipped.
- **Dashboard:** diagnoses on the run page and heal history on the case page; the chatbot knows about diagnoses.
- **LLM usage summary** after every command.

### Changed
- **Faster LLM pipeline** (about 6× faster `srg build`):
  - each conversation has its own OpenCode session;
  - thinking is off for the synthesizer (`SRG_THINKING`);
  - default concurrency is 8;
  - prompts use compact JSON.
- **More robust LLM calls:**
  - automatic, sticky prompted-output fallback;
  - chunked planning (`SRG_PLAN_CHUNK_SIZE`);
  - a failing planner chunk no longer aborts `srg plan`.
- **Bruno validation:** upper-case response header keys (`res.headers['X-Request-ID']`) are now rejected.
- **Runner:** assertion and test failures are read from bru's per-result details.

### Schema (applied automatically by `srg doctor`)
- New: `DiagnosticNote`, `HealAttempt`, `DiagnosticCategory`, `DiagnosticSource`, `HealStatus`.
- New field: `TestRun.kind` (`SUITE` / `HEAL_VERIFY`).

## v0.1.0 — 2026-10-25
- First release. Pipeline: `srg analyze` → `plan` → `build` → `test`.
  - joint OpenAPI + PRD analysis;
  - test cases stored in Gel;
  - Bruno OpenCollection files written by the LLM;
  - execution with results in Gel and logs in MinIO.
- Spec change impact.
- `srg reset`.
- Web dashboard with a QA chatbot.
