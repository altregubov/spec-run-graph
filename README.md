# SpecRunGraph

SpecRunGraph turns API specifications (OpenAPI + Markdown PRDs) into executed API tests:

```
srg analyze  ->  srg plan  ->  srg build  ->  srg test  ->  diagnose  ->  heal
 joint LLM        recipes +     compiler        bru run;     why did it    repair TEST_DEFECT
 analysis         test cases    writes Bruno    results ->   fail? (API,   cases, rebuild,
                  -> Gel        files           Gel/MinIO    spec, test)   verify or roll back -> lessons
```

Every LLM step (analysis, recipes, test design, healing, lessons, chatbot) uses the one model configured in `.env`.
Bruno files are written by a deterministic compiler; the LLM only implements cases the compiler cannot translate.
See [architecture.md](architecture.md) for the design.

## Setup

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Node + Bruno CLI ≥ 3 (`npm i -g @usebruno/cli`).

```bash
docker compose up -d          # Gel :5656, MinIO :9000 (console :9001)
cp .env.example .env          # set OPENCODE_API_KEY and SRG_MODEL
uv sync
uv run srg doctor             # checks .env, Gel (applies the schema), MinIO, bru, LLM
```

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OPENCODE_API_KEY`, `OPENCODE_BASE_URL` | – | OpenCode Go endpoint (OpenAI-compatible) used by **every** LLM step |
| `SRG_MODEL` | `glm-5.3` | the one model for analyze, plan, build, diagnose, heal and the chatbot |
| `SRG_LLM_CONCURRENCY` | `8` | parallel LLM calls (each in its own OpenCode session) |
| `SRG_THINKING` | `auto` | reasoning on for analyze/plan/diagnose, off for build; or `on` / `off` |
| `SRG_OUTPUT_MODE` | `tool` | `tool`, `native` or `prompted`; falls back to `prompted` automatically if the model can't do tool output |
| `SRG_PLAN_CHUNK_SIZE` | `3` | endpoints per planner call (smaller = shorter, more reliable answers) |
| `SRG_LLM_RETRIES`, `SRG_MAX_TOKENS` | `3`, provider default | validation retries per call; output token limit |
| `SUT_BASE_URL`, `SUT_HEALTH_PATH` | `http://localhost:8000`, `/` | the API under test and its pre-flight health check |
| `GEL_DSN`, `S3_*` | docker compose defaults | Gel and MinIO |
| any `UPPER_CASE` secret | – | referenced by tests as `{{ $env.NAME }}` (e.g. `ADMIN_TOKEN`), passed to bru as process env, never written to files |

Each command ends with an LLM usage line (calls, tokens, call time vs wall time), so you can see where time goes.

## Commands

| Command | What it does |
|---|---|
| `srg doctor` | checks `.env`, Gel (applies the schema), MinIO, bru and the LLM |
| `srg analyze <specs>` | joint analysis → `reports/analysis/<id>.{json,md}` (no Gel writes) |
| `srg plan <specs>` / `--from <analysis.json>` | test cases → Gel only (adds only missing coverage; `--replace` starts over) |
| `srg cases`, `srg show <id> [--approve\|--deprecate]` | review the cases in Gel |
| `srg build [--id] [--tags] [--force]` | compiles the Bruno files → `test_generated/bruno/` (LLM fallback) |
| `srg test [--heal] [--no-diagnose]` | run with bru, then analyse failures (and heal test defects) |
| `srg diagnose [RUN_ID]` | analyse the failures of a run |
| `srg heal [RUN_ID]` | repair `TEST_DEFECT` cases, rebuild, verify or roll back |
| `srg go <specs>` | analyze → plan → build → test → diagnose |
| `srg recipes`, `srg lessons` | setup recipes offered to the planner; lessons learned from verified heals |
| `srg web` | dashboard + QA chatbot on http://127.0.0.1:8090 |
| `srg reset` | wipe Gel data and the S3 bucket |

## Providing specs

Pass spec files or folders; everything given to one command is analysed jointly:

```bash
uv run srg analyze specs/                              # a folder (scanned recursively)
uv run srg analyze api/swagger.yaml docs/prd.md        # individual files
```

- **API contracts:** OpenAPI 3.x (`openapi:`) or Swagger 2.0 (`swagger: "2.0"`) as `.yaml`, `.yml` or `.json`. Only local `$ref`s (`#/...`) are resolved, so bundle multi-file specs first. For Swagger 2.0 the `basePath` becomes part of each endpoint path.
- **PRDs:** every `.md` file. Put requirement ids in headings (`## PRD-ORD-42 Create draft order`) and describe test data.
- **Credentials come from the spec.** When the documents describe a login endpoint and test accounts (e.g. `POST /api/admin/login` with `admin` / `admin123`), the analyzer creates *login actors*. The generated collection logs each actor in once per run and uses the returned token (`{{auth_admin}}`), so no tokens go in `.env`. Only when no accounts are documented are actors given `{{ $env.X }}` secrets instead.
- Other YAML/JSON files in a folder are skipped with a warning; keep unrelated Markdown out of the spec folder.
- Set `SUT_BASE_URL` in `.env` to the API under test.

## Usage (with the example API)

```bash
uv run uvicorn examples.orders.app:app --port 8000      # demo SUT with seeded defects

uv run srg analyze examples/orders/                     # report only: reports/analysis/<id>.{json,md}
uv run srg plan --from reports/analysis/<id>.json       # store test cases in Gel (or: srg plan examples/orders/)
uv run srg cases                                        # review; srg show TC-ORD-001 [--approve|--deprecate]
uv run srg build                                        # LLM implements cases -> test_generated/bruno/
uv run srg test                                         # run with bru + analyse failures; exit code 1 on failures
uv run srg web                                          # dashboard + chatbot on http://127.0.0.1:8090
```

- `srg go examples/orders/` runs all four stages.
- To narrow a command, use `--id 'TC-ORD-*'` or `--tags smoke` on `build` and `test`.
- `srg plan` run again only adds cases for coverage that is still missing.
- If a spec changes (try `examples/orders_v2/`), the cases it affects become `NEEDS_REVIEW`.
- You can run the generated collection with Bruno directly: `cd test_generated/bruno && bru run orders --env local`.
- After a re-analysis that changed actors or fixtures, run `srg plan --from <analysis.json> --refresh-only` (updates Gel, no new cases) and then `srg build`.
## Failure analysis

`srg test` analyses every failed test automatically. To re-analyse a run: `srg diagnose [RUN_ID]`.

1. **Rules first, no LLM:** connection refused, DNS errors, timeouts and HTTP 502/503/504 become `ENV_FAILURE`.
2. **Grouping:** failures with the same signature (status, HTTP code, failed assertion, actor or endpoints) are investigated once. For example, 150 × HTTP 401 is usually one auth problem. The result is stored for every member of the group.
3. **Diagnostics agent (the `.env` model):** it investigates one representative per group with read-only tools: the real HTTP exchange (MinIO), the OpenAPI contract, the requirement, the generated Bruno file and the case history. It may also re-run the case once to check for flakiness. It returns:

| Category | Meaning |
|---|---|
| `APPLICATION_BUG` | the API is wrong although test and contract agree |
| `SPEC_MISMATCH` | the API deviates from the documented contract / requirements |
| `TEST_DEFECT` | the generated test is wrong; the case is set to `NEEDS_REVIEW` with the suggested fix |
| `ENV_FAILURE` | connectivity, credentials, unavailable dependencies |
| `FLAKY` | the failure does not reproduce on re-run |
| `UNKNOWN` | confidence below 0.5 |

Each diagnosis (category, root cause, evidence, recommended action, confidence) is stored in Gel as a `DiagnosticNote`. It is shown on the run page of `srg web`, and the chatbot can answer questions about it.

Options:
- `srg test --no-diagnose` skips the analysis, and `--no-rerun` forbids reruns.
- `srg diagnose --force` re-analyses failures that already have a diagnosis.
- `--max-llm N` (default 30) caps the investigations per run, and `--no-group` investigates every failure on its own.

## Self-healing tests

When a failure is diagnosed as `TEST_DEFECT` (the API follows its contract and the test is wrong), the healer can repair the test case:

```bash
uv run srg test --heal        # test -> diagnose -> heal -> verify, in one go
uv run srg heal [RUN_ID]      # heal the TEST_DEFECT failures of an already diagnosed run
```

1. **Repair.** The healer agent (the `.env` model) gets the case, its diagnosis, the recorded HTTP exchange, the OpenAPI contract and the requirements, and returns the corrected test case.
2. **Guard rails** (checked by code; problems are sent back to the model):
   - the healed case must stay valid;
   - it must keep the same case type, requirements and endpoints, so it still tests the same thing;
   - every expected HTTP status must be documented in the contract, so a test is never "healed" toward a bug;
   - the case must actually change.
3. **Verify.** The case is saved in Gel, its Bruno files are regenerated, and it runs again against the API in a `HEAL_VERIFY` run.
   - **Passes:** the case is set to `READY` and tagged `healed`.
   - **Still fails:** the healer gets one more attempt with the new failure. If that fails too, the case is **rolled back** to its previous version and set to `NEEDS_REVIEW` with the reason.

`APPLICATION_BUG` and `SPEC_MISMATCH` are never healed. Every attempt is stored as a `HealAttempt` (before/after IR, change summary, verification result) and shown by `srg show <id>` and on the case page of `srg web`.

Options: `--no-verify` applies heals without re-running them (status `UNVERIFIED`), and `--max-cases N` caps the number of cases (default 20).

## Fewer broken test cases

Most generated defects came from the LLM guessing contract details. Four mechanisms remove the guessing:

- **Contract facts.** At ingest every endpoint gets `required_body_fields` and `assertable_paths` (fields the
  contract guarantees per status), stored on `ApiEndpoint` in Gel. Prompts list them literally ("a successful
  request MUST send: balance, password, role, username"; "assertable (guaranteed) fields: ..."), the planner sees
  the full contracts of other modules' endpoints used for setup, and the IR validation rejects missing required
  request fields, optional response fields (`requestId`, `error.details`) and object keys on lists.
- **Setup recipes.** Once per analysis the LLM designs reusable setups (dedicated customer with a balance,
  dedicated product, ...). Each recipe is compiled and run once against the API; only `VERIFIED` recipes
  (`SetupRecipe` in Gel, `srg recipes`) are offered to the planner, which calls them with one step
  `{type: recipe, recipe: dedicated_customer, inputs: {balance: 5000}, alias: buyer}`. The step is expanded into
  ordinary steps before validation and storage (outputs become `{{ $extract.buyer_token }}`, cleanup is added).
- **Deterministic Bruno compiler.** Requests, auth, variables, extracts and checks translate one-to-one
  (`engines/bruno/compiler.py`); each check becomes a `test()` with a readable message
  (`expected == "PAID", got "DRAFT"`). `srg build` needs no LLM for these cases and takes seconds.
- **Lessons.** After verified heals the LLM generalises the fixes into short rules (`Lesson` in Gel,
  `srg lessons`), which are added to future planner and healer prompts.

## Cleaning Gel and S3

`srg reset` deletes all data in Gel and all objects in the MinIO bucket in one command:

```bash
uv run srg reset
```

It first shows what will be deleted (object counts per Gel type, plus the number of objects and total size in `s3://spec-run-telemetry`), then asks for confirmation.

| Flag | Effect |
|---|---|
| `--yes` / `-y` | Skip the confirmation prompt (for scripts) |
| `--files` | Also delete `reports/` (`test_generated/` is always deleted: it is generated from Gel) |
| `--schema` | Also reset the Gel schema and re-apply it (use this after incompatible schema changes) |
| `--force-remote` | Allow a reset when `GEL_DSN` or `S3_ENDPOINT_URL` isn't a local host; without it the command refuses |

A full clean start:

```bash
uv run srg reset --yes --files
```

What's kept:
- The Gel schema stays unless you pass `--schema`.
- The bucket itself stays; only its objects are deleted.
- The `TestEngine(bruno)` row is re-created automatically.

How it deletes:
- Gel is wiped in one transaction, so it either deletes everything or nothing.
- S3 objects are deleted after that. If any fail, they're listed, and you can safely run `srg reset` again.

To remove the Docker containers and their data volumes:

```bash
docker compose down -v
```

After that, `docker compose up -d` followed by `uv run srg doctor` recreates the schema and the bucket.

## Tests

```bash
uv run pytest                      # unit + integration (integration needs docker compose, bru)
uv run pytest -m "not integration"
```

Integration tests use an isolated Gel branch (`srg_test`) and a separate bucket, and fake LLMs (`tests/fake_llm.py`).
