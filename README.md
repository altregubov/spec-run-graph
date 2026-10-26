# SpecRunGraph

SpecRunGraph turns API specifications (OpenAPI + Markdown PRDs) into executed API tests:

```
srg analyze  ->  srg plan  ->  srg build  ->  srg test
 joint LLM        test cases    LLM writes      bru run; results -> Gel,
 analysis         -> Gel        Bruno files     logs -> MinIO
```

Every LLM step (analysis, test design, Bruno implementation, chatbot) uses the one model configured in `.env`.
See [architecture.md](architecture.md) for the design.

## Setup

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Node + Bruno CLI ≥ 3 (`npm i -g @usebruno/cli`).

```bash
docker compose up -d          # Gel :5656, MinIO :9000 (console :9001)
cp .env.example .env          # set OPENCODE_API_KEY and SRG_MODEL
uv sync
uv run srg doctor             # checks .env, Gel (applies the schema), MinIO, bru, LLM
```

## Usage (with the example API)

```bash
uv run uvicorn examples.orders.app:app --port 8000      # demo SUT with seeded defects

uv run srg analyze examples/orders/                     # report only: reports/analysis/<id>.{json,md}
uv run srg plan --from reports/analysis/<id>.json       # store test cases in Gel (or: srg plan examples/orders/)
uv run srg cases                                        # review; srg show TC-ORD-001 [--approve|--deprecate]
uv run srg build                                        # LLM implements cases -> test_generated/bruno/
uv run srg test                                         # run with bru; exit code 1 on failures
uv run srg web                                          # dashboard + chatbot on http://127.0.0.1:8090
```

- `srg go examples/orders/` runs all four stages.
- To narrow a command, use `--id 'TC-ORD-*'` or `--tags smoke` on `build` and `test`.
- `srg plan` run again only adds cases for coverage that is still missing.
- If a spec changes (try `examples/orders_v2/`), the cases it affects become `NEEDS_REVIEW`.
- You can run the generated collection with Bruno directly:
  `cd test_generated/bruno && CUSTOMER_TOKEN=customer-token bru run orders --env local`.
## Cleaning Gel and S3

`srg reset` deletes all data in Gel and all objects in the MinIO bucket in one command:

```bash
uv run srg reset
```

It first shows what will be deleted (object counts per Gel type, plus the number of objects and total size in `s3://spec-run-telemetry`), then asks for confirmation.

| Flag | Effect |
|---|---|
| `--yes` / `-y` | Skip the confirmation prompt (for scripts) |
| `--files` | Also delete `test_generated/` and `reports/` |
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
