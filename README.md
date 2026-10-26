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
- `srg reset` deletes **all** data in Gel and all objects in the bucket. It asks for confirmation first, and `--files` also removes `test_generated/` and `reports/`.

## Tests

```bash
uv run pytest                      # unit + integration (integration needs docker compose, bru)
uv run pytest -m "not integration"
```

Integration tests use an isolated Gel branch (`srg_test`) and a separate bucket, and fake LLMs (`tests/fake_llm.py`).
