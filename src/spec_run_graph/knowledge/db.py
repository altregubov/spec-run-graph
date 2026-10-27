"""Gel client, schema migration, seeding and wipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gel

from spec_run_graph.config import Settings, get_settings

SCHEMA_FILE = Path(__file__).resolve().parents[3] / "dbschema" / "default.gel"

# Delete order respects links (children first).
WIPE_ORDER = [
    "Lesson",
    "Observation",
    "SetupRecipe",
    "HealAttempt",
    "DiagnosticNote",
    "TestExecution",
    "TestRun",
    "TestStep",
    "TestCase",
    "Actor",
    "Fixture",
    "TestModule",
    "Analysis",
    "Requirement",
    "ApiEndpoint",
    "FeatureSpec",
    "TestEngine",
]

# Knowledge about the API itself (not about one run); `srg reset` keeps it unless asked to forget.
KNOWLEDGE = ("Lesson",)

BRUNO_ENGINE = {
    "name": "bruno",
    "file_extension": ".yml",
    "default_command": "bru run {paths} --env {env} --reporter-json {json} --reporter-junit {junit}",
}


def create_client(settings: Settings | None = None) -> gel.AsyncIOClient:
    settings = settings or get_settings()
    client = gel.create_async_client(dsn=settings.gel_dsn, tls_security=settings.gel_tls_security)
    # stages update many cases concurrently; Gel's default 3 attempts are too few for serialization conflicts
    return client.with_retry_options(gel.RetryOptions(attempts=10))


def to_json(value: Any) -> str | None:
    """Serialize a value for a <json> query argument (None stays an empty set)."""
    return None if value is None else json.dumps(value, default=str)


def from_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


async def migrate(client: gel.AsyncIOClient, schema_file: Path = SCHEMA_FILE) -> None:
    """Apply dbschema/default.gel directly (no gel CLI needed). A no-op when unchanged."""
    schema = schema_file.read_text()
    await client.execute(f"start migration to {{ {schema} }}; populate migration; commit migration;")


async def reset_schema(client: gel.AsyncIOClient) -> None:
    await client.execute("reset schema to initial;")


async def seed(client: gel.AsyncIOClient, bru_version: str | None = None) -> None:
    await client.query(
        """
        insert TestEngine {
            name := <str>$name,
            file_extension := <str>$file_extension,
            default_command := <str>$default_command,
            version := <optional str>$version,
        } unless conflict on .name else (
            update TestEngine set { version := <optional str>$version ?? .version }
        )
        """,
        version=bru_version,
        **BRUNO_ENGINE,
    )


async def counts(client: gel.AsyncIOClient) -> dict[str, int]:
    return {t: await client.query_required_single(f"select count({t})") for t in WIPE_ORDER}


async def wipe_all(client: gel.AsyncIOClient, keep: tuple[str, ...] = ()) -> dict[str, int]:
    """Delete every object (except the `keep` types) in one transaction (schema is kept), then re-seed.
    Returns deleted counts."""
    deleted: dict[str, int] = {}
    async for tx in client.transaction():
        async with tx:
            for type_name in (t for t in WIPE_ORDER if t not in keep):
                deleted[type_name] = await tx.query_required_single(f"select count((delete {type_name}))")
    await seed(client)
    return deleted


async def readonly_query(client: gel.AsyncIOClient, query: str, **args: Any) -> str:
    """Run a query in a read-only transaction and return JSON (used by the chatbot fallback tool)."""
    ro = client.with_transaction_options(gel.TransactionOptions(readonly=True))
    result = "[]"
    async for tx in ro.transaction():
        async with tx:
            result = await tx.query_json(query, **args)
    return result
