"""Raw telemetry tier: run logs and reports in MinIO (S3 API)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError

from spec_run_graph.config import Settings, get_settings


def run_prefix(run_id: str) -> str:
    return f"runs/{run_id}/"


def case_prefix(run_id: str, test_id: str, param_index: int) -> str:
    return f"runs/{run_id}/{test_id}/{param_index}/"


@dataclass
class PurgeResult:
    deleted: int
    failed: list[str]


class TelemetryStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.bucket = self.settings.s3_bucket
        self._session = aioboto3.Session()

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def key_of(self, uri: str) -> str:
        return uri.removeprefix(f"s3://{self.bucket}/")

    @asynccontextmanager
    async def _client(self):
        s = self.settings
        async with self._session.client(
            "s3",
            endpoint_url=s.s3_endpoint_url,
            aws_access_key_id=s.s3_access_key,
            aws_secret_access_key=s.s3_secret_key,
            region_name=s.s3_region,
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 3}),
        ) as client:
            yield client

    async def ensure_bucket(self) -> None:
        async with self._client() as c:
            try:
                await c.head_bucket(Bucket=self.bucket)
            except ClientError:
                await c.create_bucket(Bucket=self.bucket)

    async def put_text(self, key: str, text: str, content_type: str = "text/plain; charset=utf-8") -> str:
        return await self.put_bytes(key, text.encode(), content_type)

    async def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        async with self._client() as c:
            await c.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return self.uri(key)

    async def get_text(self, key: str) -> str:
        async with self._client() as c:
            obj = await c.get_object(Bucket=self.bucket, Key=key)
            async with obj["Body"] as stream:
                return (await stream.read()).decode(errors="replace")

    async def list_keys(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        async with self._client() as c:
            paginator = c.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                keys.extend(o["Key"] for o in page.get("Contents", []))
        return keys

    async def stats(self) -> tuple[int, int]:
        """(object count, total bytes) for the whole bucket."""
        count = size = 0
        async with self._client() as c:
            paginator = c.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self.bucket):
                for o in page.get("Contents", []):
                    count += 1
                    size += o["Size"]
        return count, size

    async def purge_bucket(self) -> PurgeResult:
        """Delete every object, keep the bucket. Safe to re-run."""
        keys = await self.list_keys()
        deleted, failed = 0, []
        async with self._client() as c:
            for i in range(0, len(keys), 1000):
                batch = [{"Key": k} for k in keys[i : i + 1000]]
                resp = await c.delete_objects(Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True})
                errors = resp.get("Errors", [])
                failed.extend(e["Key"] for e in errors)
                deleted += len(batch) - len(errors)
        return PurgeResult(deleted=deleted, failed=failed)
