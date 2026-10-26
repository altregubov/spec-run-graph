from __future__ import annotations

from dataclasses import dataclass, field

import gel
from pydantic_ai.models import Model
from rich.console import Console

from spec_run_graph.config import Settings, get_settings
from spec_run_graph.knowledge.db import create_client
from spec_run_graph.telemetry.s3 import TelemetryStore


@dataclass
class GraphDependencies:
    """Singletons shared by every pipeline stage / graph node."""

    settings: Settings = field(default_factory=get_settings)
    console: Console = field(default_factory=Console)
    model: Model | None = None  # None -> llm.build_model() (the model from .env); tests inject TestModel
    _client: gel.AsyncIOClient | None = None
    _store: TelemetryStore | None = None

    @property
    def client(self) -> gel.AsyncIOClient:
        if self._client is None:
            self._client = create_client(self.settings)
        return self._client

    @property
    def store(self) -> TelemetryStore:
        if self._store is None:
            self._store = TelemetryStore(self.settings)
        return self._store

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
