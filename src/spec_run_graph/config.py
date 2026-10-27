from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(".env")


class Settings(BaseSettings):
    """All runtime configuration, read from the environment and `.env`."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # LLM (OpenCode Go, OpenAI-compatible chat completions) - used by every agent
    opencode_api_key: str = ""
    opencode_base_url: str = "https://opencode.ai/inference/go/openai/v1"
    srg_model: str = "glm-5.3"
    srg_output_mode: Literal["tool", "native", "prompted"] = "tool"
    srg_llm_concurrency: int = 8
    srg_llm_retries: int = 3
    srg_plan_chunk_size: int = 3  # max endpoints per planner call (keeps LLM answers small)
    srg_recipes: bool = True  # design + verify setup recipes before planning (SRG_RECIPES=false skips it)
    srg_observe: bool = True  # read-only GET samples of the SUT as planner context (SRG_OBSERVE=false skips it)
    # auto: reasoning on for analyze/plan, off for the Bruno synthesizer (mechanical, many calls)
    srg_thinking: Literal["auto", "on", "off"] = "auto"
    srg_max_tokens: int | None = None  # output token limit per call (None = provider default)

    # Gel
    gel_dsn: str = "gel://admin@localhost:5656/main"
    gel_tls_security: Literal["strict", "no_host_verification", "insecure", "default"] = "insecure"

    # MinIO / S3
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "spec-run-telemetry"
    s3_region: str = "us-east-1"

    # System under test
    sut_base_url: str = "http://localhost:8000"
    sut_health_path: str = "/"

    # Paths and tools
    test_generated_dir: Path = Path("./test_generated")
    reports_dir: Path = Path("./reports")
    bru_bin: str = "bru"
    bru_timeout_sec: int = 900

    def engine_dir(self, engine: str = "bruno") -> Path:
        return self.test_generated_dir / engine


@lru_cache
def get_settings() -> Settings:
    return Settings()


def dotenv_values(path: Path = ENV_FILE) -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE, '#' comments) used to pass secrets to the bru subprocess."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip("'\"")
        values[key.strip().removeprefix("export ").strip()] = value
    return values
