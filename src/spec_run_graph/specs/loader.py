from __future__ import annotations

import contextlib
from pathlib import Path

import yaml

from spec_run_graph.specs.markdown_prd import load_prd, prd_digest
from spec_run_graph.specs.models import LoadedSpec
from spec_run_graph.specs.openapi import load_openapi, spec_digest

OPENAPI_EXT = {".yaml", ".yml", ".json"}


def _is_openapi(path: Path) -> bool:
    try:
        text = path.read_text()
        if path.suffix == ".json":
            return '"openapi"' in text[:4000]
        doc = yaml.safe_load(text)
        return isinstance(doc, dict) and "openapi" in doc
    except (yaml.YAMLError, UnicodeDecodeError):
        return False


def discover(paths: list[Path]) -> list[Path]:
    """Expand folders into spec files (OpenAPI yaml/json + markdown PRDs), sorted and de-duplicated."""
    found: list[Path] = []
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and (f.suffix == ".md" or (f.suffix in OPENAPI_EXT and _is_openapi(f))):
                    found.append(f)
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(p)
    return sorted(dict.fromkeys(f.resolve() for f in found))


def load_spec(path: Path) -> LoadedSpec:
    if path.suffix == ".md":
        return load_prd(path)
    if path.suffix in OPENAPI_EXT:
        return load_openapi(path)
    raise ValueError(f"unsupported spec file: {path}")


def load_specs(paths: list[Path]) -> list[LoadedSpec]:
    files = discover(paths)
    if not files:
        raise ValueError("no spec files found")
    specs = [load_spec(f) for f in files]
    cwd = Path.cwd()
    for s in specs:
        with contextlib.suppress(ValueError):
            s.source_path = str(Path(s.source_path).relative_to(cwd))
    return specs


def digest(spec: LoadedSpec) -> str:
    return spec_digest(spec) if spec.spec_type == "OPENAPI" else prd_digest(spec)


def all_endpoints(specs: list[LoadedSpec]):
    """Endpoint index across all OpenAPI specs: key -> (spec, endpoint)."""
    return {e.key: (s, e) for s in specs for e in s.endpoints}
