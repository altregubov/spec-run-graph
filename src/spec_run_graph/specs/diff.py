"""Change detection between two versions of the same spec (by name)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpecChanges(BaseModel):
    spec_name: str
    from_version: str
    to_version: str
    added: list[str] = Field(default_factory=list)  # endpoint keys or requirement ids
    removed: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    @property
    def affected(self) -> set[str]:
        """Items whose existing test cases need review (added items are gaps, not impact)."""
        return set(self.removed) | set(self.changed)


def diff_hashes(
    spec_name: str, from_version: str, to_version: str, old: dict[str, str], new: dict[str, str]
) -> SpecChanges:
    """Compare {key: content_hash} maps of two spec versions."""
    return SpecChanges(
        spec_name=spec_name,
        from_version=from_version,
        to_version=to_version,
        added=sorted(set(new) - set(old)),
        removed=sorted(set(old) - set(new)),
        changed=sorted(k for k in set(old) & set(new) if old[k] != new[k]),
    )
