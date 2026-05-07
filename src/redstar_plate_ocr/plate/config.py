"""Plate configuration models and parsing."""

from __future__ import annotations

import functools

import yaml
from pydantic import BaseModel, field_validator, model_validator

PLATE_TYPES: list[str] = ["standard", "square"]


class ValidChars(BaseModel):
    """Valid characters for a region's plates."""

    letters: str
    digits: str


class RegionConfig(BaseModel):
    """Configuration for a single region."""

    pattern: list[str]
    valid_chars: ValidChars
    forbidden_combos: list[str] = []
    enabled: bool = True

    @field_validator("pattern", mode="before")
    @classmethod
    def _coerce_pattern(cls, v: str | list[str]) -> list[str]:
        """YAML backward compat: string -> [string]."""
        if isinstance(v, str):
            return [v]
        return v

    @model_validator(mode="after")
    def _validate_region(self) -> RegionConfig:
        self._validate_pattern(self.pattern)
        self._validate_valid_chars(self.valid_chars)
        self._validate_alphabet_unique(self.raw_alphabet())
        return self

    @staticmethod
    def _validate_pattern(pattern: list[str]) -> None:
        if not pattern:
            raise ValueError("pattern must not be empty")
        if not all(len(p) > 0 for p in pattern):
            raise ValueError("pattern must not contain empty strings")

    @staticmethod
    def _validate_valid_chars(vc: ValidChars) -> None:
        if not vc.letters:
            raise ValueError("valid_chars.letters must not be empty")
        if not vc.digits:
            raise ValueError("valid_chars.digits must not be empty")

    @staticmethod
    def _validate_alphabet_unique(alphabet: str) -> None:
        if len(alphabet) != len(set(alphabet)):
            dupes = {c for c in alphabet if alphabet.count(c) > 1}
            raise ValueError(f"Duplicate chars in alphabet: {dupes}")

    def raw_alphabet(self) -> str:
        """Letters + digits for this region."""
        return self.valid_chars.letters + self.valid_chars.digits

    def get_patterns(self) -> list[str]:
        """Return copy of all patterns for this region.

        All patterns are returned regardless of plate_type.
        This is intentional: the model may misclassify plate_type,
        so validate_multi tries all patterns.
        """
        return list(self.pattern)

    def get_alphabet(self) -> str:
        """Return alphabet (letters + digits)."""
        return self.raw_alphabet()


class PlateConfig(BaseModel):
    """Top-level plate configuration."""

    regions: dict[str, RegionConfig]

    @classmethod
    def from_yaml(cls, path: str) -> PlateConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(regions=data["regions"])

    def get_alphabet(self, country: str) -> str:
        """Return alphabet string for country."""
        if country not in self.regions:
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return self.regions[country].get_alphabet()

    @property
    def num_countries(self) -> int:
        """Number of enabled countries."""
        return len(self.country_list)

    @property
    def country_list(self) -> list[str]:
        """List of enabled countries sorted by ISO code (alphabetical)."""
        return sorted(
            code
            for code, region in self.regions.items()
            if region.enabled
        )

    @functools.cached_property
    def union_alphabet(self) -> str:
        """Union of all enabled country alphabets (sorted, unique).

        Space is **not** included — it is not used as a CTC class in the
        model.  The internal model class layout is
        ``[...sorted_chars..., blank]`` where blank occupies the last
        index.
        """
        chars: set[str] = set()
        for code in self.country_list:
            region = self.regions[code]
            chars.update(region.get_alphabet())
        chars.discard(" ")
        return "".join(sorted(chars))

    @property
    def union_alphabet_size(self) -> int:
        """Union alphabet size including blank."""
        return len(self.union_alphabet) + 1

    def get_allowed_indices(
        self,
        country: str,
    ) -> list[int]:
        """Get allowed character indices for a country in
        union_alphabet."""
        union = self.union_alphabet
        if country not in self.regions:
            return list(range(len(union) + 1))
        region = self.regions[country]
        alphabet = region.get_alphabet()
        indices = [union.index(c) for c in alphabet if c in union]
        indices.append(len(union))  # blank
        return sorted(indices)

    def to_yaml_string(self) -> str:
        """Serialize config to YAML string."""
        data = self.model_dump()
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml_string(cls, yaml_str: str) -> PlateConfig:
        """Deserialize config from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls(**data)
