"""Tests for checkpoint migration (model + optimizer state)."""

from __future__ import annotations

import torch
import torch.nn as nn

from redstar_plate_ocr.pipeline.checkpoint import (
    migrate_checkpoint,
    migrate_optimizer_state,
)
from redstar_plate_ocr.plate.config import (
    PlateConfig,
    RegionConfig,
    ValidChars,
)


def _make_config(num_countries: int = 3) -> PlateConfig:
    codes = ["RU", "KZ", "BY", "UA", "UZ", "KG", "GE"]
    regions: dict[str, RegionConfig] = {}
    for code in codes[:num_countries]:
        regions[code] = RegionConfig(
            pattern=["X000XX00o"],
            valid_chars=ValidChars(letters="AB", digits="01"),
        )
    return PlateConfig(regions=regions)


def _make_country_head(
    num_c: int,
    in_channels: int = 64,
    hidden_size: int = 256,
) -> nn.Module:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.country_head = nn.Module()
            self.country_head.gap = nn.AdaptiveAvgPool2d(1)
            self.country_head.drop = nn.Dropout(0.1)
            self.country_head.fc = nn.Sequential(
                nn.Linear(in_channels, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_c),
            )
            self.text_head = nn.Linear(in_channels, 10)

    return _Model()


def test_migrate_checkpoint_trims_model_weights() -> None:
    pc = _make_config(num_countries=7)
    model = _make_country_head(num_c=8)
    sd = model.state_dict()
    warnings_list = migrate_checkpoint(sd, pc)
    assert sd["country_head.fc.2.weight"].shape[0] == 7
    assert sd["country_head.fc.2.bias"].shape[0] == 7
    assert any("sentinel" in w for w in warnings_list)


def test_migrate_checkpoint_noop_when_correct() -> None:
    pc = _make_config(num_countries=3)
    model = _make_country_head(num_c=3)
    sd = model.state_dict()
    warnings_list = migrate_checkpoint(sd, pc)
    assert warnings_list == []


def test_migrate_optimizer_state_trims_adam() -> None:
    pc = _make_config(num_countries=7)
    new_model = _make_country_head(num_c=7)
    new_model.eval()
    opt = torch.optim.Adam(new_model.parameters(), lr=0.001)
    fake_loss = torch.stack([p.sum() for p in new_model.parameters()]).sum()
    fake_loss.backward()
    opt.step()
    opt_sd = opt.state_dict()
    for idx, (name, param) in enumerate(new_model.named_parameters()):
        if not name.startswith("country_head.fc."):
            continue
        entry = opt_sd["state"].get(idx)
        if entry is None:
            continue
        for skey in ("exp_avg", "exp_avg_sq"):
            t = entry.get(skey)
            if t is not None and t.shape[0] == 7:
                pad_shape = (1,) + t.shape[1:]
                padded = torch.cat([t, torch.randn(*pad_shape)], dim=0)
                entry[skey] = padded
    opt_warnings = migrate_optimizer_state(opt_sd, new_model, pc)
    assert len(opt_warnings) > 0
    for idx, (name, param) in enumerate(new_model.named_parameters()):
        if not name.startswith("country_head.fc."):
            continue
        entry = opt_sd["state"].get(idx)
        if entry is None:
            continue
        for skey in ("exp_avg", "exp_avg_sq"):
            t = entry.get(skey)
            if t is not None and t.dim() >= 1:
                assert t.shape[0] == param.shape[0], (
                    f"{name}.{skey} shape mismatch: "
                    f"{t.shape[0]} != {param.shape[0]}"
                )


def test_migrate_optimizer_state_noop_when_correct() -> None:
    pc = _make_config(num_countries=3)
    model = _make_country_head(num_c=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    fake_loss = torch.stack([p.sum() for p in model.parameters()]).sum()
    fake_loss.backward()
    opt.step()
    opt_sd = opt.state_dict()
    opt_warnings = migrate_optimizer_state(opt_sd, model, pc)
    assert opt_warnings == []


def test_country_list_sorted_alphabetically() -> None:
    """country_list is sorted by ISO code (alphabetical)."""
    cfg = PlateConfig(
        regions={
            "RU": RegionConfig(
                pattern=["X000XX00o"],
                valid_chars=ValidChars(letters="AB", digits="01"),
            ),
            "KZ": RegionConfig(
                pattern=["X000XX00o"],
                valid_chars=ValidChars(letters="AB", digits="01"),
            ),
        }
    )
    assert cfg.country_list == ["KZ", "RU"]
