"""Tests for UnifiedCTCHead and build_mask_table."""

import torch

from redstar_plate_ocr.nn.heads import UnifiedCTCHead
from redstar_plate_ocr.nn.mask_table import build_mask_table
from redstar_plate_ocr.plate.config import PlateConfig


class TestUnifiedCTCHead:
    def test_forward_applies_mask_and_log_softmax(
        self, plate_config: PlateConfig
    ):
        config = plate_config
        mask_table = build_mask_table(config)
        input_size = 64
        head = UnifiedCTCHead(
            input_size=input_size,
            union_alphabet_size=config.union_alphabet_size,
        )

        B, T = 2, 5
        lstm_out = torch.randn(B, T, input_size)
        country_idx = torch.tensor([0, 0])
        mask = mask_table[country_idx]

        result = head.forward(lstm_out, mask)

        assert result.shape == (B, T, config.union_alphabet_size)
        probs = result.exp()
        assert torch.allclose(
            probs.sum(dim=-1),
            torch.ones(B, T),
            atol=1e-5,
        )

    def test_forward_masks_disallowed_chars(self, plate_config: PlateConfig):
        config = plate_config
        mask_table = build_mask_table(config)
        input_size = 64
        head = UnifiedCTCHead(
            input_size=input_size,
            union_alphabet_size=config.union_alphabet_size,
        )

        country = "RU"
        country_idx_val = config.country_list.index(country)
        B, T = 1, 1
        lstm_out = torch.randn(B, T, input_size)
        country_idx = torch.tensor([country_idx_val])
        mask = mask_table[country_idx]

        result = head.forward(lstm_out, mask)
        probs = result.exp()

        union = config.union_alphabet
        for ch_idx, ch in enumerate(union):
            if ch not in config.get_alphabet(country):
                assert probs[0, 0, ch_idx].item() < 0.05, (
                    f"Char {ch} should be masked for {country}"
                )

    def test_forward_raw_no_mask(self, plate_config: PlateConfig):
        config = plate_config
        mask_table = build_mask_table(config)
        input_size = 64
        head = UnifiedCTCHead(
            input_size=input_size,
            union_alphabet_size=config.union_alphabet_size,
        )

        B, T = 2, 5
        lstm_out = torch.randn(B, T, input_size)

        result = head.forward_raw(lstm_out)

        assert result.shape == (B, T, config.union_alphabet_size)
        country_idx = torch.tensor([0, 0])
        mask = mask_table[country_idx]
        masked = head.forward(lstm_out, mask)
        assert not torch.allclose(result, masked)

    def test_forward_raw_applies_proj_and_fc(self, plate_config: PlateConfig):
        config = plate_config
        input_size = 64
        hidden_size = 32
        head = UnifiedCTCHead(
            input_size=input_size,
            hidden_size=hidden_size,
            union_alphabet_size=config.union_alphabet_size,
        )

        B, T = 2, 5
        lstm_out = torch.randn(B, T, input_size)
        result = head.forward_raw(lstm_out)

        assert result.shape == (B, T, config.union_alphabet_size)

    def test_mask_value_is_soft(self, plate_config: PlateConfig):
        from redstar_plate_ocr.nn.mask_table import MASK_VALUE

        assert MASK_VALUE < 0.0
        assert MASK_VALUE != float("-inf")

    def test_no_hidden_size_uses_sequential(self, plate_config: PlateConfig):
        head = UnifiedCTCHead(
            input_size=64,
            union_alphabet_size=plate_config.union_alphabet_size,
        )
        assert isinstance(head.proj, torch.nn.Sequential)

    def test_with_hidden_size_uses_sequential(self, plate_config: PlateConfig):
        head = UnifiedCTCHead(
            input_size=64,
            hidden_size=32,
            union_alphabet_size=plate_config.union_alphabet_size,
        )
        assert isinstance(head.proj, torch.nn.Sequential)


class TestBuildMaskTable:
    """Tests for build_mask_table function."""

    def test_mask_table_shape(self, plate_config: PlateConfig):
        """mask_table has shape (num_countries, union_alphabet_size)."""
        config = plate_config
        mask = build_mask_table(config)

        # Only enabled countries have masks in build_mask_table
        num_countries = len(config.country_list)
        assert mask.shape == (num_countries, config.union_alphabet_size)

    def test_mask_table_is_float(self, plate_config: PlateConfig):
        """mask_table is float tensor (0.0 / MASK_VALUE)."""
        from redstar_plate_ocr.nn.mask_table import MASK_VALUE

        config = plate_config
        mask = build_mask_table(config)

        assert mask.dtype == torch.float32
        assert (mask == 0.0).logical_or(mask == MASK_VALUE).all()

    def test_mask_table_correctness(self, plate_config: PlateConfig):
        """Each row has 0.0 for chars in that country's alphabet."""
        config = plate_config
        mask = build_mask_table(config)
        union = config.union_alphabet

        for c_idx, country in enumerate(config.country_list):
            alphabet = config.get_alphabet(country)
            for ch_idx, ch in enumerate(union):
                expected = ch in alphabet
                is_allowed = mask[c_idx, ch_idx].item() == 0.0
                assert is_allowed == expected, (
                    f"country={country}, char={ch}: "
                    f"expected {expected}, got "
                    f"{'allowed' if is_allowed else 'masked'}"
                )

    def test_mask_table_blank_always_allowed(self, plate_config: PlateConfig):
        """Blank (last index) is always 0.0."""
        config = plate_config
        mask = build_mask_table(config)

        for c_idx in range(mask.shape[0]):
            assert mask[c_idx, -1].item() == 0.0

    def test_mask_table_from_yaml(self, plate_config: PlateConfig):
        """build_mask_table works with real plate.yaml config."""
        config = plate_config
        mask = build_mask_table(config)

        assert mask.shape[0] == config.num_countries
        assert mask.shape[1] == config.union_alphabet_size
        assert mask.dtype == torch.float32
