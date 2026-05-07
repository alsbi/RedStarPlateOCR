"""Tests for pipeline utility functions."""

import numpy as np
import torch

from redstar_plate_ocr.pipeline.utils import resolve_country_from_probs


def test_resolve_country_normal_index():
    """Normal argmax within country_list returns that country."""
    probs = np.array([0.8, 0.1, 0.05, 0.05])
    country_list = ["RU", "KZ", "UA", "BY"]
    result = resolve_country_from_probs(probs, country_list)
    assert result == "RU"


def test_resolve_country_argmax():
    """argmax returns correct country (no sentinel)."""
    probs = np.array([0.1, 0.3, 0.1, 0.1])
    country_list = ["RU", "KZ", "UA", "BY"]
    result = resolve_country_from_probs(probs, country_list)
    assert result == "KZ"


def test_resolve_country_torch_tensor():
    """Works with torch tensors too."""
    probs = torch.tensor([0.1, 0.7, 0.1, 0.1])
    country_list = ["RU", "KZ", "UA", "BY"]
    result = resolve_country_from_probs(probs, country_list)
    assert result == "KZ"
