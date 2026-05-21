"""Tests for metrics: CER, Accuracy, CharacterAccuracy, NED, BSR, ATR."""

from redstar_plate_ocr.nn.metrics import (
    Accuracy,
    AdjacentTranspositionRate,
    BigramSwapRate,
    CharacterAccuracy,
    CharacterErrorRate,
    NormalizedEditDistance,
    compute_per_group_metrics,
    levenshtein_distance,
)


class TestLevenshteinDistance:
    """Tests for levenshtein_distance helper."""

    def test_identical(self):
        assert levenshtein_distance("ABC", "ABC") == 0

    def test_one_substitution(self):
        assert levenshtein_distance("ABC", "ABD") == 1

    def test_empty(self):
        assert levenshtein_distance("", "ABC") == 3

    def test_both_empty(self):
        assert levenshtein_distance("", "") == 0

    def test_insertion(self):
        assert levenshtein_distance("AB", "ABC") == 1

    def test_deletion(self):
        assert levenshtein_distance("ABC", "AB") == 1


class TestCharacterErrorRate:
    """Tests for CharacterErrorRate metric."""

    def test_cer_identical(self):
        cer = CharacterErrorRate()
        cer.update(["ABC"], ["ABC"])
        assert cer.compute() == 0.0

    def test_cer_different(self):
        cer = CharacterErrorRate()
        cer.update(["ABC"], ["ABD"])
        # 1 substitution / 3 chars = 1/3
        assert abs(cer.compute() - 1.0 / 3.0) < 1e-6

    def test_cer_reset(self):
        cer = CharacterErrorRate()
        cer.update(["ABC"], ["ABD"])
        cer.reset()
        assert cer.compute() == 0.0

    def test_cer_multiple_samples(self):
        cer = CharacterErrorRate()
        cer.update(["ABC", "DEF"], ["ABC", "DEG"])
        # (0 + 1) / (3 + 3) = 1/6
        assert abs(cer.compute() - 1.0 / 6.0) < 1e-6


class TestPlateAccuracy:
    """Tests for Accuracy metric (formerly PlateAccuracy)."""

    def test_all_correct(self):
        acc = Accuracy()
        acc.update(["A123AA12", "K567BB34"], ["A123AA12", "K567BB34"])
        assert acc.compute() == 1.0

    def test_partial(self):
        acc = Accuracy()
        acc.update(
            ["A123AA12", "K567BB34", "M000XX00"],
            ["A123AA12", "K567BB35", "M000XX00"],
        )
        # 2 of 3 correct
        assert abs(acc.compute() - 2.0 / 3.0) < 1e-6

    def test_none_correct(self):
        acc = Accuracy()
        acc.update(["ABC"], ["DEF"])
        assert acc.compute() == 0.0

    def test_reset(self):
        acc = Accuracy()
        acc.update(["ABC"], ["ABC"])
        acc.reset()
        assert acc.compute() == 0.0


class TestCountryAccuracy:
    """Tests for Accuracy metric (formerly CountryAccuracy)."""

    def test_all_correct(self):
        acc = Accuracy()
        acc.update(["RU", "KZ", "BY"], ["RU", "KZ", "BY"])
        assert acc.compute() == 1.0

    def test_partial(self):
        acc = Accuracy()
        acc.update(["RU", "KZ", "BY"], ["RU", "UZ", "BY"])
        # 2 of 3
        assert abs(acc.compute() - 2.0 / 3.0) < 1e-6

    def test_reset(self):
        acc = Accuracy()
        acc.update(["RU"], ["RU"])
        acc.reset()
        assert acc.compute() == 0.0


class TestFormatAccuracy:
    """Tests for Accuracy metric (formerly FormatAccuracy)."""

    def test_all_correct(self):
        acc = Accuracy()
        acc.update(["standard", "square"], ["standard", "square"])
        assert acc.compute() == 1.0

    def test_half_correct(self):
        acc = Accuracy()
        acc.update(["standard", "square"], ["square", "square"])
        # 1 of 2
        assert abs(acc.compute() - 0.5) < 1e-6

    def test_reset(self):
        acc = Accuracy()
        acc.update(["standard"], ["standard"])
        acc.reset()
        assert acc.compute() == 0.0


class TestCharacterAccuracy:
    """Tests for CharacterAccuracy metric."""

    def test_identical(self):
        acc = CharacterAccuracy()
        acc.update(["ABC"], ["ABC"])
        assert acc.compute() == 1.0

    def test_one_substitution(self):
        acc = CharacterAccuracy()
        acc.update(["ABC"], ["ABD"])
        # correct = 3-1=2, total=3 => 2/3
        assert abs(acc.compute() - 2.0 / 3.0) < 1e-6

    def test_multiple_samples(self):
        acc = CharacterAccuracy()
        acc.update(["ABC", "DEF"], ["ABC", "DEG"])
        # (3-0) + (3-1) = 5, total=6 => 5/6
        assert abs(acc.compute() - 5.0 / 6.0) < 1e-6

    def test_reset(self):
        acc = CharacterAccuracy()
        acc.update(["ABC"], ["ABC"])
        acc.reset()
        assert acc.compute() == 0.0

    def test_no_data(self):
        acc = CharacterAccuracy()
        assert acc.compute() == 0.0


class TestNormalizedEditDistance:
    """Tests for NormalizedEditDistance metric."""

    def test_identical(self):
        ned = NormalizedEditDistance()
        assert ned(["ABC"], ["ABC"]) == 1.0

    def test_one_substitution(self):
        ned = NormalizedEditDistance()
        result = ned(["ABC"], ["ABD"])
        # NED = 1 - 1/3 = 2/3
        assert abs(result - 2.0 / 3.0) < 1e-6

    def test_completely_different(self):
        ned = NormalizedEditDistance()
        result = ned(["ABC"], ["XYZ"])
        # NED = 1 - 3/3 = 0.0
        assert abs(result - 0.0) < 1e-6

    def test_empty_both(self):
        ned = NormalizedEditDistance()
        assert ned([""], [""]) == 1.0

    def test_empty_vs_nonempty(self):
        ned = NormalizedEditDistance()
        result = ned([""], ["ABC"])
        # NED = 1 - 3/3 = 0.0
        assert abs(result - 0.0) < 1e-6

    def test_multiple_samples(self):
        ned = NormalizedEditDistance()
        result = ned(["ABC", "DEF"], ["ABC", "DEG"])
        # (1.0 + (1-1/3)) / 2 = (1 + 2/3) / 2 = 5/6
        assert abs(result - 5.0 / 6.0) < 1e-6

    def test_no_predictions(self):
        ned = NormalizedEditDistance()
        assert ned([], []) == 0.0


class TestBigramSwapRate:
    """Tests for BigramSwapRate metric."""

    def test_no_swaps(self):
        bsr = BigramSwapRate()
        assert bsr(["A123AA12"], ["A123AA12"]) == 0.0

    def test_swap_detected(self):
        bsr = BigramSwapRate()
        # target has 'KH', pred has 'HK' at same position
        assert bsr(["092AHK05"], ["092AKH05"]) == 1.0

    def test_swap_not_at_same_position(self):
        bsr = BigramSwapRate()
        # 'KH' in target but not swapped in prediction
        assert bsr(["092AKH05"], ["092AKH05"]) == 0.0

    def test_custom_bigrams(self):
        bsr = BigramSwapRate(bigrams=["AB", "BA"])
        assert bsr(["A1BA12"], ["A1AB12"]) == 1.0

    def test_mixed_samples(self):
        bsr = BigramSwapRate()
        result = bsr(
            ["092AHK05", "123ABC45"],
            ["092AKH05", "123ABC45"],
        )
        # 1 swap out of 2
        assert abs(result - 0.5) < 1e-6

    def test_empty(self):
        bsr = BigramSwapRate()
        assert bsr([], []) == 0.0


class TestAdjacentTranspositionRate:
    """Tests for AdjacentTranspositionRate metric."""

    def test_transposition_detected(self):
        atr = AdjacentTranspositionRate()
        # 'AB' -> 'BA' is adjacent transposition
        result = atr(["BA"], ["AB"])
        assert abs(result - 1.0) < 1e-6

    def test_no_errors(self):
        atr = AdjacentTranspositionRate()
        assert atr(["ABC"], ["ABC"]) == 0.0

    def test_substitution_not_transposition(self):
        atr = AdjacentTranspositionRate()
        # edit_distance=1, not a transposition
        result = atr(["AXC"], ["ABC"])
        assert abs(result - 0.0) < 1e-6

    def test_two_substitutions_not_transposition(self):
        atr = AdjacentTranspositionRate()
        # edit_distance=2 but not adjacent swap
        result = atr(["XBC"], ["ABD"])
        assert abs(result - 0.0) < 1e-6

    def test_length_mismatch_not_transposition(self):
        atr = AdjacentTranspositionRate()
        result = atr(["AB"], ["ABC"])
        assert abs(result - 0.0) < 1e-6

    def test_mixed_samples(self):
        atr = AdjacentTranspositionRate()
        result = atr(["BAC", "XYZ", "DEF"], ["ABC", "ABC", "DEG"])
        # Errors: all 3. Transpositions: only 'BAC' vs 'ABC' = 1.
        # ATR = 1/3
        assert abs(result - 1.0 / 3.0) < 1e-6

    def test_all_correct_returns_zero(self):
        atr = AdjacentTranspositionRate()
        assert atr(["ABC", "DEF"], ["ABC", "DEF"]) == 0.0


class TestComputePerGroupMetrics:
    """Tests for compute_per_group_metrics function."""

    def test_single_group(self):
        result = compute_per_group_metrics(
            ["ABC", "ABD"],
            ["ABC", "ABC"],
            ["ru", "ru"],
        )
        assert "ru" in result
        assert abs(result["ru"]["cer"] - 0.5 / 3.0) < 1e-6
        assert abs(result["ru"]["plate_acc"] - 0.5) < 1e-6

    def test_multiple_groups(self):
        result = compute_per_group_metrics(
            ["ABC", "DEF", "XYZ"],
            ["ABC", "DEG", "XYZ"],
            ["ru", "kz", "ru"],
        )
        assert set(result.keys()) == {"kz", "ru"}
        # ru group: ABC==ABC, XYZ==XYZ => cer=0, acc=1.0
        assert abs(result["ru"]["plate_acc"] - 1.0) < 1e-6
        # kz group: DEF vs DEG => cer=1/3, acc=0.0
        assert abs(result["kz"]["plate_acc"] - 0.0) < 1e-6

    def test_empty_input(self):
        result = compute_per_group_metrics([], [], [])
        assert result == {}
