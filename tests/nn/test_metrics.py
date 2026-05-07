"""Tests for metrics: CER, Accuracy, CharacterAccuracy."""

from redstar_plate_ocr.nn.metrics import (
    Accuracy,
    CharacterAccuracy,
    CharacterErrorRate,
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
