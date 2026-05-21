"""PostProcessor: confusion correction, forbidden filter, pattern."""

from __future__ import annotations

from redstar_plate_ocr.plate.config import PlateConfig
from redstar_plate_ocr.plate.confusion import (
    adjacent_swap_correct,
    correct_confusions,
)
from redstar_plate_ocr.plate.forbidden import ForbiddenFilter
from redstar_plate_ocr.plate.pattern import PatternValidator
from redstar_plate_ocr.plate.results import RawResult, RecognitionResult


class PostProcessor:
    """Apply domain post-processing to raw recognition results.

    Steps: confusion correction → adjacent-swap correction
    → forbidden filter → pattern validation.

    The adjacent-swap step is controlled by *enable_swap_correction*
    and requires CTC logits + alignment from RawResult.
    """

    def __init__(
        self,
        plate_config: PlateConfig,
        enable_swap_correction: bool = True,
    ) -> None:
        self.plate_config = plate_config
        self.enable_swap_correction = enable_swap_correction
        self._filter_cache: dict[str, ForbiddenFilter] = {}
        self._validator_cache: dict[
            tuple[str, str, str], PatternValidator
        ] = {}

    def process(
        self,
        raw: RawResult,
        hypotheses: list[tuple[str, float]] | None = None,
    ) -> RecognitionResult:
        """Apply post-processing to raw result.

        Pipeline: confusion → swap → forbidden → pattern.

        Args:
            raw: Raw recognition result.
            hypotheses: Optional list of (text, confidence) for
                forbidden filter. If provided, forbidden filter
                selects best valid hypothesis.

        Returns:
            Post-processed RecognitionResult.
        """
        text = raw.text
        needs_review = raw.needs_review
        country = raw.country
        region = self.plate_config.regions.get(country)

        # Step 1: confusion correction (fix B↔8, O↔0 etc.)
        if region is not None:
            text = correct_confusions(
                text,
                region.get_patterns(),
                region.valid_chars.letters,
                region.valid_chars.digits,
            )

        # Step 2: adjacent-swap correction (fix CX→XC etc.)
        # Only when enabled AND logits/alignment are available AND
        # the model is not highly confident (uncertainty-driven).
        if self.enable_swap_correction and region is not None:
            text = adjacent_swap_correct(
                text,
                region.get_patterns(),
                region.valid_chars.letters,
                region.valid_chars.digits,
                ctc_logits=raw.ctc_logits,
                ctc_alignment=raw.ctc_alignment,
                alphabet=self.plate_config.union_alphabet,
                text_confidence=raw.text_confidence,
            )

        # Step 3: forbidden filter
        text, needs_review = self._apply_forbidden(
            text,
            needs_review,
            country,
            region,
            hypotheses,
        )

        # Step 4: pattern validation
        if region is not None:
            text, needs_review = self._apply_pattern_validation(
                text,
                needs_review,
                region,
            )

        return RecognitionResult(
            text=text,
            text_confidence=raw.text_confidence,
            country=raw.country,
            country_confidence=raw.country_confidence,
            plate_type=raw.plate_type,
            needs_review=needs_review,
        )

    def _apply_forbidden(
        self,
        text: str,
        needs_review: bool,
        country: str,
        region,
        hypotheses: list[tuple[str, float]] | None,
    ) -> tuple[str, bool]:
        """Filter out forbidden combinations."""
        if region is None:
            return text, needs_review
        forbidden = region.forbidden_combos
        if not forbidden:
            return text, needs_review
        filt = self._get_filter(country, forbidden)
        if hypotheses:
            return filt.filter(hypotheses)
        if filt.contains_forbidden(text):
            return text, True
        return text, needs_review

    def _get_filter(
        self,
        country: str,
        forbidden: list[str],
    ) -> ForbiddenFilter:
        """Get or create cached ForbiddenFilter."""
        if country not in self._filter_cache:
            self._filter_cache[country] = ForbiddenFilter(
                forbidden,
            )
        return self._filter_cache[country]

    def _apply_pattern_validation(
        self,
        text: str,
        needs_review: bool,
        region,
    ) -> tuple[str, bool]:
        """Validate and correct text against region patterns."""
        patterns = region.get_patterns()
        validator = self._get_validator(
            patterns[0],
            region.valid_chars.letters,
            region.valid_chars.digits,
        )
        if len(patterns) > 1:
            val_result = validator.validate_multi(text, patterns)
        else:
            val_result = validator.validate(text)
        text = val_result.text
        if val_result.corrected:
            needs_review = True
        return text, needs_review

    def _get_validator(
        self,
        pattern: str,
        valid_letters: str,
        valid_digits: str,
    ) -> PatternValidator:
        """Get or create cached PatternValidator."""
        key = (pattern, valid_letters, valid_digits)
        if key not in self._validator_cache:
            self._validator_cache[key] = PatternValidator(
                pattern=pattern,
                valid_letters=valid_letters,
                valid_digits=valid_digits,
            )
        return self._validator_cache[key]
