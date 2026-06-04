"""PlateOCRModel: full OCR model combining all components."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from redstar_plate_ocr.nn.backbone import BackboneOutput, PlateBackbone
from redstar_plate_ocr.nn.char_aux import CharAuxHead
from redstar_plate_ocr.nn.compression import AdaptiveCompression
from redstar_plate_ocr.nn.film import ContextFiLM
from redstar_plate_ocr.nn.fusion import MultiScaleFusion
from redstar_plate_ocr.nn.heads import (
    CountryHead,
    FormatHead,
    PositionAwareCountryHead,
    UnifiedCTCHead,
)
from redstar_plate_ocr.nn.lstm import PlateBiLSTM
from redstar_plate_ocr.nn.mask_table import (
    build_mask_table,
    build_positional_mask_table,
)
from redstar_plate_ocr.nn.positional import SinusoidalPositionalEncoding
from redstar_plate_ocr.nn.temporal import TemporalBridge
from redstar_plate_ocr.nn.types import ModelOutput
from redstar_plate_ocr.plate.config import PlateConfig

# Fallback sequence length when neither standard nor square compressed
# output is available (both paths skipped — extremely rare edge case).
_DEFAULT_LSTM_MAX_LEN = 128


class PlateOCRModel(nn.Module):
    """Full plate OCR model: backbone + fusion + heads + compression + LSTM."""

    @staticmethod
    def _build_backbone(
        backbone_cfg: dict | None,
    ) -> tuple[PlateBackbone, int, int]:
        """Build backbone and return (backbone, stage1_ch, final_ch).

        final_ch is the number of channels in the backbone's final
        output (stage3_channels if set, otherwise stage2_channels).
        """
        bc = backbone_cfg or {}
        backbone = PlateBackbone(**bc)
        stage1_ch = bc.get("stage1_channels", 128)
        stage2_ch = bc.get("stage2_channels", 256)
        s3c = bc.get("stage3_channels")
        final_ch = s3c if s3c is not None else stage2_ch
        return backbone, stage1_ch, final_ch

    @staticmethod
    def _build_compression(
        final_ch: int,
        canvas_height: int,
        canvas_width: int,
    ) -> AdaptiveCompression:
        """Build adaptive compression module."""
        return AdaptiveCompression(
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            in_channels=final_ch,
        )

    # Aspect-ratio threshold: w/h < threshold → square, >= → standard.
    # Based on dataset analysis (max square ratio=1.91, min standard=2.00),
    # threshold=2.0 gives 100% separation on real data.
    FORMAT_RATIO_THRESHOLD: float = 2.0

    # When FormatHead confidence (softmax max prob) falls below this
    # threshold during inference, we override its prediction with the
    # aspect-ratio heuristic.  0.8 = only override when quite uncertain.
    FORMAT_CONFIDENCE_THRESHOLD: float = 0.8

    def __init__(
        self,
        plate_config: PlateConfig,
        backbone_cfg: dict | None = None,
        classification_cfg: dict | None = None,
        lstm_cfg: dict | None = None,
        canvas_height: int = 80,
        canvas_width: int = 256,
        head_hidden: int | None = None,
        char_aux: bool | dict = False,
        film_cfg: dict | None = None,
        temporal_bridge_cfg: dict | None = None,
    ):
        super().__init__()
        self.plate_config = plate_config
        self.country_list = plate_config.country_list
        self._country_to_idx = {c: i for i, c in enumerate(self.country_list)}
        self._canvas_height = canvas_height
        self._canvas_width = canvas_width

        self.backbone, stage1_ch, final_ch = self._build_backbone(backbone_cfg)
        self.fusion = MultiScaleFusion(
            stage1_channels=stage1_ch,
            stage2_channels=final_ch,
        )

        cc = classification_cfg or {}
        self.format_head = self._build_format_head(final_ch, cc, head_hidden)
        self.country_head = self._build_country_head(
            final_ch, cc, plate_config, head_hidden
        )
        self.compression = self._build_compression(
            final_ch, canvas_height, canvas_width
        )

        self._init_lstm_and_mask(cc, lstm_cfg, canvas_height, canvas_width)
        self._init_ctc_head(lstm_cfg, head_hidden, plate_config)
        self._init_char_aux(char_aux, final_ch, plate_config)
        self._init_film(film_cfg, lstm_cfg, plate_config)
        self._init_temporal_bridge(temporal_bridge_cfg, lstm_cfg)

    def _init_char_aux(
        self,
        char_aux: bool | dict,
        final_ch: int,
        plate_config: PlateConfig,
    ) -> None:
        """Initialize char-auxiliary head if enabled."""
        ca_cfg: dict = char_aux if isinstance(char_aux, dict) else {}
        self.char_aux_enabled = (
            ca_cfg.get("enabled", False)
            if isinstance(char_aux, dict)
            else bool(char_aux)
        )
        self._enable_char_aux_film = ca_cfg.get("enable_film", False)
        if not self.char_aux_enabled:
            self._enable_char_aux_film = False
            return
        self.char_aux_head = CharAuxHead(
            in_channels=final_ch,
            max_alphabet_size=plate_config.union_alphabet_size,
            enable_film=self._enable_char_aux_film,
            num_countries=(
                plate_config.num_countries
                if self._enable_char_aux_film
                else None
            ),
            country_emb_dim=ca_cfg.get("country_emb_dim", 128),
            format_emb_dim=ca_cfg.get("format_emb_dim", 64),
            hidden_dim=ca_cfg.get("hidden_dim", 1024),
        )

    def _init_film(
        self,
        film_cfg: dict | None,
        lstm_cfg: dict | None,
        plate_config: PlateConfig,
    ) -> None:
        """Initialize ContextFiLM module if enabled."""
        fc = film_cfg or {}
        self._enable_film = fc.get("enable_film", False)
        if not self._enable_film:
            self.context_film = None  # type: ignore[assignment]
            return
        lc = lstm_cfg or {}
        lstm_input_size = lc.get("input_size", 256)
        self.context_film = ContextFiLM(
            num_countries=plate_config.num_countries,
            country_emb_dim=fc.get("country_emb_dim", 128),
            format_emb_dim=fc.get("format_emb_dim", 64),
            feature_dim=lstm_input_size,
            hidden_dim=fc.get("hidden_dim", 1024),
        )

    def _init_temporal_bridge(
        self,
        temporal_bridge_cfg: dict | None,
        lstm_cfg: dict | None,
    ) -> None:
        """Initialize TemporalBridge module if enabled."""
        tbc = temporal_bridge_cfg or {}
        self._enable_temporal_bridge = tbc.get(
            "enable_temporal_bridge",
            False,
        )
        if not self._enable_temporal_bridge:
            self.temporal_bridge = None  # type: ignore[assignment]
            return
        lc = lstm_cfg or {}
        lstm_input_size = lc.get("input_size", 256)
        self.temporal_bridge = TemporalBridge(channels=lstm_input_size)

    @staticmethod
    def _build_format_head(
        in_ch: int,
        classification_cfg: dict,
        head_hidden: int | None,
    ) -> FormatHead:
        """Build format classification head."""
        kwargs = {
            k: v for k, v in classification_cfg.items() if k in ("dropout",)
        }
        return FormatHead(in_ch, hidden_size=head_hidden, **kwargs)

    @staticmethod
    def _build_country_head(
        in_ch: int,
        classification_cfg: dict,
        plate_config: PlateConfig,
        head_hidden: int | None,
    ) -> CountryHead | PositionAwareCountryHead:
        """Build country classification head."""
        country_cfg = classification_cfg.get("country_head", {})
        if country_cfg.get("pos_aware", True):
            return PositionAwareCountryHead(
                in_channels=in_ch,
                num_countries=plate_config.num_countries,
                conv_channels=country_cfg.get("conv_channels", 144),
                grid_rows=country_cfg.get("grid_rows", 2),
                grid_cols=country_cfg.get("grid_cols", 3),
                hidden_size=country_cfg.get("hidden_size", 288),
                dropout=country_cfg.get("dropout", 0.3),
            )
        kwargs = {
            k: v for k, v in classification_cfg.items() if k in ("dropout",)
        }
        return CountryHead(
            in_ch,
            plate_config.num_countries,
            hidden_size=head_hidden,
            **kwargs,
        )

    @staticmethod
    def _build_mask_tables(
        classification_cfg: dict,
        plate_config: PlateConfig,
        canvas_height: int,
        canvas_width: int,
    ) -> tuple[Tensor, Tensor]:
        """Build flat and positional mask lookup tables."""
        ctc_cfg = classification_cfg.get("unified_ctc_head", {})
        mask_value = ctc_cfg.get("mask_value", -15.0)
        use_positional = ctc_cfg.get("positional_mask", True)
        max_seq_len = max(canvas_width, canvas_height) // 4 * 2
        flat_mask = build_mask_table(plate_config, mask_value=mask_value)
        if use_positional:
            pos_mask = build_positional_mask_table(
                plate_config,
                max_seq_len=max_seq_len,
                mask_value=mask_value,
            )
        else:
            pos_mask = flat_mask.unsqueeze(1).expand(-1, max_seq_len, -1)
        return flat_mask, pos_mask

    def _init_lstm_and_mask(
        self,
        classification_cfg: dict,
        lstm_cfg: dict | None,
        canvas_height: int,
        canvas_width: int,
    ) -> None:
        """Initialize BiLSTM, positional encoding, mask tables, and ramp."""
        lc = lstm_cfg or {}
        # Extract positional-encoding params before passing lc to BiLSTM
        lstm_input_size = lc.get("input_size", 256)
        pe_dropout = lc.get("positional_dropout", 0.0)
        lstm_kwargs = {
            k: v for k, v in lc.items() if k not in ("positional_dropout",)
        }
        self.bilstm = PlateBiLSTM(**lstm_kwargs)

        # Sinusoidal positional encoding — gives the LSTM an explicit
        # absolute-position signal so it can distinguish horizontal
        # order of adjacent same-type characters (e.g. CX vs XC).
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=lstm_input_size,
            max_len=max(canvas_width, canvas_height) // 2 * 2,
            dropout=pe_dropout,
        )

        flat_mask, pos_mask = self._build_mask_tables(
            classification_cfg,
            self.plate_config,
            canvas_height,
            canvas_width,
        )
        self.register_buffer("_flat_mask_table", flat_mask)
        self.register_buffer("_pos_mask_table", pos_mask)
        self._flat_mask_table: Tensor
        self._pos_mask_table: Tensor

        # No-mask table: all zeros = every symbol in union alphabet allowed.
        # Used during warmup when we don't want to constrain predictions.
        no_mask = torch.zeros_like(flat_mask)
        self.register_buffer("_no_mask_table", no_mask)
        self._no_mask_table: Tensor

        head_cfg = classification_cfg.get("unified_ctc_head", {})
        self._mask_ramp_warmup = head_cfg.get("mask_ramp_warmup", 3)
        self._mask_ramp_epochs = head_cfg.get("mask_ramp_epochs", 8)
        self._mask_disable_warmup = head_cfg.get("mask_disable_warmup", True)

    def _init_ctc_head(  # noqa: N802
        self,
        lstm_cfg: dict | None,
        head_hidden: int | None,
        plate_config: PlateConfig,
    ) -> None:
        """Initialize the unified CTC head."""
        lc = lstm_cfg or {}
        lstm_hidden = lc.get("hidden_size", 256)
        self.ctc_head = UnifiedCTCHead(
            input_size=lstm_hidden * 2,
            hidden_size=head_hidden,
            union_alphabet_size=plate_config.union_alphabet_size,
        )

    def _resolve_batch(
        self,
        country_logits: Tensor,
        format_logits: Tensor,
        gt_countries: list[str] | None,
        gt_plate_types: list[str] | None,
        scheduled_sampling_prob: float,
        orig_h: Tensor | None = None,
        orig_w: Tensor | None = None,
    ) -> tuple[list[str], list[str], Tensor]:
        """Resolve countries, plate_types, country_indices.

        Country always uses GT (teacher forcing) because wrong
        country → wrong CTC mask → toxic gradient.
        Format uses scheduled sampling when enabled.
        During inference, format uses ratio-based fallback when
        confidence is low.
        """
        pred_country_idx = country_logits[:, : len(self.country_list)].argmax(
            dim=1
        )
        pred_format_idx = format_logits.argmax(dim=1)
        countries, country_indices = self._resolve_countries(
            gt_countries,
            pred_country_idx,
        )

        b = format_logits.shape[0]
        use_pred_mask: Tensor | None = None
        if gt_plate_types is not None and scheduled_sampling_prob > 0.0:
            rand_vals = torch.rand(b, device=format_logits.device)
            use_pred_mask = rand_vals < scheduled_sampling_prob
        plate_types = self._resolve_plate_types(
            gt_plate_types,
            pred_format_idx,
            format_logits,
            orig_h if orig_h is not None else torch.zeros(b),
            orig_w if orig_w is not None else torch.zeros(b),
            use_pred_mask,
            b,
        )
        return countries, plate_types, country_indices

    def _resolve_countries(
        self,
        gt_countries: list[str] | None,
        pred_country_idx: Tensor,
    ) -> tuple[list[str], Tensor]:
        """Resolve country — always GT during training."""
        if gt_countries is not None:
            indices_list = [self._country_to_idx[c] for c in gt_countries]
            country_indices = torch.tensor(
                indices_list, device=pred_country_idx.device
            )
            return list(gt_countries), country_indices
        indices_list = pred_country_idx.tolist()
        countries = [self.country_list[i] for i in indices_list]
        return countries, pred_country_idx

    @staticmethod
    def _apply_scheduled_sampling(
        base_types: list[str],
        fmt_indices: list[int],
        use_pred_mask: Tensor,
    ) -> list[str]:
        """Apply scheduled sampling to plate types."""
        return [
            ("square" if fmt_indices[i] == 1 else "standard")
            if bool(use_pred_mask[i])
            else base_types[i]
            for i in range(len(base_types))
        ]

    def _resolve_plate_types(
        self,
        gt_plate_types: list[str] | None,
        pred_format_idx: Tensor,
        format_logits: Tensor,
        orig_h: Tensor,
        orig_w: Tensor,
        use_pred_mask: Tensor | None,
        b: int,
    ) -> list[str]:
        """Resolve plate type with ratio-based fallback.

        During training: uses GT plate_types (with optional scheduled
        sampling).  During inference: uses FormatHead prediction, but
        overrides with aspect-ratio heuristic when confidence is low
        — because misclassifying a square plate as standard collapses
        its temporal resolution from ~68 to ~20 frames, destroying
        recognition quality.
        """
        if gt_plate_types is not None:
            plate_types = list(gt_plate_types)
            if use_pred_mask is not None:
                fmt_indices = pred_format_idx.tolist()
                return self._apply_scheduled_sampling(
                    plate_types, fmt_indices, use_pred_mask
                )
            return plate_types

        # Inference path: FormatHead + ratio fallback
        fmt_indices = pred_format_idx.tolist()
        plate_types = [
            "square" if idx == 1 else "standard" for idx in fmt_indices
        ]

        # Ratio-based override when FormatHead confidence is low
        probs = format_logits.softmax(dim=1)
        max_probs = probs.max(dim=1).values
        ratios = orig_w.float() / orig_h.float().clamp(min=1)
        for i in range(b):
            if max_probs[i] < self.FORMAT_CONFIDENCE_THRESHOLD:
                # Low confidence → trust aspect ratio
                if ratios[i] < self.FORMAT_RATIO_THRESHOLD:
                    plate_types[i] = "square"
                else:
                    plate_types[i] = "standard"
        return plate_types

    def _compute_mask_ramp(self, epoch: int) -> float:
        """Compute mask ramp factor.

        Returns:
            -1.0  — no mask at all (full union alphabet, warmup phase)
             0.0  — flat mask only (country-level character filtering)
             0..1 — linear blend flat → positional
             1.0  — full positional mask
        """
        if not self.training:
            return 1.0
        if self._mask_disable_warmup and epoch < self._mask_ramp_warmup:
            return -1.0
        if epoch < self._mask_ramp_warmup:
            return 0.0
        progress = (epoch - self._mask_ramp_warmup) / self._mask_ramp_epochs
        return min(progress, 1.0)

    def _run_compression(
        self,
        features: Tensor,
        orig_h: Tensor,
        orig_w: Tensor,
        content_mask: Tensor,
        sq_mask: torch.BoolTensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Run compression paths for standard and square samples."""
        if not sq_mask.all():
            std = self.compression.forward_standard(
                features,
                orig_h,
                orig_w,
                content_mask=content_mask,
            )
        else:
            std = None
        if sq_mask.any():
            sq = self.compression.forward_square(
                features,
                orig_h,
                orig_w,
                content_mask=content_mask,
            )
        else:
            sq = None
        return std, sq

    def forward(
        self,
        images: Tensor,
        orig_h: Tensor,
        orig_w: Tensor,
        gt_countries: list[str] | None = None,
        gt_plate_types: list[str] | None = None,
        scheduled_sampling_prob: float = 0.0,
        epoch: int = 0,
    ) -> ModelOutput:
        """Full forward pass."""
        backbone_out: BackboneOutput = self.backbone(images)
        features = self.fusion(backbone_out.stage1, backbone_out.final)

        # Ensure orig_h/orig_w are on the same device as features —
        # guards against accidental CPU tensors passed from user code.
        orig_h = orig_h.to(features.device)
        orig_w = orig_w.to(features.device)

        content_mask = self.compression.compute_content_mask(
            orig_h,
            orig_w,
        )
        format_logits = self.format_head(
            features,
            content_mask=content_mask,
            orig_h=orig_h,
            orig_w=orig_w,
        )
        country_logits = self.country_head(features, content_mask)

        countries, plate_types, country_indices = self._resolve_batch(
            country_logits,
            format_logits,
            gt_countries,
            gt_plate_types,
            scheduled_sampling_prob,
            orig_h=orig_h,
            orig_w=orig_w,
        )

        # Compression needs GT plate types (not scheduled-sampled ones)
        # to correctly route standard vs square paths during training.
        sample_types = (
            list(gt_plate_types) if gt_plate_types is not None else plate_types
        )
        sq_mask = torch.tensor(
            [t == "square" for t in sample_types],
            device=features.device,
        )
        compressed_std, compressed_sq = self._run_compression(
            features,
            orig_h,
            orig_w,
            content_mask,
            sq_mask,
        )
        # Derive format indices from resolved plate_types for FiLM
        format_indices = torch.tensor(
            [1 if t == "square" else 0 for t in plate_types],
            device=features.device,
            dtype=torch.long,
        )
        lstm_out = self._run_lstm_paths(
            compressed_std,
            compressed_sq,
            sq_mask,
            country_idx=country_indices,
            format_idx=format_indices,
        )

        ramp = self._compute_mask_ramp(epoch)
        T = lstm_out.shape[1]

        # Build effective mask from no-mask / flat / positional tables
        if ramp < 0.0:
            # Warmup: no mask — full union alphabet allowed everywhere
            effective_mask = (
                self._no_mask_table[country_indices]
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
        elif ramp >= 1.0:
            masks = self._pos_mask_table[country_indices]
            effective_mask = masks[:, :T, :]
        elif ramp <= 0.0:
            effective_mask = (
                self._flat_mask_table[country_indices]
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
        else:
            flat = (
                self._flat_mask_table[country_indices]
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            pos = self._pos_mask_table[country_indices][:, :T, :]
            effective_mask = (1.0 - ramp) * flat + ramp * pos

        ctc_output = self.ctc_head(lstm_out, effective_mask)

        char_aux_logits = None
        if self.char_aux_enabled:
            char_aux_logits = self.char_aux_head(
                features,
                content_mask,
                country_idx=country_indices,
                format_idx=format_indices,
            )

        return ModelOutput(
            format_logits=format_logits,
            country_logits=country_logits,
            ctc_output=ctc_output,
            content_mask=content_mask,
            plate_types=plate_types,
            char_aux_logits=char_aux_logits,
        )

    def _apply_pe_film_bridge(
        self,
        x: Tensor,
        country_idx: Tensor | None,
        format_idx: Tensor | None,
    ) -> Tensor:
        """Apply PE → FiLM → TemporalBridge pipeline before BiLSTM.

        Processing order matters: positional encoding first (gives the
        LSTM an absolute-position signal), then FiLM modulation
        conditioned on country/format, then local temporal context
        via the residual Conv1d bridge.
        """
        x = self.pos_encoding(x)
        if self.context_film is not None:
            x = self.context_film(x, country_idx, format_idx)
        if self.temporal_bridge is not None:
            x = self.temporal_bridge(x)
        return x

    def _run_lstm_paths(
        self,
        compressed_std: Tensor | None,
        compressed_sq: Tensor | None,
        sq_mask: Tensor,
        country_idx: Tensor | None = None,
        format_idx: Tensor | None = None,
    ) -> Tensor:
        """Run LSTM on standard and square paths separately.

        Each path applies the same PE → FiLM → TemporalBridge → BiLSTM
        pipeline; see :meth:`_apply_pe_film_bridge` for details.
        """
        std_mask = ~sq_mask
        lstm_hidden = self.bilstm.hidden_size
        batch_size = sq_mask.shape[0]
        max_len, device = self._resolve_lstm_shape(
            compressed_std, compressed_sq, sq_mask
        )

        lstm_out = torch.zeros(
            batch_size, max_len, lstm_hidden * 2, device=device
        )
        if std_mask.any():
            assert compressed_std is not None
            x = self._apply_pe_film_bridge(
                compressed_std[std_mask],
                country_idx[std_mask],
                format_idx[std_mask],
            )
            lstm_out[std_mask, : compressed_std.shape[1], :] = self.bilstm(x)
        if sq_mask.any():
            assert compressed_sq is not None
            x = self._apply_pe_film_bridge(
                compressed_sq[sq_mask],
                country_idx[sq_mask],
                format_idx[sq_mask],
            )
            lstm_out[sq_mask] = self.bilstm(x)
        return lstm_out

    @staticmethod
    def _resolve_lstm_shape(
        compressed_std: Tensor | None,
        compressed_sq: Tensor | None,
        sq_mask: Tensor,
    ) -> tuple[int, torch.device]:
        """Determine max_len and device for LSTM output tensor."""
        if compressed_sq is not None and compressed_std is not None:
            return max(
                compressed_sq.shape[1], compressed_std.shape[1]
            ), compressed_sq.device
        if compressed_sq is not None:
            return compressed_sq.shape[1], compressed_sq.device
        if compressed_std is not None:
            return compressed_std.shape[1], compressed_std.device
        return _DEFAULT_LSTM_MAX_LEN, sq_mask.device

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Load state dict with backward-compatible FiLM & bridge handling.

        When ``enable_film=True`` but the checkpoint was saved without
        FiLM parameters, the missing ``context_film.*`` keys are
        silently ignored (they are zero-initialised anyway, so the
        model behaves as identity).  When ``enable_film=False``, any
        unexpected ``context_film.*`` keys in the checkpoint are
        dropped.

        The same logic applies to ``temporal_bridge.*`` keys: when
        ``enable_temporal_bridge=True`` and the checkpoint lacks them,
        they are ignored (residual connection means untrained bridge is
        near-identity).  When disabled, bridge keys from the checkpoint
        are dropped.
        """
        state_dict = self._patch_compat_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    def _patch_compat_keys(
        self,
        state_dict: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Patch FiLM / TemporalBridge keys for backward compatibility.

        When a module is enabled but checkpoint lacks its keys, inject
        the model's own default-initialised values so ``strict=True``
        succeeds (residual connections make untrained modules near-identity).

        When a module is disabled, drop its keys from the checkpoint so
        they don't appear as unexpected.
        """
        model_sd = self.state_dict()

        for prefix, enabled in (
            ("context_film.", self._enable_film),
            ("temporal_bridge.", self._enable_temporal_bridge),
            ("char_aux_head.context_film.", self._enable_char_aux_film),
        ):
            if enabled:
                # Inject defaults for any missing keys
                for k in model_sd:
                    if k.startswith(prefix) and k not in state_dict:
                        state_dict[k] = model_sd[k]
            else:
                # Drop module params from checkpoint
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith(prefix)
                }

        return state_dict

    def encode_countries(
        self,
        countries: list[str],
    ) -> torch.Tensor:
        """Encode country names to index tensor.

        Unknown countries raise ValueError.
        """
        labels = []
        for c in countries:
            if c not in self._country_to_idx:
                raise ValueError(
                    f"Unknown country '{c}' not in country_list. "
                    f"Allowed: {self.country_list}"
                )
            labels.append(self._country_to_idx[c])
        return torch.tensor(labels, dtype=torch.long)
