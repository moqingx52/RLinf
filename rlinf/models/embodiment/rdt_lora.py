from __future__ import annotations

import os

import torch

from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.rdt_policy import RdtPolicyConfig, RdtPolicyForRLinf


def _pick(raw: dict, key: str, default):
    return raw.get(key, default)


def get_model(cfg: DictConfig, torch_dtype=None):
    del torch_dtype
    raw = OmegaConf.to_container(cfg, resolve=True)
    nested = raw.get("rdt_lora", {}) if isinstance(raw.get("rdt_lora", {}), dict) else {}

    c = RdtPolicyConfig()
    c.ckpt_path = _pick(raw, "ckpt_path", _pick(nested, "ckpt_path", c.ckpt_path))
    c.config_path = _pick(
        raw,
        "config_path",
        _pick(
            nested,
            "config_path",
            os.environ.get("ROBOTWIN_RDT_CONFIG", ""),
        ),
    )
    c.vision_encoder_path = _pick(
        raw,
        "vision_encoder_path",
        _pick(nested, "vision_encoder_path", os.environ.get("ROBOTWIN_RDT_VISION_ENCODER", "")),
    )
    c.text_encoder_path = _pick(
        raw,
        "text_encoder_path",
        _pick(nested, "text_encoder_path", os.environ.get("ROBOTWIN_RDT_TEXT_ENCODER", "")),
    )
    c.precision = str(_pick(raw, "precision", _pick(nested, "precision", c.precision)))
    c.control_frequency = int(_pick(raw, "control_frequency", _pick(nested, "control_frequency", c.control_frequency)))
    c.num_action_chunks = int(_pick(raw, "num_action_chunks", _pick(nested, "num_action_chunks", c.num_action_chunks)))
    c.action_dim = int(_pick(raw, "action_dim", _pick(nested, "action_dim", c.action_dim)))
    c.n_obs_steps = int(_pick(raw, "n_obs_steps", _pick(nested, "n_obs_steps", c.n_obs_steps)))

    c.use_precomputed_lang_embed = bool(
        _pick(
            raw,
            "use_precomputed_lang_embed",
            _pick(nested, "use_precomputed_lang_embed", c.use_precomputed_lang_embed),
        )
    )
    c.freeze_vision_encoder = bool(
        _pick(raw, "freeze_vision_encoder", _pick(nested, "freeze_vision_encoder", c.freeze_vision_encoder))
    )
    c.freeze_text_encoder = bool(
        _pick(raw, "freeze_text_encoder", _pick(nested, "freeze_text_encoder", c.freeze_text_encoder))
    )
    c.freeze_rdt_backbone = bool(
        _pick(raw, "freeze_rdt_backbone", _pick(nested, "freeze_rdt_backbone", c.freeze_rdt_backbone))
    )
    c.freeze_final_layer = bool(
        _pick(raw, "freeze_final_layer", _pick(nested, "freeze_final_layer", c.freeze_final_layer))
    )
    c.train_lora_only = bool(_pick(raw, "train_lora_only", _pick(nested, "train_lora_only", c.train_lora_only)))

    c.lora_enable = bool(_pick(raw, "lora_enable", _pick(nested, "lora_enable", c.lora_enable)))
    c.lora_rank = int(_pick(raw, "lora_rank", _pick(nested, "lora_rank", c.lora_rank)))
    c.lora_alpha = int(_pick(raw, "lora_alpha", _pick(nested, "lora_alpha", c.lora_alpha)))
    c.lora_dropout = float(_pick(raw, "lora_dropout", _pick(nested, "lora_dropout", c.lora_dropout)))
    lora_targets = _pick(raw, "lora_target_modules", _pick(nested, "lora_target_modules", c.lora_target_modules))
    c.lora_target_modules = tuple(lora_targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return RdtPolicyForRLinf(c, device=device)
