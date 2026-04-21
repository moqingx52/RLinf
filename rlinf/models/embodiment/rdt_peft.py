from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn


@dataclass
class RdtLoraSpec:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("qkv", "proj", "q", "kv", "out_proj")


def _is_candidate_module(name: str, module: nn.Module, targets: Iterable[str]) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    lname = name.lower()
    return any(token.lower() in lname for token in targets)


def resolve_target_modules(module: nn.Module, targets: Iterable[str]) -> list[str]:
    hits: set[str] = set()
    for name, sub_module in module.named_modules():
        if _is_candidate_module(name, sub_module, targets):
            # PEFT supports suffix matching by leaf module name.
            hits.add(name.split(".")[-1])
    return sorted(hits)


def inject_lora(module: nn.Module, spec: RdtLoraSpec) -> nn.Module:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as err:
        raise ImportError("请先安装 peft 以启用 RDT LoRA 注入。") from err

    target_modules = resolve_target_modules(module, spec.target_modules)
    if not target_modules:
        raise ValueError(
            f"未在 RDT policy 中匹配到 LoRA 目标模块，targets={spec.target_modules}"
        )

    lora_cfg = LoraConfig(
        r=int(spec.rank),
        lora_alpha=int(spec.alpha),
        lora_dropout=float(spec.dropout),
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    return get_peft_model(module, lora_cfg)


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
