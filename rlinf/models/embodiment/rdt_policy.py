from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.rdt_peft import RdtLoraSpec, freeze_module, inject_lora


@dataclass
class RdtPolicyConfig:
    ckpt_path: str = ""
    config_path: str = ""
    vision_encoder_path: str = ""
    text_encoder_path: str = ""
    precision: str = "bf16"
    control_frequency: int = 25

    num_action_chunks: int = 8
    action_dim: int = 14
    n_obs_steps: int = 2

    use_precomputed_lang_embed: bool = True
    freeze_vision_encoder: bool = True
    freeze_text_encoder: bool = True
    freeze_rdt_backbone: bool = False
    freeze_final_layer: bool = False
    train_lora_only: bool = True

    lora_enable: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("attn", "cross_attn", "qkv", "proj", "q", "kv")


class RdtPolicyForRLinf(nn.Module, BasePolicy):
    def __init__(self, cfg: RdtPolicyConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = torch.bfloat16 if "bf16" in str(cfg.precision).lower() else torch.float32

        self.rdt_model, self.text_tokenizer, self.text_encoder = self._build_rdt_bundle(cfg)
        self.exec_action_steps = int(cfg.num_action_chunks)
        self.action_dim = int(cfg.action_dim)
        self.n_obs_steps = int(cfg.n_obs_steps)
        self._hist_main: Optional[torch.Tensor] = None
        self._hist_state: Optional[torch.Tensor] = None
        self._hist_ready: Optional[torch.Tensor] = None
        self._text_embed_cache: dict[str, torch.Tensor] = {}

        self._apply_freeze_policy()
        self._maybe_inject_lora()

    def _build_rdt_bundle(self, cfg: RdtPolicyConfig):
        try:
            from policy.RDT.multimodal_encoder.t5_encoder import T5Embedder
            from policy.RDT.scripts.agilex_model import create_model
        except ImportError as err:
            raise ImportError(
                "未找到 RoboTwin RDT 依赖，请将 RoboTwin 根目录加入 PYTHONPATH。"
            ) from err

        if not cfg.ckpt_path:
            raise ValueError("rdt_lora 需要配置 ckpt_path。")
        if not cfg.config_path:
            raise ValueError("rdt_lora 需要配置 config_path。")

        import yaml

        with open(cfg.config_path, "r", encoding="utf-8") as fp:
            model_cfg = yaml.safe_load(fp)
        model_cfg["arm_dim"] = {"left_arm_dim": 6, "right_arm_dim": 6}

        model = create_model(
            args=model_cfg,
            dtype=self.dtype,
            pretrained=cfg.ckpt_path,
            pretrained_vision_encoder_name_or_path=cfg.vision_encoder_path or None,
            control_frequency=int(cfg.control_frequency),
            device=str(self.device),
        )
        model.policy.eval()
        model.vision_model.eval()
        model.policy = model.policy.to(device=self.device, dtype=self.dtype)
        model.vision_model = model.vision_model.to(device=self.device, dtype=self.dtype)

        if cfg.text_encoder_path:
            text_embedder = T5Embedder(
                from_pretrained=cfg.text_encoder_path,
                model_max_length=model_cfg["dataset"]["tokenizer_max_length"],
                device=self.device,
                use_offload_folder=None,
            )
            tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
            text_encoder.eval()
        else:
            tokenizer, text_encoder = None, None

        return model, tokenizer, text_encoder

    def _apply_freeze_policy(self) -> None:
        if self.cfg.freeze_vision_encoder:
            freeze_module(self.rdt_model.vision_model)
        if self.cfg.freeze_text_encoder and self.text_encoder is not None:
            freeze_module(self.text_encoder)
        if self.cfg.freeze_rdt_backbone:
            freeze_module(self.rdt_model.policy)

    def _maybe_inject_lora(self) -> None:
        if not self.cfg.lora_enable:
            return
        spec = RdtLoraSpec(
            rank=int(self.cfg.lora_rank),
            alpha=int(self.cfg.lora_alpha),
            dropout=float(self.cfg.lora_dropout),
            target_modules=tuple(self.cfg.lora_target_modules),
        )
        self.rdt_model.policy = inject_lora(self.rdt_model.policy, spec)

    def _encode_instruction(self, instruction: str) -> torch.Tensor:
        instruction = (instruction or "").strip() or "Place the empty cup to the target area."
        if instruction in self._text_embed_cache:
            return self._text_embed_cache[instruction]
        if self.text_tokenizer is None or self.text_encoder is None:
            raise ValueError("未提供 text_encoder_path，无法在线编码 instruction。")
        with torch.no_grad():
            tokens = self.text_tokenizer(
                instruction,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )["input_ids"].to(self.device)
            embed = self.text_encoder(tokens).last_hidden_state.detach()
        self._text_embed_cache[instruction] = embed
        return embed

    def _select_text_embed(
        self,
        env_obs: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        if self.cfg.use_precomputed_lang_embed:
            embed = env_obs.get("lang_embed", None)
            if embed is None:
                embed = env_obs.get("lang_embeds", None)
            if embed is not None:
                if isinstance(embed, torch.Tensor):
                    sample = embed[batch_idx]
                else:
                    sample = torch.as_tensor(embed[batch_idx])
                if sample.dim() == 2:
                    return sample.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                if sample.dim() == 1:
                    return sample.unsqueeze(0).unsqueeze(0).to(
                        device=self.device, dtype=self.dtype
                    )

        task_descs = env_obs.get("task_descriptions", None)
        instruction = ""
        if task_descs is not None:
            instruction = str(task_descs[batch_idx])
        return self._encode_instruction(instruction).to(device=self.device, dtype=self.dtype)

    def _update_history(self, main_bhwc: torch.Tensor, state_bd: torch.Tensor) -> None:
        bsz = int(main_bhwc.shape[0])
        if self._hist_main is None or self._hist_main.shape[0] != bsz:
            self._hist_main = main_bhwc.unsqueeze(1).expand(bsz, self.n_obs_steps, -1, -1, -1).clone()
            self._hist_state = state_bd.unsqueeze(1).expand(bsz, self.n_obs_steps, -1).clone()
            self._hist_ready = torch.ones(bsz, dtype=torch.bool, device=main_bhwc.device)
            return
        if self._hist_ready is not None and (~self._hist_ready).any():
            cold = ~self._hist_ready
            self._hist_main[cold] = main_bhwc[cold].unsqueeze(1).expand(-1, self.n_obs_steps, -1, -1, -1)
            self._hist_state[cold] = state_bd[cold].unsqueeze(1).expand(-1, self.n_obs_steps, -1)
            self._hist_ready[cold] = True
        self._hist_main = torch.roll(self._hist_main, shifts=-1, dims=1)
        self._hist_state = torch.roll(self._hist_state, shifts=-1, dims=1)
        self._hist_main[:, -1] = main_bhwc
        self._hist_state[:, -1] = state_bd

    def reset_obs_history(self, env_idx=None, env_mask=None) -> None:
        if env_idx is None and env_mask is None:
            self._hist_main = None
            self._hist_state = None
            self._hist_ready = None
            return
        if self._hist_ready is None:
            return
        mask = torch.zeros(self._hist_ready.shape[0], dtype=torch.bool, device=self._hist_ready.device)
        if env_mask is not None:
            mask |= torch.as_tensor(env_mask, dtype=torch.bool, device=mask.device)
        if env_idx is not None:
            for idx in env_idx:
                mask[int(idx)] = True
        self._hist_ready[mask] = False

    def _predict_chunk(self, env_obs: dict[str, Any]) -> torch.Tensor:
        main = env_obs["main_images"].to(device=self.device)
        state = env_obs["states"].to(device=self.device, dtype=torch.float32)
        self._update_history(main, state)

        out: list[torch.Tensor] = []
        for i in range(main.shape[0]):
            prev_idx = -2 if self.n_obs_steps >= 2 else -1
            prev = self._hist_main[i, prev_idx].detach().cpu().numpy()
            curr = self._hist_main[i, -1].detach().cpu().numpy()
            imgs = [
                Image.fromarray(prev),
                Image.fromarray(prev),
                Image.fromarray(prev),
                Image.fromarray(curr),
                Image.fromarray(curr),
                Image.fromarray(curr),
            ]
            proprio = self._hist_state[i : i + 1, -1].to(dtype=torch.float32)
            text_embed = self._select_text_embed(env_obs, i)
            pred = self.rdt_model.step(proprio=proprio, images=imgs, text_embeds=text_embed)
            out.append(pred[:, : self.exec_action_steps, : self.action_dim].to(torch.float32))
        return torch.cat(out, dim=0)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"rdt_lora 不支持 {forward_type=}")

    def default_forward(self, data=None, **kwargs):
        payload = data or kwargs
        obs = payload.get("observation", payload.get("obs", None))
        actions = payload.get("actions", payload.get("action", None))
        if obs is None or actions is None:
            raise ValueError("rdt_lora default_forward 需要 observation 与 actions。")

        pred = self._predict_chunk(obs)
        if actions.dim() == 2:
            target = actions.view(actions.shape[0], self.exec_action_steps, self.action_dim)
        else:
            target = actions[..., : self.action_dim]
            target = target[:, : self.exec_action_steps, :]
        target = target.to(device=pred.device, dtype=pred.dtype)
        return torch.nn.functional.mse_loss(pred, target)

    @torch.inference_mode()
    def predict_action_batch(self, env_obs, mode: str = "train", return_obs: bool = True, **kwargs):
        del mode, return_obs, kwargs
        chunk_actions = self._predict_chunk(env_obs)
        prev_logprobs = torch.zeros_like(chunk_actions)
        prev_values = torch.zeros(
            chunk_actions.shape[0],
            chunk_actions.shape[1],
            1,
            device=chunk_actions.device,
            dtype=chunk_actions.dtype,
        )
        forward_inputs = {
            "action": chunk_actions.reshape(chunk_actions.shape[0], -1).contiguous(),
            "main_images": env_obs.get("main_images"),
            "states": env_obs.get("states"),
        }
        return chunk_actions, {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
