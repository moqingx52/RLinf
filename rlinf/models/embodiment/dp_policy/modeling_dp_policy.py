# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


def _load_dp_policy_from_ckpt_payload(ckpt_path: str, device: torch.device):
    """与 RoboTwin `policy.DP.dp_model.load_diffusion_policy_from_checkpoint` 等价，供旧版 dp_model 无该符号时使用。"""
    import dill
    import hydra
    from diffusion_policy.workspace.robotworkspace import RobotWorkspace

    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(torch.device(str(device)))
    policy.eval()
    return policy


def _load_robotwin_dp_expert(ckpt_path: str, device: torch.device):
    try:
        from policy.DP.dp_model import load_diffusion_policy_from_checkpoint

        return load_diffusion_policy_from_checkpoint(
            ckpt_path, map_location=str(device)
        )
    except ImportError:
        pass
    try:
        return _load_dp_policy_from_ckpt_payload(ckpt_path, device)
    except Exception as e:
        raise ImportError(
            "无法从 RoboTwin DP checkpoint 加载策略：请确认 RoboTwin 在 PYTHONPATH 上，且已安装 "
            "`diffusion_policy`；或将本仓库中的 `policy/DP/dp_model.py`（含 "
            "`load_diffusion_policy_from_checkpoint`）同步到服务器。"
        ) from e


def _rlinf_to_dp_timestep(main_images, states):
    try:
        from policy.DP.rlinf_adapter import rlinf_main_state_to_dp_timestep
    except ImportError as e:
        raise ImportError(
            "无法导入 RoboTwin `policy.DP.rlinf_adapter`。请将 RoboTwin 仓库根目录加入 PYTHONPATH。"
        ) from e
    return rlinf_main_state_to_dp_timestep(main_images, states)


@dataclass
class DpPolicyConfig:
    ckpt_path: str = ""
    n_obs_steps: int = 3
    num_action_chunks: int = 6
    action_dim: int = 14
    use_dsrl: bool = False
    dsrl_state_dim: int = 14
    dsrl_image_latent_dim: int = 64
    dsrl_state_latent_dim: int = 64
    dsrl_hidden_dims: tuple = field(default_factory=lambda: (128, 128, 128))
    dsrl_num_q_heads: int = 10


class DpPolicyForRL(nn.Module, BasePolicy):
    """Wraps RoboTwin Diffusion Policy for RLinf rollout / DSRL (frozen expert)."""

    def __init__(self, cfg: DpPolicyConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        if not cfg.ckpt_path:
            raise ValueError("dp_policy 需要在配置中设置 ckpt_path（RoboTwin DP .ckpt）。")

        expert = _load_robotwin_dp_expert(cfg.ckpt_path, device)
        self.add_module("dp_expert", expert)
        for p in self.dp_expert.parameters():
            p.requires_grad = False

        self.n_obs_steps = int(getattr(expert, "n_obs_steps", cfg.n_obs_steps))
        self.diffusion_horizon = int(expert.horizon)
        self.diffusion_action_dim = int(expert.action_dim)
        self._flat_noise_dim = self.diffusion_horizon * self.diffusion_action_dim

        if cfg.num_action_chunks != expert.n_action_steps:
            # 允许覆盖但提醒对齐
            pass
        self.exec_action_steps = int(expert.n_action_steps)

        self._hist_head: Optional[torch.Tensor] = None
        self._hist_state: Optional[torch.Tensor] = None

        self.use_dsrl = cfg.use_dsrl
        if self.use_dsrl:
            from rlinf.models.embodiment.modules.compact_encoders import (
                CompactMultiQHead,
                CompactStateEncoder,
                LightweightImageEncoder64,
            )
            from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy

            dsrl_in = cfg.dsrl_state_latent_dim + cfg.dsrl_image_latent_dim
            self.dsrl_action_noise_net = GaussianPolicy(
                input_dim=dsrl_in,
                output_dim=self._flat_noise_dim,
                hidden_dims=cfg.dsrl_hidden_dims,
                low=None,
                high=None,
                action_horizon=1,
            )
            self.actor_image_encoder = LightweightImageEncoder64(
                num_images=1,
                latent_dim=cfg.dsrl_image_latent_dim,
                image_size=64,
            )
            self.actor_state_encoder = CompactStateEncoder(
                state_dim=cfg.dsrl_state_dim,
                hidden_dim=cfg.dsrl_state_latent_dim,
            )
            self.critic_image_encoder = LightweightImageEncoder64(
                num_images=1,
                latent_dim=cfg.dsrl_image_latent_dim,
                image_size=64,
            )
            self.critic_state_encoder = CompactStateEncoder(
                state_dim=cfg.dsrl_state_dim,
                hidden_dim=cfg.dsrl_state_latent_dim,
            )
            self.q_head = CompactMultiQHead(
                state_dim=cfg.dsrl_state_latent_dim,
                image_dim=cfg.dsrl_image_latent_dim,
                action_dim=self._flat_noise_dim,
                hidden_dims=cfg.dsrl_hidden_dims,
                num_q_heads=cfg.dsrl_num_q_heads,
                output_dim=1,
            )

    def reset_obs_history(self):
        self._hist_head = None
        self._hist_state = None

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"{forward_type=}")

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "dp_policy 当前不支持 PPO/default 训练前向；请使用 only_eval 或 embodied_sac + use_dsrl。"
        )

    def _preprocess_dsrl_images(self, images, train: bool = False):
        import torch.nn.functional as F

        if isinstance(images, list):
            agentview_img = images[0]
        else:
            agentview_img = images
        if agentview_img.shape[-1] == 3:
            agentview_img = agentview_img.permute(0, 3, 1, 2)
        if agentview_img.dtype == torch.uint8:
            agentview_img = agentview_img.float() / 255.0
        elif agentview_img.min() < 0:
            agentview_img = (agentview_img + 1.0) / 2.0
        agentview_img = agentview_img.clamp(0.0, 1.0)
        resized = F.interpolate(
            agentview_img,
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        resized = resized * 2.0 - 1.0
        return resized.unsqueeze(1)

    def _preprocess_states(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() > 2:
            states = states.reshape(states.shape[0], -1)
        return states

    def sac_forward(self, obs=None, data=None, train=False, **kwargs):
        if not self.use_dsrl:
            raise ValueError("sac_forward 仅在 use_dsrl=True 时可用")
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})
        if "images" not in obs:
            if "main_images" in obs:
                obs = {"images": [obs["main_images"]], "states": obs["states"]}
            else:
                raise ValueError(f"无效的 obs 键: {obs.keys()}")

        mode = kwargs.get("mode", "train")
        deterministic = mode == "eval"

        images = self._preprocess_dsrl_images(obs["images"], train=train)
        states = self._preprocess_states(obs["states"])
        dev = next(self.actor_image_encoder.parameters()).device
        images = images.to(device=dev, dtype=torch.float32)
        states = states.to(device=dev, dtype=torch.float32)

        img_f = self.actor_image_encoder(images)
        st_f = self.actor_state_encoder(states)
        feat = torch.cat([st_f, img_f], dim=-1)
        action_noise, logprobs = self.dsrl_action_noise_net.sample(
            feat, deterministic=deterministic
        )
        return action_noise, logprobs, None

    def sac_q_forward(
        self,
        obs=None,
        data=None,
        actions=None,
        detach_encoder: bool = False,
        train: bool = False,
        **kwargs,
    ):
        if not self.use_dsrl:
            raise ValueError("sac_q_forward 仅在 use_dsrl=True 时可用")
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})
        if actions is None:
            actions = kwargs.get("actions")
        if "images" not in obs:
            if "main_images" in obs:
                obs = {"images": [obs["main_images"]], "states": obs["states"]}
            else:
                raise ValueError(f"无效的 obs 键: {obs.keys()}")

        images = self._preprocess_dsrl_images(obs["images"], train=train)
        states = self._preprocess_states(obs["states"])
        dev = next(self.critic_image_encoder.parameters()).device
        images = images.to(device=dev, dtype=torch.float32)
        states = states.to(device=dev, dtype=torch.float32)
        actions = actions.to(device=dev, dtype=torch.float32)

        img_f = self.critic_image_encoder(images)
        st_f = self.critic_state_encoder(states)
        if detach_encoder:
            img_f = img_f.detach()
            st_f = st_f.detach()
        if actions.dim() == 3:
            actions = actions[:, 0, :]
        return self.q_head(st_f, img_f, actions)

    def _update_history(self, head_bchw: torch.Tensor, state_bd: torch.Tensor):
        B, dev, dt = head_bchw.shape[0], head_bchw.device, head_bchw.dtype
        T = self.n_obs_steps
        if self._hist_head is None or self._hist_head.shape[0] != B:
            self._hist_head = head_bchw.unsqueeze(1).expand(B, T, -1, -1, -1).clone()
            self._hist_state = state_bd.unsqueeze(1).expand(B, T, -1).clone()
        else:
            self._hist_head = torch.roll(self._hist_head, shifts=-1, dims=1)
            self._hist_state = torch.roll(self._hist_state, shifts=-1, dims=1)
            self._hist_head = self._hist_head.to(device=dev, dtype=dt)
            self._hist_state = self._hist_state.to(device=dev, dtype=state_bd.dtype)
            self._hist_head[:, -1] = head_bchw
            self._hist_state[:, -1] = state_bd

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        mode: str = "train",
        calculate_values: bool = True,
        compute_values: bool = True,
        return_obs: bool = True,
        **kwargs,
    ):
        del kwargs  # 预留
        main = env_obs["main_images"]
        states = env_obs["states"]
        step = _rlinf_to_dp_timestep(main, states)
        expert_dev = next(self.dp_expert.parameters()).device
        exp_dtype = next(self.dp_expert.parameters()).dtype

        head = step["head_cam"].to(device=expert_dev, dtype=torch.float32)
        st = step["agent_pos"].to(device=expert_dev, dtype=torch.float32)
        self._update_history(head, st)

        obs_dict = {
            "head_cam": self._hist_head.to(dtype=exp_dtype),
            "agent_pos": self._hist_state.to(dtype=exp_dtype),
        }

        init_noise = None
        noise_flat = None
        noise_logprob = None
        if self.use_dsrl:
            sac_obs = {"main_images": main, "states": states}
            noise_3, noise_logprob, _ = self.sac_forward(
                obs=sac_obs, train=False, mode=mode
            )
            noise_flat = noise_3.squeeze(1)
            init_noise = noise_flat.view(
                noise_flat.shape[0],
                self.diffusion_horizon,
                self.diffusion_action_dim,
            ).to(dtype=exp_dtype, device=expert_dev)

        out = self.dp_expert.predict_action(obs_dict, init_noise=init_noise)
        actions = out["action"]
        chunk_actions = actions.reshape(
            actions.shape[0], self.exec_action_steps, self.diffusion_action_dim
        )

        if self.use_dsrl and noise_logprob is not None:
            prev_logprobs = noise_logprob.unsqueeze(-1).unsqueeze(-1).expand_as(
                chunk_actions
            )
        else:
            prev_logprobs = torch.zeros_like(chunk_actions)

        calc_v = calculate_values or compute_values
        if calc_v:
            prev_values = torch.zeros(
                chunk_actions.shape[0],
                chunk_actions.shape[1],
                1,
                device=chunk_actions.device,
                dtype=chunk_actions.dtype,
            )
        else:
            prev_values = torch.zeros_like(chunk_actions[..., :1])

        forward_inputs = {
            "action": actions.reshape(actions.shape[0], -1).contiguous(),
            "main_images": main,
            "states": states,
        }
        if self.use_dsrl and noise_flat is not None:
            forward_inputs["action"] = noise_flat.contiguous()

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result


def get_model(cfg: DictConfig, torch_dtype=torch.float32):
    del torch_dtype  # DP expert 保持 float32；可训练 DSRL 头同为 float32
    device = torch.device("cpu")
    raw = OmegaConf.to_container(cfg, resolve=True)
    mcfg = DpPolicyConfig()
    skip = {"model_type", "precision", "is_lora", "lora_rank", "lora_path"}
    nested = raw.get("dp_policy")
    if isinstance(nested, dict):
        for k, v in nested.items():
            if hasattr(mcfg, k):
                setattr(mcfg, k, v)
    for k, v in raw.items():
        if k in skip or k == "dp_policy":
            continue
        if hasattr(mcfg, k):
            setattr(mcfg, k, v)
    if isinstance(mcfg.dsrl_hidden_dims, list):
        mcfg.dsrl_hidden_dims = tuple(mcfg.dsrl_hidden_dims)
    ck = raw.get("ckpt_path") or raw.get("model_path")
    if ck:
        mcfg.ckpt_path = ck
    model = DpPolicyForRL(mcfg, device=device)
    return model
