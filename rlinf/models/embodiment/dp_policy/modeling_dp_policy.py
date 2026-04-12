# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import numpy as np
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


class RemoteDpExpertModule(torch.nn.Module):
    """Frozen DP on ``robotwin_dp_server``; RLinf 进程不 import diffusers/peft。"""

    def __init__(self, server_addr: str, ckpt_path: str, torch_device: torch.device):
        super().__init__()
        self.register_parameter(
            "_placeholder",
            nn.Parameter(
                torch.zeros(1, device=torch_device, dtype=torch.float32),
                requires_grad=False,
            ),
        )
        from rlinf.models.embodiment.robotwin_dp_remote_client import DpRemoteClient

        self._client = DpRemoteClient(server_addr)
        meta = self._client.init(ckpt_path)
        self.n_obs_steps = int(meta["n_obs_steps"])
        self.horizon = int(meta["horizon"])
        self.action_dim = int(meta["action_dim"])
        self.n_action_steps = int(meta["n_action_steps"])

    def predict_action(self, obs_dict, init_noise=None):
        raise RuntimeError(
            "RemoteDpExpertModule 仅用于远程推理；请使用 DpPolicyForRL 的 remote 分支。"
        )

    def remote_predict(
        self,
        main_images: torch.Tensor,
        states: torch.Tensor,
        init_noise: Optional[torch.Tensor],
    ) -> torch.Tensor:
        main_np = main_images.detach().cpu().numpy()
        st_np = states.detach().cpu().numpy()
        noise_np = None
        if init_noise is not None:
            noise_np = init_noise.detach().cpu().numpy()
        out_np = self._client.predict(main_np, st_np, noise_np)
        return torch.from_numpy(np.ascontiguousarray(out_np)).to(
            device=main_images.device, dtype=torch.float32
        )

    def reset_remote_history(
        self,
        env_idx: Optional[Sequence[int]] = None,
        env_mask: Optional[Union[Sequence[bool], np.ndarray]] = None,
    ) -> None:
        self._client.reset_history(env_idx=env_idx, env_mask=env_mask)


@dataclass
class DpPolicyConfig:
    ckpt_path: str = ""
    expert_backend: str = "local"
    dp_server_addr: str = ""
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

        backend = (cfg.expert_backend or "local").strip().lower()
        self._expert_backend = backend
        if backend == "remote":
            addr = (cfg.dp_server_addr or "").strip() or os.environ.get(
                "ROBOTWIN_DP_SERVER_ADDR", ""
            ).strip()
            if not addr:
                raise ValueError(
                    "expert_backend=remote 时需要配置 dp_server_addr 或环境变量 ROBOTWIN_DP_SERVER_ADDR"
                )
            expert = RemoteDpExpertModule(addr, cfg.ckpt_path, device)
            self.add_module("dp_expert", expert)
        else:
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
        self._hist_ready: Optional[torch.Tensor] = None

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

    def reset_obs_history(
        self,
        env_idx: Optional[Sequence[int]] = None,
        env_mask: Optional[Any] = None,
    ):
        """Clear DP obs history. Remote: forwards to server (full clear or per-slot). Local: same semantics."""
        if self._expert_backend == "remote":
            if env_idx is None and env_mask is None:
                self.dp_expert.reset_remote_history()
            else:
                em: Optional[np.ndarray] = None
                if env_mask is not None:
                    if isinstance(env_mask, torch.Tensor):
                        em = env_mask.detach().cpu().numpy().astype(np.bool_)
                    else:
                        em = np.asarray(env_mask, dtype=np.bool_)
                self.dp_expert.reset_remote_history(env_idx=env_idx, env_mask=em)
            self._hist_head = None
            self._hist_state = None
            self._hist_ready = None
            return

        if env_idx is None and env_mask is None:
            self._hist_head = None
            self._hist_state = None
            self._hist_ready = None
            return
        if self._hist_head is None or self._hist_ready is None:
            return
        B = int(self._hist_head.shape[0])
        dev = self._hist_head.device
        mask = torch.zeros(B, dtype=torch.bool, device=dev)
        if env_mask is not None:
            mask = torch.as_tensor(env_mask, dtype=torch.bool, device=dev)
            if mask.numel() != B:
                raise ValueError(f"env_mask length {mask.numel()} != batch {B}")
        if env_idx is not None:
            for i in env_idx:
                ii = int(i)
                if ii < 0 or ii >= B:
                    raise ValueError(f"env_idx out of range: {ii} (B={B})")
                mask[ii] = True
        if mask.any():
            self._hist_ready = self._hist_ready.clone()
            self._hist_ready[mask] = False

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
            self._hist_ready = torch.ones(B, dtype=torch.bool, device=dev)
            return
        assert self._hist_ready is not None
        self._hist_head = self._hist_head.to(device=dev, dtype=dt)
        self._hist_state = self._hist_state.to(device=dev, dtype=state_bd.dtype)
        old_ready = self._hist_ready
        need_init = ~old_ready
        rolling = old_ready
        if need_init.any():
            self._hist_head = self._hist_head.clone()
            self._hist_state = self._hist_state.clone()
            self._hist_ready = old_ready.clone()
            self._hist_head[need_init] = (
                head_bchw[need_init].unsqueeze(1).expand(-1, T, -1, -1, -1).clone()
            )
            self._hist_state[need_init] = (
                state_bd[need_init].unsqueeze(1).expand(-1, T, -1).clone()
            )
            self._hist_ready[need_init] = True
        if rolling.any():
            self._hist_head[rolling] = torch.roll(self._hist_head[rolling], shifts=-1, dims=1)
            self._hist_state[rolling] = torch.roll(self._hist_state[rolling], shifts=-1, dims=1)
            self._hist_head[rolling, -1] = head_bchw[rolling]
            self._hist_state[rolling, -1] = state_bd[rolling]

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
            ).to(dtype=torch.float32, device=main.device)

        if self._expert_backend == "remote":
            chunk_actions = self.dp_expert.remote_predict(main, states, init_noise)
        else:
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

            in_noise = (
                init_noise.to(device=expert_dev, dtype=exp_dtype)
                if init_noise is not None
                else None
            )
            out = self.dp_expert.predict_action(obs_dict, init_noise=in_noise)
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

        flat_chunk = chunk_actions.reshape(chunk_actions.shape[0], -1).contiguous()
        forward_inputs = {
            "action": flat_chunk,
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
    nested = raw.get("dp_policy") or raw.get("robotwin_dp_dsrl")
    if isinstance(nested, dict):
        for k, v in nested.items():
            if hasattr(mcfg, k):
                setattr(mcfg, k, v)
    for k, v in raw.items():
        if k in skip or k in ("dp_policy", "robotwin_dp_dsrl"):
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
