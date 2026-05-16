#!/usr/bin/env python3
"""Run an end-to-end RoboTwin remote RDT smoke evaluation.

This bypasses RL training and executes:
RoboTwin env server observation -> RDT remote expert -> RoboTwin env server step.
It is intended to verify that the RDT checkpoint/server/env wiring still reaches
the expected task success rate before launching DSRL training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_remote_clients():
    # Avoid importing the top-level rlinf package here; its package initializer
    # pulls in heavyweight training dependencies, while this script only needs
    # the two lightweight TCP client modules.
    for name in ("rlinf", "rlinf.envs", "rlinf.envs.robotwin"):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = pkg
    robotwin_client = _load_module(
        "rlinf.envs.robotwin.robotwin_client",
        RLINF_ROOT / "rlinf" / "envs" / "robotwin" / "robotwin_client.py",
    )
    dp_remote_client = _load_module(
        "rlinf.envs.robotwin.dp_remote_client",
        RLINF_ROOT / "rlinf" / "envs" / "robotwin" / "dp_remote_client.py",
    )
    return (
        dp_remote_client.DpRemoteClient,
        robotwin_client.ClientVectorEnv,
        robotwin_client.parse_server_addr,
    )


DpRemoteClient, ClientVectorEnv, parse_server_addr = _load_remote_clients()


def _repo_default_config() -> Path:
    return (
        RLINF_ROOT
        / "examples"
        / "embodiment"
        / "config"
        / "env"
        / "robotwin_place_empty_cup.yaml"
    )


def _load_task_config(
    config_path: Path,
    *,
    task_name: str,
    step_lim: int | None,
) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    task_config = cfg.get("task_config") if isinstance(cfg.get("task_config"), dict) else cfg
    if not isinstance(task_config, dict):
        raise ValueError(f"{config_path} does not contain a yaml mapping")
    task_config = dict(task_config)
    task_config.setdefault("task_name", task_name)
    if step_lim is not None:
        task_config["step_lim"] = int(step_lim)
    elif "step_lim" not in task_config:
        task_config["step_lim"] = _default_step_limit(task_config["task_name"])
    return task_config


def _default_step_limit(task_name: str) -> int:
    for root in (
        os.environ.get("ROBOTWIN_ROOT"),
        str(RLINF_ROOT.parent / "RoboTwin"),
        "/workspace/RoboTwin",
    ):
        if not root:
            continue
        path = Path(root) / "task_config" / "_eval_step_limit.yml"
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                limits = yaml.safe_load(f) or {}
            if task_name in limits:
                return int(limits[task_name])
    return 500 if task_name == "place_empty_cup" else 1000


def _compact_info(info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "success",
        "reward_sum",
        "episode_return",
        "rdt_success_dataset_saved",
        "rdt_success_dataset_episode_dir",
        "xy_dist",
        "z_abs",
        "lift_height",
        "grasped",
        "chunk_len",
        "run_steps",
        "take_action_cnt",
        "step_lim",
        "reward_milestones",
        "reward_components",
    )
    return {k: info[k] for k in keys if k in info}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def _run_episode(
    *,
    env: ClientVectorEnv,
    expert: DpRemoteClient,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    env.reset(env_seeds=[seed])
    expert.reset_history()

    episode_return = 0.0
    success_once = False
    steps = 0
    instruction = ""
    last_info: dict[str, Any] = {}

    for step_i in range(max_steps):
        obs = env.get_obs()[0]
        main_image = obs.get("full_image")
        state = obs.get("state")
        instruction = str(obs.get("instruction") or instruction)
        if main_image is None:
            raise RuntimeError("RoboTwin observation does not contain full_image")
        if state is None:
            raise RuntimeError("RoboTwin observation does not contain state")
        left_wrist = obs.get("left_wrist_image")
        right_wrist = obs.get("right_wrist_image")

        predict_kwargs = {}
        if left_wrist is not None and right_wrist is not None:
            predict_kwargs["left_wrist_images"] = np.expand_dims(
                np.asarray(left_wrist), axis=0
            )
            predict_kwargs["right_wrist_images"] = np.expand_dims(
                np.asarray(right_wrist), axis=0
            )

        actions = expert.predict(
            np.expand_dims(np.asarray(main_image), axis=0),
            np.expand_dims(np.asarray(state, dtype=np.float32).reshape(-1), axis=0),
            task_description=instruction,
            **predict_kwargs,
        )
        _, rewards, terminated, truncated, infos = env.step(actions)
        reward = float(np.asarray(rewards, dtype=np.float32).reshape(-1)[0])
        term = bool(np.asarray(terminated, dtype=np.bool_).reshape(-1)[0])
        trunc = bool(np.asarray(truncated, dtype=np.bool_).reshape(-1)[0])
        last_info = dict(infos[0] if infos else {})
        success_once = success_once or bool(last_info.get("success", False)) or term
        episode_return += reward
        steps = step_i + 1

        if term or trunc:
            break

    return {
        "seed": int(seed),
        "success": bool(success_once),
        "return": float(episode_return),
        "steps": int(steps),
        "instruction": instruction,
        "last_info": _compact_info(last_info),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate remote RDT policy on RoboTwin place_empty_cup "
            "for a small fixed number of episodes."
        )
    )
    parser.add_argument(
        "--env-server-addr",
        default=os.environ.get("ROBOTWIN_SERVER_ADDR", "127.0.0.1:8765"),
    )
    parser.add_argument(
        "--rdt-server-addr",
        default=os.environ.get("ROBOTWIN_RDT_SERVER_ADDR", "127.0.0.1:8769"),
    )
    parser.add_argument(
        "--ckpt",
        default=os.environ.get("ROBOTWIN_RDT_CKPT", ""),
        help="RDT checkpoint directory. May be omitted if robotwin_rdt_server was started with --ckpt.",
    )
    parser.add_argument("--config", type=Path, default=_repo_default_config())
    parser.add_argument("--task-name", default="place_empty_cup")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--collect-success-dataset-dir",
        type=Path,
        default=None,
        help="When set, ask RoboTwin env server to write successful episodes as processed RDT HDF5 data.",
    )
    parser.add_argument(
        "--target-successes",
        type=int,
        default=None,
        help="Stop after this many successful collected episodes. Uses --episodes only when unset.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum evaluated policy episodes when collecting by target successes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Matches RoboTwin eval seed convention: start_seed=100000*(1+seed).",
    )
    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--step-lim", type=int, default=None)
    parser.add_argument(
        "--no-expert-check",
        action="store_true",
        help="Do not filter seeds through RoboTwin play_once success before policy eval.",
    )
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RLINF_ROOT / "logs" / "robotwin-rdt-smoke-eval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_config = _load_task_config(
        args.config,
        task_name=args.task_name,
        step_lim=args.step_lim,
    )
    if args.collect_success_dataset_dir is not None:
        collect_dir = args.collect_success_dataset_dir.resolve()
        collect_dir.mkdir(parents=True, exist_ok=True)
        task_config["rdt_success_dataset_dir"] = str(collect_dir)
        camera_cfg = dict(task_config.get("camera") or {})
        camera_cfg["collect_wrist_camera"] = True
        task_config["camera"] = camera_cfg
    max_steps = int(args.max_steps or task_config.get("step_lim", 200))
    start_seed = int(
        args.start_seed if args.start_seed is not None else 100000 * (1 + args.seed)
    )

    env_host, env_port = parse_server_addr(args.env_server_addr)
    env = None
    expert = None

    try:
        env = ClientVectorEnv(
            host=env_host,
            port=env_port,
            n_envs=1,
            env_seeds=[start_seed],
            request_timeout=args.request_timeout,
            heartbeat_interval=20.0,
            task_config=task_config,
        )
        expert = DpRemoteClient(
            args.rdt_server_addr,
            request_timeout=args.request_timeout,
        )
        dp_meta = expert.init(args.ckpt)
        print(
            "[rdt-smoke] "
            f"dp_meta={json.dumps(dp_meta, default=_json_default, ensure_ascii=False)}",
            flush=True,
        )
        print(
            f"[rdt-smoke] task={task_config.get('task_name')} episodes={args.episodes} "
            f"start_seed={start_seed} max_steps={max_steps} "
            f"collect_success_dataset_dir={task_config.get('rdt_success_dataset_dir', '')}",
            flush=True,
        )

        results: list[dict[str, Any]] = []
        seed_cursor = start_seed
        ep = 0
        successes = 0
        target_successes = args.target_successes
        max_attempts = int(args.max_attempts or args.episodes)
        if target_successes is not None and max_attempts < target_successes:
            raise ValueError("--max-attempts must be >= --target-successes")

        def _should_continue() -> bool:
            if target_successes is None:
                return ep < args.episodes
            return ep < max_attempts and successes < target_successes

        while _should_continue():
            seed = seed_cursor
            seed_cursor += 1
            if not args.no_expert_check:
                seed_check = env.check_seeds([seed])[0]
                if not seed_check.get("play_once_success", False):
                    print(
                        "[rdt-smoke] "
                        f"skip_seed={seed} expert_check={json.dumps(seed_check, default=_json_default)}",
                        flush=True,
                    )
                    continue
            result = _run_episode(env=env, expert=expert, seed=seed, max_steps=max_steps)
            results.append(result)
            ep += 1
            successes += int(bool(result["success"]))
            if target_successes is None:
                progress = f"{ep}/{args.episodes}"
            else:
                progress = f"{ep}/{max_attempts} successes={successes}/{target_successes}"
            print(
                "[rdt-smoke] "
                f"episode={progress} seed={seed} "
                f"success={int(result['success'])} return={result['return']:.4f} "
                f"steps={result['steps']} "
                "last_info="
                f"{json.dumps(result['last_info'], default=_json_default, ensure_ascii=False)}",
                flush=True,
            )

        successes = sum(1 for r in results if r["success"])
        episode_count = len(results)
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task": task_config.get("task_name"),
            "episodes": int(episode_count),
            "requested_episodes": int(args.episodes),
            "target_successes": args.target_successes,
            "max_attempts": args.max_attempts,
            "successes": int(successes),
            "success_rate": float(successes / max(1, episode_count)),
            "start_seed": int(start_seed),
            "max_steps": int(max_steps),
            "env_server_addr": args.env_server_addr,
            "rdt_server_addr": args.rdt_server_addr,
            "ckpt": args.ckpt,
            "dp_meta": dp_meta,
            "collect_success_dataset_dir": task_config.get(
                "rdt_success_dataset_dir",
                None,
            ),
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_path = args.output_dir / f"place_empty_cup_rdt_smoke_{stamp}.jsonl"
        summary_path = args.output_dir / f"place_empty_cup_rdt_smoke_{stamp}_summary.json"
        with open(result_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(
                    json.dumps(result, default=_json_default, ensure_ascii=False) + "\n"
                )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=_json_default, ensure_ascii=False)

        print(
            f"[rdt-smoke] success_rate={successes}/{episode_count} "
            f"({summary['success_rate'] * 100:.1f}%) "
            f"result_path={result_path} summary_path={summary_path}",
            flush=True,
        )
    finally:
        if expert is not None:
            expert.close()
        if env is not None:
            env.close(clear_cache=True)


if __name__ == "__main__":
    main()
