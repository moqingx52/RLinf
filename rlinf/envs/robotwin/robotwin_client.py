# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TCP client for RoboTwin VectorEnv running in a separate process (port bridge)."""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import uuid
from typing import Any, Optional

_logger = logging.getLogger(__name__)

import numpy as np

__all__ = [
    "API_VERSION",
    "ClientVectorEnv",
    "RemoteEnvError",
    "parse_server_addr",
]

API_VERSION = 1


class RemoteEnvError(RuntimeError):
    """Structured error from ``robotwin_env_server``."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "RemoteError",
        can_retry: bool = False,
        should_recreate_env: bool = False,
        request_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.can_retry = can_retry
        self.should_recreate_env = should_recreate_env
        self.request_id = request_id


def parse_server_addr(addr: str) -> tuple[str, int]:
    s = addr.strip()
    if s.startswith("tcp://"):
        s = s[6:]
    if ":" not in s:
        raise ValueError(f"Invalid server_addr (expected host:port or tcp://host:port): {addr!r}")
    host, _, port_s = s.rpartition(":")
    host = host.strip() or "127.0.0.1"
    return host, int(port_s)


def _recvn(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed while reading")
        buf += chunk
    return buf


def _send_frame(sock: socket.socket, meta: dict[str, Any], blob: bytes = b"") -> None:
    body = json.dumps(meta, separators=(",", ":"), default=str).encode("utf-8")
    sock.sendall(struct.pack(">I", len(body)) + body + struct.pack(">I", len(blob)) + blob)


def _recv_frame(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    (lb,) = struct.unpack(">I", _recvn(sock, 4))
    body = _recvn(sock, lb)
    (lb2,) = struct.unpack(">I", _recvn(sock, 4))
    blob = _recvn(sock, lb2) if lb2 else b""
    return json.loads(body.decode("utf-8")), blob


def _deserialize_obs_list(meta: dict, blob: bytes) -> list[dict]:
    out: list[dict] = []
    for o in meta["obs"]:
        obs: dict[str, Any] = {"instruction": o["instruction"]}
        for k in ("full_image", "left_wrist_image", "right_wrist_image", "state"):
            d = o.get(k)
            if d is None:
                obs[k] = None
            else:
                dt = np.dtype(d["dtype"])
                raw = blob[d["start"] : d["start"] + d["len"]]
                obs[k] = np.frombuffer(raw, dtype=dt).reshape(tuple(d["shape"]))
        out.append(obs)
    return out


def _meta_error(meta: dict[str, Any], req_id: str) -> RemoteEnvError:
    msg = meta.get("error_message") or meta.get("error") or "remote error"
    return RemoteEnvError(
        str(msg),
        error_type=str(meta.get("error_type", "RemoteError")),
        can_retry=bool(meta.get("can_retry", False)),
        should_recreate_env=bool(meta.get("should_recreate_env", False)),
        request_id=meta.get("request_id", req_id),
    )


class ClientVectorEnv:
    """Drop-in replacement for ``robotwin.envs.vector_env.VectorEnv`` over TCP."""

    def __init__(
        self,
        host: str,
        port: int,
        n_envs: int,
        env_seeds: Optional[list[int]],
        request_timeout: float = 180.0,
        heartbeat_interval: float = 20.0,
        task_config: Optional[dict[str, Any]] = None,
    ):
        self.n_envs = n_envs
        self.env_seeds = list(env_seeds) if env_seeds is not None else None
        self._task_config = task_config
        self._host = host
        self._port = port
        self._timeout = float(request_timeout)
        self._heartbeat_interval = float(heartbeat_interval)
        self._sock: Optional[socket.socket] = None
        self._call_lock = threading.Lock()
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        self._last_reset_seeds: Optional[list[int]] = None
        self.remote_obs_schema: Optional[dict[str, Any]] = None

        _logger.info(
            "ClientVectorEnv: connecting TCP %s:%s (n_envs=%s, op_timeout=%ss)",
            host,
            port,
            n_envs,
            self._timeout,
        )
        self._connect()
        _logger.info(
            "ClientVectorEnv: TCP connected, sending remote init (n_envs=%s)...",
            n_envs,
        )
        self._call_init()
        _logger.info("ClientVectorEnv: remote init RPC finished")
        if self._heartbeat_interval > 0:
            self._hb_stop.clear()
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._hb_thread.start()

    def _stop_heartbeat(self) -> None:
        self._hb_stop.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=2.0)
            self._hb_thread = None

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.wait(timeout=self._heartbeat_interval):
            if self._sock is None:
                continue
            with self._call_lock:
                if self._sock is None or self._hb_stop.is_set():
                    continue
                try:
                    self._call_unlocked("heartbeat")
                except (OSError, RemoteEnvError, TimeoutError, json.JSONDecodeError):
                    pass

    def _connect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        s = socket.create_connection((self._host, self._port), timeout=self._timeout)
        s.settimeout(self._timeout)
        self._sock = s

    def _call_unlocked(self, op: str, **kwargs: Any) -> dict[str, Any]:
        assert self._sock is not None
        req_id = str(uuid.uuid4())
        req: dict[str, Any] = {"api_version": API_VERSION, "request_id": req_id, "op": op, **kwargs}
        _send_frame(self._sock, req)
        meta, blob = _recv_frame(self._sock)
        rid = meta.get("request_id")
        if rid is not None and rid != req_id:
            raise RemoteEnvError(
                f"request_id mismatch: expected {req_id}, got {rid}",
                error_type="ProtocolError",
                request_id=req_id,
            )
        ver = meta.get("api_version", API_VERSION)
        if ver != API_VERSION:
            raise RemoteEnvError(
                f"unsupported api_version in response: {ver}",
                error_type="ProtocolError",
                request_id=req_id,
            )
        if not meta.get("ok", False):
            raise _meta_error(meta, req_id)
        if blob and "obs_meta" in meta:
            meta["obs"] = _deserialize_obs_list(meta["obs_meta"], blob)
            del meta["obs_meta"]
        return meta

    def _call(self, op: str, **kwargs: Any) -> dict[str, Any]:
        with self._call_lock:
            return self._call_unlocked(op, **kwargs)

    def _call_init(self) -> None:
        with self._call_lock:
            init_kw: dict[str, Any] = {
                "n_envs": self.n_envs,
                "env_seeds": self.env_seeds,
            }
            if self._task_config is not None:
                init_kw["task_config"] = self._task_config
            meta = self._call_unlocked("init", **init_kw)
        self.remote_obs_schema = meta.get("obs_schema")

    def _remote_close_socket_only(self) -> None:
        self._stop_heartbeat()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _recreate_remote_full(self, replay_reset: Optional[dict[str, Any]] = None) -> None:
        """Reconnect, init, then reset (replay partial reset payload when provided)."""
        self._remote_close_socket_only()
        self._connect()
        self._call_init()
        if self._heartbeat_interval > 0:
            self._hb_stop.clear()
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._hb_thread.start()
        with self._call_lock:
            if replay_reset:
                self._call_unlocked("reset", **replay_reset)
            elif self._last_reset_seeds is not None:
                self._call_unlocked("reset", env_seeds=self._last_reset_seeds)
            elif self.env_seeds is not None:
                self._call_unlocked("reset", env_seeds=list(self.env_seeds))
            else:
                self._call_unlocked("reset")

    def _run_step_with_recovery(self, actions_list: list) -> tuple:
        """Do not auto-recreate after ``step`` (would replay actions on a fresh env)."""
        try:
            meta = self._call("step", actions=actions_list)
        except RemoteEnvError as e:
            if e.should_recreate_env:
                raise
            if e.can_retry:
                meta = self._call("step", actions=actions_list)
            else:
                raise
        obs_list = meta["obs"]
        return (
            obs_list,
            meta["rewards"],
            meta["terminated"],
            meta["truncated"],
            meta["infos"],
        )

    def reset(self, env_idx=None, env_seeds=None):
        payload: dict[str, Any] = {}
        if env_idx is not None:
            if hasattr(env_idx, "tolist"):
                payload["env_idx"] = env_idx.tolist()
            elif isinstance(env_idx, (list, tuple)):
                payload["env_idx"] = list(env_idx)
            else:
                payload["env_idx"] = [int(env_idx)]
        if env_seeds is not None:
            payload["env_seeds"] = [int(x) for x in env_seeds]
            self._last_reset_seeds = list(payload["env_seeds"])
        try:
            self._call("reset", **payload)
        except RemoteEnvError as e:
            if e.should_recreate_env:
                self._recreate_remote_full(
                    replay_reset=payload if payload else None,
                )
            elif e.can_retry:
                self._call("reset", **payload)
            else:
                raise

    def step(self, actions: np.ndarray):
        actions_list = np.asarray(actions, dtype=np.float64).tolist()
        return self._run_step_with_recovery(actions_list)

    def get_obs(self):
        try:
            meta = self._call("get_obs")
        except RemoteEnvError as e:
            if e.should_recreate_env:
                raise
            if e.can_retry:
                meta = self._call("get_obs")
            else:
                raise
        return meta["obs"]

    def close(self, clear_cache=True):
        self._stop_heartbeat()
        if self._sock is not None:
            try:
                self._call("close", clear_cache=bool(clear_cache))
            except (RemoteEnvError, OSError, TimeoutError, json.JSONDecodeError, ConnectionError):
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def check_seeds(self, seeds: list[int]):
        meta = self._call("check_seeds", seeds=[int(s) for s in seeds])
        return meta["results"]
