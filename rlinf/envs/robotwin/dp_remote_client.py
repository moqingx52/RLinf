# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TCP client for ``robotwin_dp_server.py`` (remote DP expert, no diffusers in RLinf)."""

from __future__ import annotations

import json
import socket
import struct
import uuid
from typing import Any, Optional

import numpy as np

from rlinf.envs.robotwin.robotwin_client import RemoteEnvError, parse_server_addr

__all__ = ["DpRemoteClient"]

API_VERSION = 1


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


def _meta_error(meta: dict[str, Any], req_id: str) -> RemoteEnvError:
    msg = meta.get("error_message") or meta.get("error") or "remote error"
    return RemoteEnvError(
        str(msg),
        error_type=str(meta.get("error_type", "RemoteError")),
        can_retry=bool(meta.get("can_retry", False)),
        should_recreate_env=bool(meta.get("should_recreate_env", False)),
        request_id=meta.get("request_id", req_id),
    )


class DpRemoteClient:
    """Stateful client matching one ``robotwin_dp_server`` session (history on server)."""

    def __init__(self, server_addr: str, request_timeout: float = 300.0):
        host, port = parse_server_addr(server_addr)
        self._host = host
        self._port = port
        self._timeout = float(request_timeout)
        self._sock: Optional[socket.socket] = None
        self._meta: dict[str, Any] = {}

    def close(self) -> None:
        if self._sock is not None:
            try:
                req_id = str(uuid.uuid4())
                _send_frame(
                    self._sock,
                    {"api_version": API_VERSION, "request_id": req_id, "op": "close"},
                )
                _recv_frame(self._sock)
            except (OSError, RemoteEnvError, json.JSONDecodeError, ConnectionError):
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _connect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        s = socket.create_connection((self._host, self._port), timeout=self._timeout)
        s.settimeout(self._timeout)
        self._sock = s

    def _call(self, op: str, blob: bytes = b"", **kwargs: Any) -> tuple[dict[str, Any], bytes]:
        if self._sock is None:
            self._connect()
        assert self._sock is not None
        req_id = str(uuid.uuid4())
        req: dict[str, Any] = {"api_version": API_VERSION, "request_id": req_id, "op": op, **kwargs}
        _send_frame(self._sock, req, blob)
        meta, rblob = _recv_frame(self._sock)
        rid = meta.get("request_id")
        if rid is not None and rid != req_id:
            raise RemoteEnvError(
                f"request_id mismatch: expected {req_id}, got {rid}",
                error_type="ProtocolError",
                request_id=req_id,
            )
        if int(meta.get("api_version", API_VERSION)) != API_VERSION:
            raise RemoteEnvError(
                f"unsupported api_version in response: {meta.get('api_version')}",
                error_type="ProtocolError",
                request_id=req_id,
            )
        if not meta.get("ok", False):
            raise _meta_error(meta, req_id)
        return meta, rblob

    def init(self, ckpt_path: str) -> dict[str, Any]:
        """Load remote checkpoint; returns dp_meta (n_obs_steps, horizon, action_dim, ...)."""
        meta, _ = self._call("init", ckpt_path=ckpt_path)
        self._meta = dict(meta.get("dp_meta") or {})
        return self._meta

    def reset_history(self) -> None:
        self._call("dp_reset_history")

    def predict(
        self,
        main_images: np.ndarray,
        states: np.ndarray,
        init_noise: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """main_images: (B,H,W,C) uint8 or float; states: (B,D) float32; init_noise: (B,T,Da) optional."""
        main_images = np.ascontiguousarray(main_images)
        states = np.ascontiguousarray(states, dtype=np.float32)
        blob = main_images.tobytes() + states.tobytes()
        kw: dict[str, Any] = {
            "main_shape": list(main_images.shape),
            "state_shape": list(states.shape),
            "main_dtype": str(main_images.dtype),
            "state_dtype": str(states.dtype),
            "has_init_noise": init_noise is not None,
        }
        if init_noise is not None:
            init_noise = np.ascontiguousarray(init_noise, dtype=np.float32)
            blob += init_noise.tobytes()
            kw["noise_horizon"] = int(init_noise.shape[1])
            kw["noise_action_dim"] = int(init_noise.shape[2])
            kw["noise_batch"] = int(init_noise.shape[0])
            kw["noise_dtype"] = str(init_noise.dtype)
        meta, rblob = self._call("dp_predict", blob=blob, **kw)
        shape = tuple(meta["action_shape"])
        dt = np.dtype(meta["action_dtype"])
        return np.frombuffer(rblob, dtype=dt).reshape(shape)
