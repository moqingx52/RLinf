# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""RoboTwin 远程 RDT 专家 + DSRL steering 头。

当前与 ``rlinf.models.embodiment.dp_policy`` 共享 ``DpPolicyForRL`` 实现，
通过兼容 TCP 协议的 ``robotwin_rdt_server`` 复用 rollout/replay/SAC 框架。
"""

from rlinf.models.embodiment.dp_policy import get_model

__all__ = ["get_model"]
