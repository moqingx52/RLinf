# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""RoboTwin 远程/本地 DP 专家 + DSRL  steering 头。

与 ``rlinf.models.embodiment.dp_policy`` 共用 ``DpPolicyForRL`` 实现；仅 ``model_type`` 不同，
便于在配置中区分「OpenVLA/OpenPI」与「RoboTwin+DP+DSRL」路线。
"""

from rlinf.models.embodiment.dp_policy import get_model

__all__ = ["get_model"]
