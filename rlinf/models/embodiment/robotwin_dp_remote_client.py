# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""RoboTwin ``robotwin_dp_server`` 的 TCP 客户端（协议边界）。

实现位于 ``rlinf.envs.robotwin.dp_remote_client``，此处提供显式别名以便在模型/训练文档中引用，
强调「RLinf 训练进程不 import DP/diffusers，只走 socket」。
"""

from rlinf.envs.robotwin.dp_remote_client import DpRemoteClient

# 与文档/配置中的命名一致
RobotwinDpRemoteClient = DpRemoteClient

__all__ = ["DpRemoteClient", "RobotwinDpRemoteClient"]
