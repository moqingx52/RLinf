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

import logging
from enum import Enum, auto
from typing import Optional

from omegaconf import DictConfig

from rlinf.scheduler import (
    Cluster,
    ComponentPlacement,
    FlexiblePlacementStrategy,
    NodePlacementStrategy,
    PackedPlacementStrategy,
)


class PlacementMode(Enum):
    COLLOCATED = auto()
    DISAGGREGATED = auto()
    HYBRID = auto()
    AUTO = auto()


class HybridComponentPlacement(ComponentPlacement):
    """Hybrid component placement that allows components to run on any sets of GPUs."""

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Initialize HybridComponentPlacement

        Args:
            config (DictConfig): The configuration dictionary.
        """
        super().__init__(config, cluster)
        self._placement_mode = PlacementMode.HYBRID
        self._maybe_shrink_embodied_placement_for_small_parallel_envs(config)

    @staticmethod
    def _largest_valid_embodied_world_size(
        configured_ws: int,
        *,
        train_total: Optional[int],
        eval_total: Optional[int],
        train_constraints_active: bool,
        eval_constraints_active: bool,
        stage_num: int,
        train_group_size: int,
        eval_group_size: int,
    ) -> Optional[int]:
        """Largest g in [1, configured_ws] such that train/eval env counts shard evenly."""
        for g in range(configured_ws, 0, -1):
            ok = True
            if train_constraints_active:
                assert train_total is not None
                if train_total % g != 0:
                    ok = False
                else:
                    q = train_total // g // stage_num
                    if q < 1 or q % train_group_size != 0:
                        ok = False
            if ok and eval_constraints_active:
                assert eval_total is not None
                if eval_total % g != 0:
                    ok = False
                else:
                    q = eval_total // g // stage_num
                    if q < 1 or q % eval_group_size != 0:
                        ok = False
            if ok:
                return g
        return None

    def _filter_rank_map_to_num_processes(self, rank_map, num_processes: int):
        """Keep only process ranks < num_processes (continuous 0..N-1 layout)."""
        new_map = {}
        for res_key, proc_ranks in rank_map.items():
            kept = [p for p in proc_ranks if p < num_processes]
            if kept:
                new_map[res_key] = kept
        return new_map

    def _slice_shared_placement_strategies(self, shared_components: list[str], new_ws: int):
        """Replace placement strategies for components that shared the same strategy object."""
        template = self._placements[shared_components[0]]
        if isinstance(template, FlexiblePlacementStrategy):
            hlist = template._hardware_ranks_list[:new_ws]
            new_strategy = FlexiblePlacementStrategy(
                hlist, node_group_label=template._node_group_labels
            )
        elif isinstance(template, NodePlacementStrategy):
            nr = template._node_ranks[:new_ws]
            new_strategy = NodePlacementStrategy(
                nr, node_group_label=template._node_group_labels
            )
        else:
            raise AssertionError(
                "Shrinking embodied placement is only implemented for "
                "FlexiblePlacementStrategy or NodePlacementStrategy; got "
                f"{type(template)}. Reduce cluster.component_placement manually or increase "
                "total_num_envs."
            )

        rank_map = self._component_rank_map[shared_components[0]]
        new_rank_map = self._filter_rank_map_to_num_processes(rank_map, new_ws)
        for c in shared_components:
            self._placements[c] = new_strategy
            self._component_world_size[c] = new_ws
            self._component_rank_map[c] = new_rank_map

    def _maybe_shrink_embodied_placement_for_small_parallel_envs(self, config: DictConfig):
        """Shrink shared env/rollout/actor placement when total_num_envs is small.

        With ``env, rollout, actor: all`` on many GPUs, a small ``total_num_envs`` would
        otherwise yield zero envs per worker. We lower the worker count for all components
        that share the same placement object so divisibility and CommMapper constraints hold.
        """
        if getattr(config.runner, "task_type", None) != "embodied":
            return
        if "env" not in self._components:
            return
        shrink = getattr(config.cluster, "shrink_embodied_placement", True)
        if not shrink:
            return

        configured_ws = self._component_world_size["env"]
        stage_num = config.rollout.pipeline_stage_num
        train_group = int(config.env.train.group_size)
        eval_group = int(config.env.eval.group_size)

        train_constraints = not config.runner.only_eval
        eval_constraints = (
            config.runner.val_check_interval > 0 or config.runner.only_eval
        )
        train_total = (
            int(config.env.train.total_num_envs) if train_constraints else None
        )
        eval_total = int(config.env.eval.total_num_envs) if eval_constraints else None

        effective = self._largest_valid_embodied_world_size(
            configured_ws,
            train_total=train_total,
            eval_total=eval_total,
            train_constraints_active=train_constraints,
            eval_constraints_active=eval_constraints,
            stage_num=stage_num,
            train_group_size=train_group,
            eval_group_size=eval_group,
        )
        if effective is None:
            return
        if effective >= configured_ws:
            return

        env_strategy = self._placements["env"]
        full_env_hlist = None
        if isinstance(env_strategy, FlexiblePlacementStrategy):
            full_env_hlist = [list(x) for x in env_strategy._hardware_ranks_list]
        shared = [c for c in self._components if self._placements[c] is env_strategy]
        self._slice_shared_placement_strategies(shared, effective)

        # Separate YAML entries produce distinct strategy objects with the same layout; align those too.
        if full_env_hlist is not None:
            for c in ("rollout", "actor"):
                if c not in self._components or c in shared:
                    continue
                if self._component_world_size.get(c) != configured_ws:
                    continue
                st = self._placements[c]
                if not isinstance(st, FlexiblePlacementStrategy):
                    continue
                if [list(x) for x in st._hardware_ranks_list] != full_env_hlist:
                    continue
                self._slice_shared_placement_strategies([c], effective)

        logging.getLogger(__name__).info(
            "Embodied placement: shrunk env/rollout/actor world size from %s to %s "
            "(shared components %s) so small total_num_envs divides evenly. "
            "Set cluster.shrink_embodied_placement=false to disable.",
            configured_ws,
            effective,
            shared,
        )


class ModelParallelComponentPlacement(ComponentPlacement):
    """Component placement for model-parallel components.

    The components must be actor, rollout, and optionally inference, whose GPUs must be continuous.

    This placement supports both collocated and disaggregated modes.

    In the collocated mode, all components share the same set of GPUs. In particular, the rollout group is specially placed in a strided manner to enable fast cudaIPC-based weight sync.
    In the disaggregated mode, each component has its own dedicated set of GPUs.

    In the collocated mode, only actor and rollout exist. While in the disaggregated mode, actor, rollout, and inference should all exist.
    """

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Initialize ModelParallelComponentPlacement

        Args:
            config (DictConfig): The configuration dictionary for the component placement.
        """
        super().__init__(config, cluster)

        self._actor_gpus = self._get_component_hardware("actor")
        self._rollout_gpus = self._get_component_hardware("rollout")
        self._inference_gpus = self._get_component_hardware("inference")
        if self._inference_gpus is None:  # try 'inference' then 'actor_inference'
            self._inference_gpus = self._get_component_hardware("actor_inference")
        self._critic_inference_gpus = self._get_component_hardware("critic_inference")
        self._reward_gpus = self._get_component_hardware("reward")
        self._critic_gpus = self._get_component_hardware("critic")
        self._cluster_num_gpus = cluster.num_accelerators
        assert self._actor_gpus is not None, (
            "Actor GPUs must be specified in the component_placement config."
        )
        assert self._rollout_gpus is not None, (
            "Rollout GPUs must be specified in the component_placement config."
        )
        assert self._reward_gpus is not None, (
            "Reward GPUs must be specified in the component_placement config."
        )
        assert self._actor_gpus == list(
            range(self._actor_gpus[0], self._actor_gpus[-1] + 1)
        ), f"Actor GPUs {self._actor_gpus} must be continuous."
        assert self._rollout_gpus == list(
            range(self._rollout_gpus[0], self._rollout_gpus[-1] + 1)
        ), f"Rollout GPUs {self._rollout_gpus} must be continuous."
        if self._inference_gpus is not None:
            assert self._inference_gpus == list(
                range(self._inference_gpus[0], self._inference_gpus[-1] + 1)
            ), f"Inference GPUs {self._inference_gpus} must be continuous."
        if self._critic_inference_gpus is not None:
            assert self._critic_inference_gpus == list(
                range(
                    self._critic_inference_gpus[0], self._critic_inference_gpus[-1] + 1
                )
            ), (
                f"Critic inference GPUs {self._critic_inference_gpus} must be continuous."
            )

        if self._critic_gpus is not None:
            assert self._critic_gpus == list(
                range(self._critic_gpus[0], self._critic_gpus[-1] + 1)
            ), f"Critic GPUs {self._critic_gpus} must be continuous."

        self._actor_num_gpus = len(self._actor_gpus)
        self._inference_num_gpus = (
            len(self._inference_gpus) if self._inference_gpus else 0
        )
        self._critic_inference_num_gpus = (
            len(self._critic_inference_gpus) if self._critic_inference_gpus else 0
        )
        self._rollout_num_gpus = len(self._rollout_gpus)
        self._reward_num_gpus = len(self._reward_gpus) if self._reward_gpus else 0
        self._critic_num_gpus = len(self._critic_gpus) if self._critic_gpus else 0

        if self._is_auto():
            self._placement_mode = PlacementMode.AUTO
            logging.info("Running in auto mode")
        elif self._is_collocated():
            assert self._inference_gpus is None, (
                "Inference GPUs must not be specified in collocated mode."
            )
            assert self._critic_inference_gpus is None, (
                "Critic inference GPUs must not be specified in collocated mode."
            )
            self._placement_mode = PlacementMode.COLLOCATED
            logging.info("Running in collocated mode")
        elif self._is_disaggregated():
            if self._inference_gpus is not None:
                assert self.inference_tp_size <= self.inference_world_size, (
                    f"Inference TP size {self.inference_tp_size} must be less than or equal to Inference world size {self.inference_world_size}."
                )
                assert self._config.algorithm.recompute_logprobs, (
                    f"algorithm.recompute_logprobs has been set to false, which disables inference. So inference GPUs {self._inference_gpus} must not be specified."
                )

            if self._critic_inference_gpus is not None:
                assert (
                    self.critic_inference_tp_size <= self.critic_inference_world_size
                ), (
                    f"Inference TP size {self.critic_inference_tp_size} must be less than or equal to Inference world size {self.critic_inference_world_size}."
                )

            self._placement_mode = PlacementMode.DISAGGREGATED
            logging.info("Running in disaggregated mode")
        else:
            raise ValueError(
                f"The specified placement does not match either the collocated mode (all the components use the same GPUs) or the disaggregated mode (all the components use completely different GPUs), but got {self._component_rank_map}"
            )

        # Sanity checking
        assert self.actor_tp_size <= self.actor_world_size, (
            f"Actor TP size {self.actor_tp_size} must be less than or equal to Actor world size {self.actor_world_size}."
        )
        assert self.rollout_tp_size <= self.rollout_world_size, (
            f"Rollout TP size {self.rollout_tp_size} must be less than or equal to Rollout world size {self.rollout_world_size}."
        )

        self._generate_placements()

    def _is_auto(self):
        if not getattr(self._config.cluster, "auto_scheduler", False):
            return False

        # TODO for now critic model is not supported in auto scheduling mode
        if self._critic_gpus is not None:
            return False

        assert self._is_disaggregated(), (
            "AUTO mode is a more advanced version of disaggregated mode, so it must satisfy the requirements of disaggregated mode."
        )

        # Assert components order is : actor -> rollout -> inference
        order_error_msg = "AUTO mode requires components to be placed in the order of actor -> rollout -> inference."
        assert (
            self._actor_gpus[0] == 0
            and self._actor_gpus[-1] == self._rollout_gpus[0] - 1
        ), order_error_msg
        if self._inference_gpus is None:
            assert self._rollout_gpus[-1] == self._cluster_num_gpus - 1, order_error_msg
        else:
            assert self._rollout_gpus[-1] == self._inference_gpus[0] - 1, (
                order_error_msg
            )
            assert self._inference_gpus[-1] == self._cluster_num_gpus - 1, (
                order_error_msg
            )
        return True

    def _is_collocated(self):
        if self._actor_gpus == self._rollout_gpus:
            return True
        return False

    def _is_disaggregated(self):
        actor_gpu_set = set(self._actor_gpus)
        critic_gpu_set = set([] if self._critic_gpus is None else self._critic_gpus)
        rollout_gpu_set = set(self._rollout_gpus)
        inference_gpu_set = (
            [] if self._inference_gpus is None else set(self._inference_gpus)
        )
        critic_inference_gpu_set = set(
            [] if self._critic_inference_gpus is None else self._critic_inference_gpus
        )

        return (
            actor_gpu_set.isdisjoint(rollout_gpu_set)
            and actor_gpu_set.isdisjoint(inference_gpu_set)
            and rollout_gpu_set.isdisjoint(inference_gpu_set)
            and critic_gpu_set.isdisjoint(actor_gpu_set)
            and critic_gpu_set.isdisjoint(rollout_gpu_set)
            and critic_gpu_set.isdisjoint(critic_inference_gpu_set)
            and rollout_gpu_set.isdisjoint(critic_inference_gpu_set)
        )

    def _generate_placements(self):
        if self._placement_mode == PlacementMode.COLLOCATED:
            self._placements["actor"] = PackedPlacementStrategy(
                self._actor_gpus[0], self._actor_gpus[-1]
            )

            if self.actor_tp_size > self.rollout_tp_size:
                assert self.actor_tp_size % self.rollout_tp_size == 0, (
                    f"Actor TP size ({self.actor_tp_size}) must be divisible by Rollout TP size ({self.rollout_tp_size})"
                )
            stride = (
                self.actor_tp_size // self.rollout_tp_size
                if self.actor_tp_size > self.rollout_tp_size
                else 1
            )
            self._placements["rollout"] = PackedPlacementStrategy(
                self._rollout_gpus[0],
                self._rollout_gpus[-1],
                num_hardware_per_process=self.rollout_tp_size,
                stride=stride,
            )
            if self._reward_gpus:
                self._placements["reward"] = PackedPlacementStrategy(
                    self._reward_gpus[0], self._reward_gpus[-1]
                )
            if self._critic_gpus is not None:
                self._placements["critic"] = PackedPlacementStrategy(
                    self._critic_gpus[0], self._critic_gpus[-1]
                )
        elif self._placement_mode == PlacementMode.DISAGGREGATED:
            num_gpus_per_rollout_dp = len(self._rollout_gpus) // self.rollout_dp_size
            self._placements["rollout"] = PackedPlacementStrategy(
                self._rollout_gpus[0],
                self._rollout_gpus[-1],
                num_hardware_per_process=num_gpus_per_rollout_dp,
            )
            if self._inference_gpus is not None:
                # TODO check the placement name
                self._placements[
                    "inference"
                    if self._critic_inference_gpus is None
                    else "actor_inference"
                ] = PackedPlacementStrategy(
                    self._inference_gpus[0], self._inference_gpus[-1]
                )
            if self._critic_inference_gpus is not None:
                self._placements["critic_inference"] = PackedPlacementStrategy(
                    self._critic_inference_gpus[0], self._critic_inference_gpus[-1]
                )
            self._placements["actor"] = PackedPlacementStrategy(
                self._actor_gpus[0], self._actor_gpus[-1]
            )
            if self._reward_gpus:
                self._placements["reward"] = PackedPlacementStrategy(
                    self._reward_gpus[0], self._reward_gpus[-1]
                )
            if self._critic_gpus is not None:
                self._placements["critic"] = PackedPlacementStrategy(
                    self._critic_gpus[0], self._critic_gpus[-1]
                )
        elif self._placement_mode == PlacementMode.AUTO:
            # In AUTO mode, actor will be placed on all GPUs
            self._placements["actor"] = PackedPlacementStrategy(
                0, self._cluster_num_gpus - 1
            )

            if self._critic_gpus is not None:
                assert False, (
                    "auto placement is not supported when having critic model for now"
                )

            use_pre_process_policy = getattr(
                self._config.cluster, "use_pre_process_policy", False
            )
            if use_pre_process_policy:
                assert (
                    self._actor_gpus[-1] - self._actor_gpus[0] + 1
                ) % self.rollout_tp_size == 0
                self._rollout_gpus = (
                    list(range(1 + self._actor_gpus[-1])) + self._rollout_gpus
                )
                self._rollout_num_gpus = len(self._rollout_gpus)

            num_gpus_per_rollout_dp = len(self._rollout_gpus) // self.rollout_dp_size
            self._placements["rollout"] = PackedPlacementStrategy(
                self._rollout_gpus[0],
                self._rollout_gpus[-1],
                num_hardware_per_process=num_gpus_per_rollout_dp,
            )

            if self._inference_gpus is not None:
                self._placements["inference"] = PackedPlacementStrategy(
                    self._inference_gpus[0], self._inference_gpus[-1]
                )
            if self._reward_gpus:
                self._placements["reward"] = PackedPlacementStrategy(
                    self._reward_gpus[0], self._reward_gpus[-1]
                )

    @property
    def is_collocated(self):
        return self._placement_mode == PlacementMode.COLLOCATED

    @property
    def is_disaggregated(self):
        return self._placement_mode == PlacementMode.DISAGGREGATED

    @property
    def is_auto(self):
        return self._placement_mode == PlacementMode.AUTO

    @property
    def is_pipeline(self):
        return self.is_disaggregated or self.is_auto

    def has_dedicated_inference_for_role(self, role):
        if role == "actor":
            return self.has_dedicated_actor_inference
        elif role == "critic":
            return self.has_dedicated_critic_inference
        else:
            assert False, (
                f"Unknown role {role} while calling has_dedicated_inference_for_role"
            )

    @property
    def has_dedicated_inference(self):
        return (
            self._placement_mode in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]
            and self._inference_gpus is not None
        )

    @property
    def has_dedicated_actor_inference(self):
        return (
            self._placement_mode in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]
            and self._inference_gpus is not None
        )

    @property
    def has_dedicated_critic_inference(self):
        return (
            self._placement_mode in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]
            and self._critic_inference_gpus is not None
        )

    @property
    def actor_dp_size(self) -> int:
        return self._actor_num_gpus // (
            self._config.actor.model.get("tensor_model_parallel_size", 1)
            * self._config.actor.model.get("context_parallel_size", 1)
            * self._config.actor.model.get("pipeline_model_parallel_size", 1)
        )

    @property
    def critic_dp_size(self) -> int:
        return self._critic_num_gpus // (
            self._config.critic.model.get("tensor_model_parallel_size", 1)
            * self._config.critic.model.get("context_parallel_size", 1)
            * self._config.critic.model.get("pipeline_model_parallel_size", 1)
        )

    @property
    def actor_tp_size(self) -> int:
        return self._config.actor.model.get("tensor_model_parallel_size", 1)

    @property
    def critic_tp_size(self) -> int:
        return self._config.critic.model.get("tensor_model_parallel_size", 1)

    @property
    def actor_pp_size(self) -> int:
        return self._config.actor.model.get("pipeline_model_parallel_size", 1)

    @property
    def critic_pp_size(self) -> int:
        return self._config.critic.model.get("pipeline_model_parallel_size", 1)

    @property
    def actor_world_size(self) -> int:
        return self._actor_num_gpus

    @property
    def critic_world_size(self) -> int:
        return self._critic_num_gpus

    @property
    def inference_tp_size(self) -> int:
        if hasattr(self._config, "inference"):
            infer_cfg = self._config.inference
        elif hasattr(self._config, "actor_inference"):
            infer_cfg = self._config.actor_inference
        else:
            return self.actor_tp_size

        return infer_cfg.model.get("tensor_model_parallel_size", 1)

    @property
    def critic_inference_tp_size(self) -> int:
        if (
            hasattr(self._config, "critic_inference")
            and hasattr(self._config.critic_inference, "model")
            and hasattr(
                self._config.critic_inference.model, "tensor_model_parallel_size"
            )
        ):
            return self._config.critic_inference.model.get(
                "tensor_model_parallel_size", 1
            )
        else:
            return self.critic_tp_size

    @property
    def inference_pp_size(self) -> int:
        if hasattr(self._config, "inference"):
            infer_cfg = self._config.inference
        elif hasattr(self._config, "actor_inference"):
            infer_cfg = self._config.actor_inference
        else:
            return self.actor_pp_size
        return infer_cfg.model.get("pipeline_model_parallel_size", self.actor_pp_size)

    @property
    def critic_inference_pp_size(self) -> int:
        if (
            hasattr(self._config, "critic_inference")
            and hasattr(self._config.critic_inference, "model")
            and hasattr(
                self._config.critic_inference.model, "pipeline_model_parallel_size"
            )
        ):
            return self._config.critic_inference.model.get(
                "pipeline_model_parallel_size", 1
            )
        else:
            return self.critic_pp_size

    @property
    def inference_dp_size(self) -> int:
        return self._inference_num_gpus // (
            self.inference_tp_size * self.inference_pp_size
        )

    @property
    def critic_inference_dp_size(self) -> int:
        return self._critic_inference_num_gpus // (
            self.critic_inference_tp_size * self.critic_inference_pp_size
        )

    @property
    def inference_world_size(self) -> int:
        return self._inference_num_gpus

    @property
    def critic_inference_world_size(self) -> int:
        return self._critic_inference_num_gpus

    @property
    def rollout_dp_size(self) -> int:
        return self._rollout_num_gpus // (
            self._config.rollout.get("tensor_parallel_size", 1)
            * self._config.rollout.get("pipeline_parallel_size", 1)
        )

    @property
    def rollout_tp_size(self) -> int:
        return self._config.rollout.get("tensor_parallel_size", 1)

    @property
    def rollout_world_size(self) -> int:
        return self._rollout_num_gpus

    @property
    def reward_world_size(self) -> int:
        return self._reward_num_gpus

    def _get_component_hardware(self, component_name: str):
        if component_name not in self._component_rank_map:
            return None
        return super().get_hardware_ranks(component_name)


class ModelParallelEvalComponentPlacement(ComponentPlacement):
    """Component placement for model-parallel components in eval.

    The components must be rollout and reward, whose GPUs must be continuous.

    This placement only supports collocated mode.
    """

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Initialize ModelParallelEvalComponentPlacement

        Args:
            config (DictConfig): The configuration dictionary for the component placement.
        """
        super().__init__(config, cluster)

        self._rollout_gpus = self._get_component_hardware("rollout")
        self._reward_gpus = self._get_component_hardware("reward")
        self._cluster_num_gpus = cluster.num_accelerators
        assert self._rollout_gpus is not None, (
            "Rollout GPUs must be specified in the component_placement config."
        )
        assert self._reward_gpus is not None, (
            "Reward GPUs must be specified in the component_placement config."
        )
        assert self._rollout_gpus == list(
            range(self._rollout_gpus[0], self._rollout_gpus[-1] + 1)
        ), f"Rollout GPUs {self._rollout_gpus} must be continuous."

        self._rollout_num_gpus = len(self._rollout_gpus)
        self._reward_num_gpus = len(self._reward_gpus) if self._reward_gpus else 0

        self._placement_mode = PlacementMode.COLLOCATED

        # Sanity checking
        assert self.rollout_tp_size <= self.rollout_world_size, (
            f"Rollout TP size {self.rollout_tp_size} must be less than or equal to Rollout world size {self.rollout_world_size}."
        )

        self._generate_placements()

    def _generate_placements(self):
        assert self._placement_mode == PlacementMode.COLLOCATED
        self._placements["rollout"] = PackedPlacementStrategy(
            self._rollout_gpus[0],
            self._rollout_gpus[-1],
            num_hardware_per_process=self.rollout_tp_size,
            stride=1,
        )
        if self._reward_gpus:
            self._placements["reward"] = PackedPlacementStrategy(
                self._reward_gpus[0], self._reward_gpus[-1]
            )

    @property
    def is_collocated(self):
        return True

    @property
    def is_disaggregated(self):
        return False

    @property
    def is_auto(self):
        return False

    @property
    def is_pipeline(self):
        return False

    @property
    def has_dedicated_inference(self):
        return False

    @property
    def rollout_dp_size(self) -> int:
        return self._rollout_num_gpus // (
            self._config.rollout.get("tensor_parallel_size", 1)
            * self._config.rollout.get("pipeline_parallel_size", 1)
        )

    @property
    def rollout_tp_size(self) -> int:
        return self._config.rollout.get("tensor_parallel_size", 1)

    @property
    def rollout_world_size(self) -> int:
        return self._rollout_num_gpus

    @property
    def reward_world_size(self) -> int:
        return self._reward_num_gpus

    def _get_component_hardware(self, component_name: str):
        if component_name not in self._component_rank_map:
            return None
        return super().get_hardware_ranks(component_name)
