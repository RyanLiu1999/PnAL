from __future__ import annotations

import os
import sys
import subprocess

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # parallel processes, one CPU core each

import highway_env  # noqa: F401  # Install dependencies explicitly before running.

import copy
import dataclasses
import hashlib
import json
import logging
import math
import numpy as np
import operator
import platform
import random
import time
import urllib.request
import warnings
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, fields, replace
from statistics import NormalDist
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from highway_env import utils
from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.intersection_env import IntersectionEnv
from highway_env.envs.roundabout_env import RoundaboutEnv
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from scipy import stats
from sklearn.cluster import KMeans
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt
import pandas as pd


warnings.filterwarnings("ignore")

# ===========================
# Global Configuration — 补充实验（只评估，不训练）: (1) 跨场景 τc 敏感性（验证集）; (2) Roundabout 组件归因缺失 seed 的补跑
# ===========================
# DeepSeek 接口（作者要求保留）。本文件不训练、不精修模板库，不发起任何 API 调用。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEEPSEEK_MODEL = "deepseek-r1-250528"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 三个窗口: 只改 SHARD ------------------------------------------------------------------------------------
SHARD = "A"                          # ← 三个 Colab 窗口分别改为 "A" / "B" / "C"（= seed 0 / 1 / 2）; "all" = 一个窗口跑 seed 0–2; "summary" = 只汇总
SEEDS = {"A": [0], "B": [1], "C": [2]}.get(SHARD, [0, 1, 2])
SUPP_WORKERS = 6                     # 每个窗口内并行的进程数（上限为 CPU 核数）; 只影响速度。内存不足时改小

# ---- 已完成 run 的来源（只读; 不重新训练）与结果文件夹 ----------------------------------------------------------------
DRIVE = "/content/drive/MyDrive"
TABLE3_DIR = f"{DRIVE}/pnal_results_table3_intersection"      # Intersection 对比表（N = 2、3、5 的 PnAL: 新 δ_temp 规则）
ATTR_V1_DIR = f"{DRIVE}/pnal_results_v1"                       # 旧版 N = 4 组件归因（旧规则; 已存档为 seed<k>_superseded_<时间>）
TABLE4_DIR = f"{DRIVE}/pnal_results_table4"                    # Table 4（Roundabout / Highway; 旧规则）
RB_ATTR_DIR = f"{DRIVE}/pnal_results_roundabout_attribution"   # Roundabout 组件归因（本文件补齐缺失的 seed 并加入新条件）
SUPP_DIR = f"{DRIVE}/pnal_results_supplement"                  # 本文件的结果（Intersection、Highway 与跨场景汇总表）
LOCAL_CHECKPOINT_DIR = "/content/pnal_local_checkpoints"      # 检查点暂存到运行时本地磁盘并校验后再加载（不放在 Drive 上）
# 来源 run 的选择: "current" = <来源>/<场景>_N<n>/<方法>/seed<k>;
#                 "archived" = 同一位置的 seed<k>_superseded_<时间> 中，配置与该设置的协议（含 δ_temp 规则）一致的那一个
#                              （没有符合的存档 run 时，再检查原位置 seed<k>）。
# 只有已完成、文件齐全且保存的配置与协议一致的 run 才会被使用，否则报错（不会静默换用其他 run）。
# 若有多个存档 run 都符合且结果不同，会报错并列出; 此时在 SOURCE_OVERRIDES 中指定论文所用的那一个:
#   SOURCE_OVERRIDES = {("intersection", 4, "full_pnal", 0): "/content/drive/MyDrive/pnal_results_v1/intersection_N4/full_pnal/seed0_superseded_20261017-154721"}
SOURCE_OVERRIDES = {}

# ---- 8 个设置: 与论文中各表使用的 run 完全相同 ------------------------------------------------------------------------
SETTINGS = {
    "intersection_N2": dict(scenario="intersection", n=2, delta_temp_rule="none", results=SUPP_DIR,
                            sources={"full_pnal": (TABLE3_DIR, "current")}),
    "intersection_N3": dict(scenario="intersection", n=3, delta_temp_rule="none", results=SUPP_DIR,
                            sources={"full_pnal": (TABLE3_DIR, "current")}),
    "intersection_N4": dict(scenario="intersection", n=4, delta_temp_rule="paper_max", results=SUPP_DIR,
                            sources={"full_pnal": (ATTR_V1_DIR, "archived")}),
    "intersection_N5": dict(scenario="intersection", n=5, delta_temp_rule="none", results=SUPP_DIR,
                            sources={"full_pnal": (TABLE3_DIR, "current")}),
    "roundabout_N2": dict(scenario="roundabout", n=2, delta_temp_rule="paper_max", results=RB_ATTR_DIR,
                          sources={"full_pnal": (TABLE4_DIR, "current"), "ippo": (TABLE4_DIR, "current"),
                                   "ippo_plus_shield": (RB_ATTR_DIR, "current")}),
    "roundabout_N4": dict(scenario="roundabout", n=4, delta_temp_rule="paper_max", results=RB_ATTR_DIR,
                          sources={"full_pnal": (TABLE4_DIR, "current"), "ippo": (TABLE4_DIR, "current"),
                                   "ippo_plus_shield": (RB_ATTR_DIR, "current")}),
    "highway_N2": dict(scenario="highway", n=2, delta_temp_rule="paper_max", results=SUPP_DIR,
                       sources={"full_pnal": (TABLE4_DIR, "current")}),
    "highway_N4": dict(scenario="highway", n=4, delta_temp_rule="paper_max", results=SUPP_DIR,
                       sources={"full_pnal": (TABLE4_DIR, "current")}),
}
# 论文表中的 3-seed 测试成功率（%），只用于核对所用 run 是否正确（来源表中并列显示）
PAPER_TEST_SUCCESS = {("intersection_N2", "full_pnal"): 91.5, ("intersection_N3", "full_pnal"): 78.5,
                      ("intersection_N4", "full_pnal"): 62.8, ("intersection_N5", "full_pnal"): 49.3,
                      ("roundabout_N2", "full_pnal"): 85.7, ("roundabout_N2", "ippo"): 75.2, ("roundabout_N2", "ippo_plus_shield"): 83.0,
                      ("roundabout_N4", "full_pnal"): 63.8, ("roundabout_N4", "ippo"): 49.8, ("roundabout_N4", "ippo_plus_shield"): 58.3,
                      ("highway_N2", "full_pnal"): 98.3, ("highway_N4", "full_pnal"): 97.2}

# ---- 评估期条件（每个只改一个因素）----------------------------------------------------------------------------------------
# (1) τc 敏感性: 8 个设置的 PnAL, 验证集; 0.6 = 论文设置; 0.0 = 匹配即采纳; 1.0 = 实际上从不采纳（准入率列给出实测值）
TAU_GRID = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TAU_SWEEP = [f"sweep_tau_{t}" for t in TAU_GRID]
# (2) Roundabout 组件归因（与归因文件完全相同的 20 个条件）: 已完成的直接沿用缓存, 只补缺失的 seed
RB_ATTRIBUTION_TEST = ["gate_uncalibrated", "gate_always_trust", "no_semantic", "no_coordination", "no_shield", "no_fusion",
                       "source_untrained", "source_cruise", "semantic_reversed_recalibrated", "semantic_corrupted_always_trust",
                       "density_heavy", "noise_1m", "semantic_unrefined_recalibrated"]
COORD_SWEEP = ["sweep_coord_paper", "sweep_coord_equal", "sweep_coord_agent_id_order"]
SUPPLEMENT_PLAN = {   # 设置 -> 方法 -> {"validation": [...], "test": [...]}
    "intersection_N2": {"full_pnal": {"validation": TAU_SWEEP}},
    "intersection_N3": {"full_pnal": {"validation": TAU_SWEEP}},
    "intersection_N4": {"full_pnal": {"validation": TAU_SWEEP}},
    "intersection_N5": {"full_pnal": {"validation": TAU_SWEEP}},
    "roundabout_N2": {"full_pnal": {"validation": TAU_SWEEP + COORD_SWEEP, "test": RB_ATTRIBUTION_TEST},
                      "ippo": {"test": ["shield_added"]},
                      "ippo_plus_shield": {"test": ["no_shield", "density_heavy", "noise_1m"]}},
    "highway_N2": {"full_pnal": {"validation": TAU_SWEEP}},
    "highway_N4": {"full_pnal": {"validation": TAU_SWEEP}},
}
SUPPLEMENT_PLAN["roundabout_N4"] = SUPPLEMENT_PLAN["roundabout_N2"]
ANALYSIS_EPISODES = 200              # 测试集条件: 公共测试局的前 200 局（与论文主评估相同）
SWEEP_EPISODES = 100                 # 敏感性扫描: 验证集 100 局（与协调权重扫描相同）
REPRO_EPISODES = 50                  # 复现核对（Roundabout 测试集条件之前）: 重新评估未改动的通路, 必须与保存的测试结果逐局相同

# ---- 协议（与所有来源 run 相同; 每个设置只改 δ_temp 规则, 由上面的 SETTINGS 给出）------------------------------------
PROFILE = "medium"
BUDGETS = {"medium": dict(meta_steps=100, episodes_per_meta_step=20, calibration_episodes=200, evaluation_episodes=200,
                          selection_episodes=50, refinement_episodes=150)}
SCENARIO, N_AGENTS = "intersection", 4   # 运行时按设置切换
DELTA_TEMP_RULE = "paper_max"            # 运行时按设置切换（"none" = 新规则; "paper_max" = 旧规则 max(δ_t, δ_temp)）
DELTA_TEMP_CAP = 1.0
RISK_THRESHOLD_DELTA = 0.3               # δ_t（三个 testbed 共用）
INTERSECTION_DESTINATIONS = "random_turn"
INTERSECTION_TARGET_SPEEDS = (0.0, 5.0, 10.0)
INTERSECTION_LEGACY_SPEEDS = False
INTERSECTION_DURATION_S = None
INTERSECTION_CONTROL_MODE = "cautious_to_full"
INTERSECTION_ACTION_SPACE = "paper"
INTERSECTION_SECOND_ROW_OFFSET_M = 30.0
ROUNDABOUT_BACKGROUND_VEHICLES = 2
ROUNDABOUT_SPEED_FLOOR_INDEX = 1
ROUNDABOUT_NORMALIZE_REWARD = False
HIGHWAY_FAST_SIMULATION = True
REWARD_ARRIVAL_ONCE = True
REWARD_MODE = "individual"
COMMON_EVAL_SET = True
MODEL_SELECTION = "best_validation"
CHECKPOINT_VARIANTS = False
SEMANTIC_REFINEMENT_ROUNDS = 3
SEMANTIC_REFINEMENT_MODE = "rule"
RAW_CONFIDENCE_MODE = "conditional_reliability"
CALIBRATION_CACHE = False
# 条件注册表中引用的取值（与组件归因相同; 只运行 SUPPLEMENT_PLAN 中列出的条件, 其余条件不运行）
DELTA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
OBS_NOISE_LEVELS = {"noise_1m": (1.0, 0.5), "noise_2m": (2.0, 1.0)}
DENSITY_LEVELS = {"density_light": (5, 0.3), "density_heavy": (10, 0.6)}   # Intersection
ROUNDABOUT_DENSITY_LEVELS = {"density_heavy": 6}                              # Roundabout: 背景车 2 → 6 辆
COORDINATION_WEIGHT_SETS = {
    "paper": (0.4, 0.2, 0.3, 0.1), "equal": (0.25, 0.25, 0.25, 0.25), "risk_only": (1.0, 0.0, 0.0, 0.0),
    "urgency_only": (0.0, 1.0, 0.0, 0.0), "rule_only": (0.0, 0.0, 1.0, 0.0), "agent_id_order": (0.0, 0.0, 0.0, 0.0)}
FALLBACK_RANKINGS = {
    "paper": ("DECELERATE", "MAINTAIN", "LANE_RIGHT", "LANE_LEFT", "ACCELERATE"),
    "maintain_first": ("MAINTAIN", "DECELERATE", "LANE_RIGHT", "LANE_LEFT", "ACCELERATE"),
    "lateral_first": ("LANE_RIGHT", "LANE_LEFT", "DECELERATE", "MAINTAIN", "ACCELERATE")}
PRESET_ORDER = ["full_pnal", "ippo", "ippo_plus_shield", "ippo_full_fusion"]
PRESETS = ["full_pnal"]
REFERENCE_PRESET = "full_pnal"
USE_DRIVE = True
RESULTS_DIR = SUPP_DIR                   # 运行时按设置切换（Roundabout 写入 RB_ATTR_DIR）

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ===========================
# Global Configuration Classes: 唯一阈值 τc、实验开关、ablation preset、testbed 与各模块参数
# ===========================
# Central configuration for the revised PnAL implementation.
#
# Every tunable quantity used by the online fusion pathway, the calibration stage,
# the learned source and the training-time adaptation interface lives here, so that
# (i) all ablations run from ONE codebase with flags, and (ii) the full configuration
# can be serialised into experiment metadata.
#
# Nothing in this file is changed online during evaluation.

# ---------------------------------------------------------------------------
# Global method constants (single source of truth)
# ---------------------------------------------------------------------------
#: The ONLY semantic admission threshold tau_c (paper Eq. 25; Appendix A selects 0.6).
#: The online gate is exactly ``calibrated_confidence >= SEMANTIC_CONFIDENCE_THRESHOLD``.
SEMANTIC_CONFIDENCE_THRESHOLD: float = 0.6

#: Reported experiments use a frozen semantic library (no online template edits).
SEMANTIC_LIBRARY_FROZEN: bool = True

#: Canonical primitive action names used everywhere inside the algorithm.
CANONICAL_ACTIONS: Tuple[str, ...] = (
    "MAINTAIN",
    "LANE_LEFT",
    "LANE_RIGHT",
    "ACCELERATE",
    "DECELERATE",
)

#: highway-env DiscreteMetaAction label for each canonical name (label, not integer id).
CANONICAL_TO_HIGHWAY_LABEL: Dict[str, str] = {
    "MAINTAIN": "IDLE",
    "LANE_LEFT": "LANE_LEFT",
    "LANE_RIGHT": "LANE_RIGHT",
    "ACCELERATE": "FASTER",
    "DECELERATE": "SLOWER",
}


# ---------------------------------------------------------------------------
# Experiment flags (ablations)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentFlags:
    use_semantic: bool = True
    use_calibration: bool = True
    always_trust_semantic: bool = False
    use_safety_shield: bool = True
    use_coordination: bool = True
    use_training_adaptation: bool = True
    freeze_semantic_library: bool = True
    # Optional, OFF by default (not described in the first-version paper).
    # NOTE: the original EnhancedSemanticRewardFunction (reward shaping added to the env
    # reward and to the reported team reward) is REMOVED: it is not described in the paper,
    # baselines did not receive it, and it used hard-coded action integers.
    use_curriculum_aux: bool = False
    adapt_safety_layer_online: bool = False
    #: False (default): template threshold delta_temp enters delta_eff only if the template is ADMITTED
    #: (kappa >= tau_c).  True: literal paper Algorithm 1 (delta_temp whenever a template is matched,
    #: even if the calibrated gate rejects it) -> at Intersection N=4 this multiplied risk-threshold
    #: fallbacks ~20x and turned most episodes into timeouts (report #37).
    delta_temp_when_matched: bool = False

    def validate(self) -> None:
        if self.always_trust_semantic and not self.use_semantic:
            raise ValueError("always_trust_semantic requires use_semantic=True")
        if not self.freeze_semantic_library:
            raise ValueError(
                "Reported experiments must use a frozen semantic library "
                "(freeze_semantic_library=True). Online template editing is not part of PnAL."
            )


#: Named ablations required by the revision. All share one codebase.
ABLATION_PRESETS: Dict[str, ExperimentFlags] = {
    "full_pnal": ExperimentFlags(),
    "pnal_wo_semantic": ExperimentFlags(use_semantic=False, use_calibration=False),
    "pnal_wo_safety_shield": ExperimentFlags(use_safety_shield=False),
    "pnal_wo_coordination": ExperimentFlags(use_coordination=False),
    "pnal_wo_adaptation": ExperimentFlags(use_training_adaptation=False),
    "pnal_semantic_uncalibrated": ExperimentFlags(use_calibration=False),
    "pnal_always_trust_semantic": ExperimentFlags(always_trust_semantic=True),
    "pnal_policy_only": ExperimentFlags(
        use_semantic=False,
        use_calibration=False,
        use_safety_shield=False,
        use_coordination=False,
    ),
    # IPPO (independent PPO, plain MLP, no PathNet, no meta scheduler) + the SAME SafetyShield.
    "ippo_plus_shield": ExperimentFlags(
        use_semantic=False,
        use_calibration=False,
        use_coordination=False,
        use_training_adaptation=False,
    ),
    # Reviewer 3 comment 1 (reference row): plain IPPO -- independent PPO on a plain MLP, no PathNet, no meta
    # scheduler, NO online fusion component (no semantic source, no SafetyShield, no coordination).
    "ippo": ExperimentFlags(
        use_semantic=False,
        use_calibration=False,
        use_safety_shield=False,
        use_coordination=False,
        use_training_adaptation=False,
    ),
    # Reviewer 2 comment 9: the SAME online fusion stack (semantic + calibration + shield + coordination) on
    # an IPPO learner (plain MLP, no PathNet, no SAC meta-scheduler).
    "ippo_full_fusion": ExperimentFlags(use_training_adaptation=False),
    # Factorial example requested by Reviewer 1.
    "pnal_wo_semantic_wo_shield": ExperimentFlags(
        use_semantic=False, use_calibration=False, use_safety_shield=False
    ),
}

#: Presets whose learned source is a plain IPPO MLP instead of PathNet.
IPPO_BACKBONE_PRESETS = {"ippo", "ippo_plus_shield", "ippo_full_fusion"}

#: MARL baselines of paper Tables 3/4 (pnal/baselines.py).  They run inside the SAME protocol as PnAL / IPPO
#: (testbed, action protocol incl. the nominal-speed floor, reward, seeds, budget, model selection, test set) and,
#: like plain IPPO, have NO online fusion component: their joint action is executed as proposed.
BASELINE_PRESETS = ("mappo", "hqmix", "cs_maddpg")
for _b in BASELINE_PRESETS:
    ABLATION_PRESETS[_b] = ExperimentFlags(
        use_semantic=False, use_calibration=False, use_safety_shield=False, use_coordination=False,
        use_training_adaptation=False,
    )
del _b


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------
@dataclass
class ScenarioConfig:
    name: str  # "intersection" | "roundabout" | "highway"
    n_agents: int
    duration_s: float
    policy_frequency: int = 1
    simulation_frequency: int = 15
    #: explicit DiscreteMetaAction settings (never rely on library defaults)
    longitudinal: bool = True
    lateral: bool = True
    target_speeds: Tuple[float, ...] = (0.0, 5.0, 10.0)
    observation_vehicles_count: int = 10
    enable_background: bool = False
    total_vehicles: Optional[int] = None
    #: Intersection interaction diversity (report #56/#57):
    #:   "fixed_o1"    : every controlled vehicle exits at o1 (highway-env default implicitly used by the original code)
    #:   "random_turn" : each controlled vehicle exits at a uniformly random exit other than its own entry
    #:                   (right turn / straight / left turn), drawn with the same RNG call as the stock env
    intersection_destinations: str = "fixed_o1"
    #: env-specific extra highway-env config (rewards etc.)
    extra_env_config: Dict = field(default_factory=dict)
    #: False = background vehicles do not check collisions among themselves (highway-env ``highway-fast-v0``); collisions
    #: of a CONTROLLED vehicle with any vehicle are still detected (the controlled vehicle keeps its check).  Highway only.
    background_collisions: bool = True
    #: highway-fast-v0 execution (Highway only): background vehicles do not collision-check against each other
    #: (controlled vehicles still collide with everything); used together with simulation_frequency = 5.
    background_vehicle_collisions: bool = True
    #: Highway has no destination: success requires every controlled vehicle to
    #: travel at least this distance [m] before the time limit.
    highway_target_distance_m: float = 160.0
    #: Roundabout: arrival once this far along the destination exit lane [m].
    roundabout_exit_distance_m: float = 20.0
    #: Intersection arrival distance on the exit lane (highway-env default 25 m).
    intersection_exit_distance_m: float = 25.0
    #: None = stock intersection-v0 spawning (valid only for N <= 4; N >= 5 raises
    #: TestbedIncompatibilityError).  A number enables StaggeredIntersectionEnv (documented
    #: testbed change: vehicles k >= 4 spawn this many metres further upstream).
    intersection_second_row_offset_m: Optional[float] = None
    #: Nominal-speed floor (index into target_speeds).  The NOMINAL action (learned proposal or admitted
    #: semantic prior) may not command a speed below target_speeds[floor]: DECELERATE at/below the floor
    #: becomes MAINTAIN, and MAINTAIN / DECELERATE / a lateral no-op while stopped below the floor becomes
    #: ACCELERATE (resume).  Standstill is thereby reserved for the SafetyShield / coordination fallback,
    #: which still decides after this step.  None = disabled (default).  Tested with index 1 (5 m/s): more
    #: collisions and lower return in IPPO+Shield training (report #42), so it is OFF by default.
    nominal_speed_floor_index: Optional[int] = None
    #: Reward-exploit fix (report #43, author decision): stock intersection-v0 pays arrived_reward (+5) to an
    #: arrived vehicle at EVERY step, while the episode only ends when ALL vehicles arrive -> the return is
    #: maximised by letting some vehicles arrive and one linger until the time limit.  With True, each
    #: agent's reward after the step on which it arrived is 0 (the arrival reward is paid exactly once), the
    #: shared reward is the mean of these per-agent rewards, and the arrived agent's PPO trajectory ends
    #: (terminal) at arrival.  Reward values themselves are unchanged.  Applied to every method alike.
    reward_arrival_once: bool = True
    #: Actions the LEARNED policy may propose (logit mask; identical in sampling, log-prob and PPO update).
    #: None = all five canonical actions.  The env action space, the SafetyShield and the coordination
    #: fallback still use all five.  Intersection "cautious_to_full" mode: (MAINTAIN, ACCELERATE, DECELERATE)
    #: because LANE_LEFT/RIGHT are physical no-ops on its single-lane roads (report #45).
    policy_action_set: Optional[Tuple[str, ...]] = None

    @property
    def max_decision_steps(self) -> int:
        return int(round(self.duration_s * self.policy_frequency))


def intersection_config(
    n_agents: int, legacy_target_speeds: bool = False, second_row_offset_m: Optional[float] = None
) -> ScenarioConfig:
    """Intersection testbed, paper Table 2: 15 s, controlled agents 2-5.

    The original script overrode highway-env's intersection action config with a bare
    ``{"type": "DiscreteMetaAction"}``, which silently inherited MDPVehicle default
    target speeds [20, 25, 30] m/s (lane speed limit is 10 m/s).  The revision sets
    target speeds explicitly.  ``legacy_target_speeds=True`` reproduces the original
    effective values for a controlled comparison.
    """
    # Default [0, 5, 10] m/s (report #40): top speed = lane speed limit (10 m/s), a zero step allows
    # yielding, and the 69-88 m routes take 7-9 s at full speed within the 15 s episode.
    # stock intersection-v0 uses [0, 4.5, 9]; legacy_target_speeds=True reproduces the original script.
    speeds = (20.0, 25.0, 30.0) if legacy_target_speeds else (0.0, 5.0, 10.0)
    return ScenarioConfig(
        name="intersection",
        n_agents=n_agents,
        duration_s=15.0,
        target_speeds=speeds,
        observation_vehicles_count=n_agents,  # original: controlled vehicles only
        enable_background=False,
        intersection_second_row_offset_m=second_row_offset_m,
        nominal_speed_floor_index=None,  # tested (report #42): speed floor increased collisions -> off by default
        extra_env_config={
            "collision_reward": -10,
            "arrived_reward": 5,
            "high_speed_reward": 1,
            "right_lane_reward": 0.2,
            "on_road_reward": 0.2,
            "normalize_reward": False,
            "destination": "o1",  # highway-env default used implicitly by the original code
            "initial_vehicle_count": 0,
            "spawn_probability": 0.0,
            "offroad_terminal": False,
        },
    )


def configure_intersection_traffic(sc: ScenarioConfig, destinations: str = "fixed_o1", background_vehicles: int = 0,
                                   spawn_probability: float = 0.0, observation_vehicles_count: int = 10) -> ScenarioConfig:
    """Intersection traffic setting (report #57).  Background vehicles are stock highway-env IDM vehicles
    (``initial_vehicle_count`` at reset, ``spawn_probability`` per decision step); with background traffic the
    observation covers the nearest ``observation_vehicles_count`` vehicles (as in Roundabout / Highway)."""
    if sc.name != "intersection":
        raise ValueError("configure_intersection_traffic applies to the intersection testbed only")
    if destinations not in ("fixed_o1", "random_turn"):
        raise ValueError("destinations must be 'fixed_o1' or 'random_turn'")
    if background_vehicles < 0 or not 0.0 <= spawn_probability <= 1.0:
        raise ValueError("invalid background traffic setting")
    sc.intersection_destinations = destinations
    sc.extra_env_config["destination"] = "o1" if destinations == "fixed_o1" else None
    sc.extra_env_config["initial_vehicle_count"] = int(background_vehicles)
    sc.extra_env_config["spawn_probability"] = float(spawn_probability)
    sc.enable_background = background_vehicles > 0 or spawn_probability > 0.0
    sc.observation_vehicles_count = int(observation_vehicles_count) if sc.enable_background else sc.n_agents
    return sc


def roundabout_config(n_agents: int) -> ScenarioConfig:
    """Roundabout testbed, paper Table 2: 40 s, 10 vehicles in total.

    Stock ``roundabout-v0`` supports ONE controlled vehicle only; the revision provides
    ``MultiAgentRoundaboutEnv`` (see envs.py) which must be described in the paper.
    """
    return ScenarioConfig(
        name="roundabout",
        n_agents=n_agents,
        duration_s=40.0,
        target_speeds=(0.0, 8.0, 16.0),  # roundabout-v0 native
        observation_vehicles_count=10,
        enable_background=True,
        total_vehicles=10,
        extra_env_config={
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "lane_change_reward": -0.05,
            "normalize_reward": True,
        },
    )


def highway_config(n_agents: int) -> ScenarioConfig:
    """Highway testbed, paper Table 2: 10 s, 30 vehicles in total."""
    return ScenarioConfig(
        name="highway",
        n_agents=n_agents,
        duration_s=10.0,
        target_speeds=(20.0, 25.0, 30.0),  # highway-v0 native (MDPVehicle default)
        observation_vehicles_count=10,
        enable_background=True,
        total_vehicles=30,
        extra_env_config={
            "lanes_count": 4,
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "lane_change_reward": 0,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
        },
    )


SCENARIO_FACTORIES = {
    "intersection": intersection_config,
    "roundabout": roundabout_config,
    "highway": highway_config,
}


# ---------------------------------------------------------------------------
# Feature / safety configuration (physical units)
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    sensing_radius_m: float = 50.0
    density_radius_m: float = 30.0
    close_radius_m: float = 15.0
    crossing_heading_deg: float = 30.0
    crossing_radius_m: float = 40.0
    conflict_radius_m: float = 3.5  # predicted centre-to-centre distance counted as conflict
    conflict_horizon_s: float = 5.0
    ttc_cap_s: float = 10.0
    closing_speed_eps: float = 0.1
    # near-miss definition (used for labels and statistics)
    near_miss_ttc_s: float = 1.0
    near_miss_gap_m: float = 2.0


@dataclass
class ShieldConfig:
    """CBF-inspired / RSS-inspired DISCRETE feasibility screening parameters.

    These are screening rules for a discrete action set, not a formal CBF/RSS guarantee.
    """
    risk_threshold_delta: float = 0.5  # delta_t in Eq. (28); state-safety score threshold
    ttc_min_accelerate_s: float = 3.0
    ttc_min_maintain_s: float = 1.5
    ttc_min_lane_change_s: float = 3.0
    rear_ttc_min_s: float = 1.0
    rss_reaction_time_s: float = 0.5
    rss_a_max_accel: float = 3.0     # [m/s^2] worst-case accel of the rear vehicle during response
    rss_b_min_brake: float = 4.0     # [m/s^2] minimal (comfortable) braking of the rear vehicle
    rss_b_max_brake: float = 8.0     # [m/s^2] maximal braking of the front vehicle
    rss_min_gap_m: float = 2.0
    cbf_alpha: float = 0.5           # discrete-time barrier decay rate h_{k+1} >= (1-alpha) h_k
    comfort_accel: float = 3.0
    comfort_decel: float = 4.0
    lane_change_front_gap_m: float = 8.0
    lane_change_rear_gap_m: float = 8.0
    max_lateral_offset_m: float = 3.0  # |offset| beyond this -> only lane-keeping longitudinal actions
    speed_limit_tolerance: float = 1.0
    #: Deterministic conservative fallback ranking (Pi_safe).  Rationale: minimise kinetic
    #: energy first, avoid lateral manoeuvres next (keep-right before left), accelerate last.
    fallback_ranking: Tuple[str, ...] = (
        "DECELERATE",
        "MAINTAIN",
        "LANE_RIGHT",
        "LANE_LEFT",
        "ACCELERATE",
    )
    #: Explicit emergency action when A_hard(x) is empty.
    emergency_action: str = "DECELERATE"
    #: Pi_safe(a_nom; A_hard, s) variant.
    #: "topology_noop_to_maintain" (default): a LANE_LEFT/LANE_RIGHT nominal action for which no adjacent
    #:     lane exists (failed check lane_topology) is executed by highway-env as keep-lane/keep-speed, i.e.
    #:     it is physically identical to MAINTAIN; it is therefore judged exactly like MAINTAIN: mapped to
    #:     MAINTAIN when MAINTAIN is admissible and s >= delta_eff (reason "topology_noop"), otherwise the
    #:     fixed fallback_ranking applies.
    #:     Without this, the fixed ranking turned every such no-op into DECELERATE (42% of early-training
    #:     steps at Intersection N=4 -> vehicles stopped, 100% timeouts).
    #: "fixed_ranking": always use fallback_ranking (strict reading of the revision prompt).
    fallback_mode: str = "topology_noop_to_maintain"
    #: Risk gate of Pi_safe (report #58, Reviewer 2 comment 2):
    #:   "state"              : keep a_nom iff a_nom in A_hard(x) and s(x) >= delta_eff (first submission; the
    #:                          score does not depend on the action, so a low score always yields the fallback
    #:                          head -- a stopped vehicle then stays stopped: report #57, permanent freezing)
    #:   "action_conditioned" : s_a = f_NF(x'_a) on the one-step prediction x'_a of executing a (ego: first-order
    #:                          speed response to the new target speed; neighbours: constant velocity).  Keep a_nom iff
    #:                          a_nom in A_hard(x) and s_{a_nom} >= delta_eff; otherwise the first action of the
    #:                          fallback ranking with a in A_hard(x) and s_a >= delta_eff; if none, the fallback head.
    risk_mode: str = "state"
    speed_response_tau_s: float = 0.6  # ControlledVehicle speed controller: KP_A = 1/0.6 s^-1
    # online safety adaptation (only if flags.adapt_safety_layer_online)
    prune_every_episodes: int = 50
    attribute_prune_threshold: float = 0.1
    rule_prune_threshold: float = 0.05
    rss_min_gap_max_m: float = 6.0  # cap for online tightening


@dataclass
class NeuroFuzzyConfig:
    n_rules: int = 7
    label_horizon_steps: int = 3  # H for the state-risk supervision
    epochs: int = 200
    batch_size: int = 256
    lr: float = 5e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    patience: int = 20
    prune_every_epochs: int = 50  # structure pruning only at these epochs
    #: False (default) = maximum-likelihood BCE, so s estimates P(no safety event within H | x)
    #: and the thresholds delta_t / delta_temp keep a meaning on the score scale (its calibration
    #: is reported on the held-out calibration split C).  True = class-balanced BCE, which
    #: deliberately pulls scores towards 0.5 and makes any fixed delta trigger far more often.
    class_balanced_loss: bool = False


@dataclass
class SemanticConfig:
    tau_c: float = SEMANTIC_CONFIDENCE_THRESHOLD
    #: How an ADMITTED template's delta_temp (its min_safety_score) enters the SafetyShield threshold
    #: delta_eff = max(delta_t, delta_temp'):
    #:   "paper_max" : delta_temp' = delta_temp (first-version Algorithm 1)
    #:   "none"      : delta_temp' = 0 -- the admitted template changes only the nominal action (report #70: with
    #:                 delta_temp ~0.67 the shield stopped vehicles in 40% of agent-steps at Intersection N=2 -> 42-46%
    #:                 time-outs; selected on validation, ties -> simplest rule)
    #:   "cap"       : delta_temp' = min(delta_temp, delta_temp_cap)
    delta_temp_rule: str = "paper_max"
    delta_temp_cap: float = 1.0
    library_version: str = "paper_table1"  # or "legacy_code_v1" (the four templates of the original script)
    match_min_score: float = 0.5  # template matched iff best score > match_min_score
    lambda_template: float = 1.0 / 3.0
    lambda_match: float = 1.0 / 3.0
    lambda_validation: float = 1.0 / 3.0
    reliability_prior_alpha: float = 1.0  # Beta prior for validation reliability
    reliability_prior_beta: float = 1.0
    label_horizon_steps: int = 3  # H for semantic_acceptability
    #: "safety_only" (recommended by the revision prompt): y=1 iff a_sp in A_hard(x) and no
    #:     collision / off-road / near-miss of the agent in the next H decision steps.
    #: "safety_and_task_completion": additionally requires that the agent completes its task
    #:     objective in that calibration episode (a conservative recommendation that is safe but
    #:     leads to a stall/timeout is not 'acceptable').  Author decision; see report.
    label_definition: str = "safety_and_task_completion"
    calibration_method: str = "platt"  # "platt" (main) | "isotonic" (sensitivity)
    #: calibration behaviour policy: probability of executing a_sp in a matched context
    #: (otherwise a uniformly random primitive action).  1.0 made Intersection calibration
    #: rollouts collision-free (templates match ~80% of steps and recommend DECELERATE), so the
    #: state-risk model had no negative examples.
    calibration_execute_prob: float = 0.5
    #: behaviour on steps that do not execute a_sp: "uniform" | "cruise_epsilon" (see calibration_pipeline)
    calibration_behaviour: str = "cruise_epsilon"
    calibration_epsilon: float = 0.3
    min_calibration_samples: int = 50
    #: raw semantic confidence (report #53):
    #:   "conditional_reliability" (default): raw = l_T*conf_T + l_M*q(T, match level) + l_V*q(T), where the
    #:       retrieval-match term is the empirical reliability of template T AT THE OBSERVED MATCH LEVEL
    #:       (estimated on calibration split A).  The fraction of satisfied conditions itself is NOT used:
    #:       Table 1 conditions are risk indicators, so a fuller match means a riskier context, not a more
    #:       reliable template (report #51: it made the Platt slope negative).
    #:   "paper_eq22": raw = l_T*conf_T + l_M*match_fraction + l_V*q(T)  (first-submission Eq. 22; ablation)
    raw_confidence_mode: str = "conditional_reliability"
    #: calibration / refinement rollouts pass every nominal action through the deployment speed floor AND the
    #: SafetyShield hard-feasibility screening (A_hard, fallback ranking, topology no-op remap) -- the parts of
    #: the deployed pipeline that do not depend on the neuro-fuzzy model, which is fitted FROM these rollouts.
    #: The NF risk gate and the coordination layer are not applied (report #52).  False = unshielded rollouts.
    rollout_hard_shield: bool = True
    #: pre-deployment library refinement (report #54).  Runs on its own environment-seed range BEFORE
    #: calibration; the library is frozen afterwards.  0 rounds = use the initial library unchanged.
    refinement_rounds: int = 3
    refinement_episodes: int = 150
    #: "rule": deterministic; "llm": an LLM proposes the revision from the same statistics (offline,
    #: logged, validated, and subject to the same acceptance test in the next round)
    refinement_mode: str = "rule"
    refinement_z: float = 1.96          # two-sided 95% normal interval for the follow-vs-not benefit
    refinement_min_samples: int = 30    # per arm, per template, before any decision is taken


@dataclass
class CoordinationConfig:
    neighbour_radius_m: float = 40.0
    horizon_s: float = 3.0
    dt_s: float = 0.25
    conflict_distance_m: float = 3.5
    same_target_lane_gap_m: float = 15.0
    lane_change_duration_s: float = 2.0
    # priority P_i = w_r R_i + w_u U_i + w_g G_i - w_m M_i  (fixed; recorded in metadata)
    w_r: float = 0.4
    w_u: float = 0.2
    w_g: float = 0.3
    w_m: float = 0.1
    maneuver_aggressiveness: Dict[str, float] = field(
        default_factory=lambda: {
            "ACCELERATE": 1.0,
            "LANE_LEFT": 0.75,
            "LANE_RIGHT": 0.75,
            "MAINTAIN": 0.25,
            "DECELERATE": 0.0,
        }
    )
    reference_speed_mps: float = 10.0  # route-urgency normalisation; make_config sets it per scenario


@dataclass
class PPOConfig:
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatch: int = 32
    buffer_size: int = 256
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    #: Dropout makes pi_old and pi_new different stochastic functions of the same weights
    #: and breaks the importance ratio; the revision disables it (see report).
    policy_dropout: float = 0.0
    #: "shared" (original script): every agent is trained on the mean team reward.
    #: "individual": each agent is trained on its own per-agent reward (standard IPPO credit assignment).
    #: Reward values are identical in both modes; only their assignment to agents differs (report #44).
    reward_mode: str = "shared"
    pathnet_layers: int = 4
    pathnet_modules_per_layer: int = 15
    pathnet_active_modules: int = 5
    pathnet_hidden: int = 256
    # optional curriculum auxiliary objective (flag use_curriculum_aux)
    beta_aux: float = 0.1
    curriculum_ratio: float = 0.3
    curriculum_buffer_size: int = 2000


@dataclass
class MetaConfig:
    total_meta_steps: int = 100            # TOTAL_PATHNET_EPISODES
    marl_episodes_per_meta_step: int = 50  # MARL_EPISODES_PER_PATHNET_STEP
    transitions_per_meta_step: int = 1
    batch_size: int = 32                   # was 128 > 100 transitions -> never updated
    updates_per_meta_step: int = 1
    lr_actor: float = 1e-3                 # (eta_pi, eta_Q) = (1e-3, 3e-4), Appendix A
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_temperature: float = 0.2
    replay_size: int = 5000
    reward_scale: float = 1000.0
    deterministic_after_fraction: float = 0.8
    critic_dropout: float = 0.0


#: Meta-action dimensions and ranges.  Only dimensions that are CONSUMED by training
#: computation are allowed (see meta.build_meta_action_space).
META_ACTION_RANGES: Dict[str, Tuple[float, float]] = {
    "lr_actor_factor": (0.2, 2.0),
    "lr_critic_factor": (0.2, 2.0),
    "entropy_coef": (0.0005, 0.05),
    "beta_aux": (0.05, 0.5),          # active only if flags.use_curriculum_aux
    "curriculum_ratio": (0.1, 0.5),   # active only if flags.use_curriculum_aux
}

#: Dimensions of the original 9-D meta action that never reached any computation.
REMOVED_INACTIVE_META_DIMS: Tuple[str, ...] = (
    "focus_intensity",
    "update_ratio",
    "exploration_bonus",
    "safety_weight",
)


@dataclass
class SeedConfig:
    seed: int = 0
    # Disjoint environment-seed ranges; asserted disjoint at runtime.
    calibration_offset: int = 10_000_000
    refinement_offset: int = 50_000_000
    train_offset: int = 20_000_000
    validation_offset: int = 30_000_000
    test_offset: int = 40_000_000
    range_width: int = 1_000_000
    # True: validation and test episodes use the SAME environment seeds for every training seed and every
    # method (a fixed benchmark set; common random numbers), so seed-to-seed spread reflects the learned
    # policies rather than different traffic draws. Still disjoint from calibration/train seeds.
    common_eval_set: bool = True

    def env_seed(self, split: str, episode: int) -> int:
        offset = {
            "calibration": self.calibration_offset,
            "refinement": self.refinement_offset,
            "train": self.train_offset,
            "validation": self.validation_offset,
            "test": self.test_offset,
        }[split]
        if not 0 <= episode < self.range_width // 100:
            raise ValueError("episode index out of the reserved seed range")
        if self.common_eval_set and split in ("validation", "test"):
            return offset + episode
        return offset + self.seed * (self.range_width // 100) + episode


@dataclass
class RunConfig:
    calibration_episodes: int = 200
    evaluation_episodes: int = 100
    # Model selection (applied identically to every method): "final" = policy after the last meta-step;
    # "best_validation" = the checkpoint with the highest success rate on the held-out VALIDATION split
    # (ties: lower collision, then the later checkpoint). The test split is never used for selection.
    model_selection: str = "final"
    selection_interval_meta_steps: int = 10
    selection_episodes: int = 20
    device: str = "cpu"
    output_dir: str = "runs"


@dataclass
class PnALConfig:
    scenario: ScenarioConfig
    flags: ExperimentFlags = field(default_factory=ExperimentFlags)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    neuro_fuzzy: NeuroFuzzyConfig = field(default_factory=NeuroFuzzyConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    coordination: CoordinationConfig = field(default_factory=CoordinationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    run: RunConfig = field(default_factory=RunConfig)
    preset_name: str = "full_pnal"

    def validate(self) -> None:
        self.flags.validate()
        sm = self.semantic
        if sm.delta_temp_rule not in ("paper_max", "none", "cap") or not 0.0 <= sm.delta_temp_cap <= 1.0:
            raise ValueError("semantic.delta_temp_rule must be 'paper_max', 'none' or 'cap' (cap in [0, 1])")
        if sm.raw_confidence_mode not in ("conditional_reliability", "paper_eq22"):
            raise ValueError("semantic.raw_confidence_mode must be 'conditional_reliability' or 'paper_eq22'")
        if sm.refinement_mode not in ("rule", "llm"):
            raise ValueError("semantic.refinement_mode must be 'rule' or 'llm'")
        if sm.refinement_rounds < 0 or sm.refinement_episodes < 1 or sm.refinement_min_samples < 1 or sm.refinement_z <= 0:
            raise ValueError("invalid semantic refinement settings")
        if self.run.model_selection not in ("final", "best_validation"):
            raise ValueError("run.model_selection must be 'final' or 'best_validation'")
        if self.run.selection_interval_meta_steps < 1 or self.run.selection_episodes < 1:
            raise ValueError("selection interval and episodes must be >= 1")
        if abs(self.semantic.tau_c - SEMANTIC_CONFIDENCE_THRESHOLD) > 1e-12:
            # tau_c may be varied in a sensitivity study, but only through this field.
            pass
        lam = (
            self.semantic.lambda_template
            + self.semantic.lambda_match
            + self.semantic.lambda_validation
        )
        if abs(lam - 1.0) > 1e-9 or min(
            self.semantic.lambda_template,
            self.semantic.lambda_match,
            self.semantic.lambda_validation,
        ) < 0:
            raise ValueError("lambda weights must be non-negative and sum to 1 (paper Eq. 23)")
        if set(self.shield.fallback_ranking) != set(CANONICAL_ACTIONS):
            raise ValueError("fallback ranking must be a permutation of the canonical actions")
        if self.shield.emergency_action not in CANONICAL_ACTIONS:
            raise ValueError("emergency action must be a canonical action")
        if self.shield.fallback_mode not in ("topology_noop_to_maintain", "fixed_ranking"):
            raise ValueError("unknown shield fallback_mode")
        if self.shield.risk_mode not in ("state", "action_conditioned"):
            raise ValueError("shield.risk_mode must be 'state' or 'action_conditioned'")
        if self.shield.speed_response_tau_s <= 0:
            raise ValueError("shield.speed_response_tau_s must be > 0")
        if self.scenario.policy_action_set is not None:
            bad = [a for a in self.scenario.policy_action_set if a not in CANONICAL_ACTIONS]
            if bad or not self.scenario.policy_action_set:
                raise ValueError(f"invalid policy_action_set {self.scenario.policy_action_set}")
        if self.scenario.nominal_speed_floor_index is not None and not (
            0 <= self.scenario.nominal_speed_floor_index < len(self.scenario.target_speeds)
        ):
            raise ValueError("nominal_speed_floor_index outside target_speeds")
        if self.ppo.reward_mode not in ("shared", "individual"):
            raise ValueError("ppo.reward_mode must be 'shared' or 'individual'")
        if self.ppo.policy_dropout != 0.0:
            raise ValueError(
                "policy_dropout > 0 makes old/new log-probabilities inconsistent in PPO; "
                "set it to 0.0 (see revision report, item C)."
            )
        meta_total = self.meta.total_meta_steps * self.meta.transitions_per_meta_step
        if self.flags.use_training_adaptation and meta_total < self.meta.batch_size:
            raise ValueError(
                f"meta replay can hold at most {meta_total} transitions < batch_size "
                f"{self.meta.batch_size}; the SAC meta learner would never update"
            )
        starts = sorted(
            [
                self.seeds.calibration_offset,
                self.seeds.refinement_offset,
                self.seeds.train_offset,
                self.seeds.validation_offset,
                self.seeds.test_offset,
            ]
        )
        for a, b in zip(starts, starts[1:]):
            if b - a < self.seeds.range_width:
                raise ValueError("environment seed ranges overlap")

    def to_dict(self) -> Dict:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def make_config(scenario: str, n_agents: int, preset: str = "full_pnal", seed: int = 0, **kw) -> PnALConfig:
    if preset not in ABLATION_PRESETS:
        raise KeyError(f"unknown preset {preset}; choose from {sorted(ABLATION_PRESETS)}")
    sc = SCENARIO_FACTORIES[scenario](n_agents, **kw)
    cfg = PnALConfig(scenario=sc, flags=ABLATION_PRESETS[preset], preset_name=preset)
    cfg.seeds.seed = seed
    if scenario == "intersection":
        cfg.coordination.reference_speed_mps = float(max(sc.target_speeds))
    if scenario == "highway":
        cfg.coordination.reference_speed_mps = 25.0
    elif scenario == "roundabout":
        cfg.coordination.reference_speed_mps = 12.0
    cfg.validate()
    return cfg


# ===========================
# Canonical Action Semantics: 规范动作名 <-> 环境真实整数 id
# ===========================
# Canonical action semantics <-> environment integer ids.
#
# The algorithm only ever manipulates the canonical names in ``CANONICAL_ACTIONS``.
# The integer ids of the running environment are read from the live action type
# (``DiscreteMetaAction.actions_indexes``) after the environment is constructed.
# Nothing outside this module (and the env wrapper that calls it) may use integers.

CANONICAL_INDEX: Dict[str, int] = {a: i for i, a in enumerate(CANONICAL_ACTIONS)}


class ActionProtocolError(RuntimeError):
    """Raised when a testbed cannot provide the five-action protocol declared in the paper."""


@dataclass(frozen=True)
class ActionMap:
    canonical_to_env: Dict[str, int]
    env_to_canonical: Dict[int, str]
    env_labels: Dict[int, str]
    target_speeds: Tuple[float, ...]
    longitudinal: bool
    lateral: bool

    @classmethod
    def from_action_type(cls, action_type) -> "ActionMap":
        """Build the map from a live highway-env ``DiscreteMetaAction`` instance."""
        actions = getattr(action_type, "actions", None)
        indexes = getattr(action_type, "actions_indexes", None)
        if actions is None or indexes is None:
            raise ActionProtocolError(
                f"action type {type(action_type).__name__} exposes no discrete meta-actions"
            )
        c2e: Dict[str, int] = {}
        missing = []
        for name in CANONICAL_ACTIONS:
            label = CANONICAL_TO_HIGHWAY_LABEL[name]
            if label not in indexes:
                missing.append(f"{name}(env label '{label}')")
            else:
                c2e[name] = int(indexes[label])
        if missing:
            raise ActionProtocolError(
                "The environment action space does not contain the paper's five-action "
                f"protocol. Missing: {missing}. Available labels: {dict(actions)}. "
                "Update Section 5.1 / Table 2 of the paper or enable lateral+longitudinal actions."
            )
        e2c = {v: k for k, v in c2e.items()}
        if len(e2c) != len(c2e):
            raise ActionProtocolError("two canonical actions map to the same env id")
        return cls(
            canonical_to_env=c2e,
            env_to_canonical=e2c,
            env_labels={int(k): str(v) for k, v in actions.items()},
            target_speeds=tuple(float(s) for s in getattr(action_type, "target_speeds", ())),
            longitudinal=bool(getattr(action_type, "longitudinal", True)),
            lateral=bool(getattr(action_type, "lateral", True)),
        )

    # ------------------------------------------------------------------
    def to_env(self, name: str) -> int:
        if name not in self.canonical_to_env:
            raise KeyError(f"unknown canonical action {name!r}")
        return self.canonical_to_env[name]

    def to_env_tuple(self, names: Sequence[str]) -> Tuple[int, ...]:
        return tuple(self.to_env(n) for n in names)

    def from_env(self, env_id: int) -> str:
        return self.env_to_canonical[int(env_id)]

    @property
    def n(self) -> int:
        return len(CANONICAL_ACTIONS)

    def assert_valid(self, names: Iterable[str], where: str) -> None:
        for n in names:
            if n not in self.canonical_to_env:
                raise ActionProtocolError(f"{where}: action {n!r} is not available in the env")

    def metadata(self) -> Dict:
        return {
            "canonical_to_env": dict(self.canonical_to_env),
            "env_labels": {str(k): v for k, v in self.env_labels.items()},
            "target_speeds": list(self.target_speeds),
            "longitudinal": self.longitudinal,
            "lateral": self.lateral,
        }


def index_to_name(idx: int) -> str:
    return CANONICAL_ACTIONS[int(idx)]


def name_to_index(name: str) -> int:
    return CANONICAL_INDEX[name]


def order_by_ranking(actions: Iterable[str], ranking: Sequence[str]) -> List[str]:
    s = set(actions)
    return [a for a in ranking if a in s]


def apply_nominal_speed_floor(nominal: str, speed_index: int, floor_index, left_lane_exists: bool,
                              right_lane_exists: bool) -> Tuple[str, bool]:
    """Nominal-speed floor (report #45/#52), shared by deployment (fusion) and the pre-deployment
    calibration / refinement rollouts so that both label exactly the nominal action that is deployed.

    The nominal action (learned proposal or admitted semantic prior) may not command a speed below
    ``target_speeds[floor_index]``; standstill is left to the SafetyShield / coordination fallback.
    Returns (nominal_after_floor, floor_applied)."""
    if floor_index is None:
        return nominal, False
    idx, f = int(speed_index), int(floor_index)
    lateral_noop = (nominal == "LANE_LEFT" and not left_lane_exists) or (nominal == "LANE_RIGHT" and not right_lane_exists)
    if idx < f and (nominal in ("MAINTAIN", "DECELERATE") or lateral_noop):
        return "ACCELERATE", True  # resume towards the floor (the shield still checks it)
    if idx <= f and nominal == "DECELERATE":
        return "MAINTAIN", True    # the nominal action cannot command a speed below the floor
    return nominal, False


# ===========================
# Environment Wrapper: 三个 testbed、多智能体子类、episode outcome
# ===========================
# Environment interface layer.
#
# Responsibilities
# ----------------
# * build the three testbeds with EXPLICIT action/observation configuration;
# * read the live action map (``ActionMap``) and check the five-action protocol;
# * keep ``terminated`` and ``truncated`` separate;
# * track per-agent collision / off-road / arrival (latched) and expose
#   ``evaluate_task_outcome(env)`` with mutually exclusive outcomes;
# * provide scenario geometry (conflict zone, route progress, right-of-way) to the
#   feature extractor, the SafetyShield and the coordination layer.
#
# Custom multi-agent subclasses
# -----------------------------
# Stock highway-env (v1.10.x) supports several controlled vehicles in Intersection and
# Highway, but (i) ``roundabout-v0`` creates exactly ONE ego vehicle and (ii) ``highway-v0``
# rewards / terminates on ``controlled_vehicles[0]`` only.  ``MultiAgentRoundaboutEnv`` and
# ``MultiAgentHighwayEnv`` below apply the stock per-vehicle reward formula to every
# controlled vehicle and average it (exactly what stock ``IntersectionEnv`` does).  These
# testbed definitions must be stated in the paper and used for ALL compared methods.

OBS_FEATURES = ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]


# ---------------------------------------------------------------------------
# Custom multi-agent testbeds
# ---------------------------------------------------------------------------
class MultiAgentHighwayEnv(HighwayEnv):
    """highway-v0 with the stock reward applied to every controlled vehicle.

    ``config["background_collisions"] = False`` reproduces the execution shortcut of highway-env's
    ``highway-fast-v0``: uncontrolled vehicles stop checking collisions among themselves (every collision that involves
    a controlled vehicle is still detected, because the controlled vehicle keeps its own check)."""

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        if not self.config.get("background_collisions", True):
            for v in self.road.vehicles:
                if v not in self.controlled_vehicles:
                    v.check_collisions = False

    def _agent_rewards(self, action, vehicle) -> Dict[str, float]:
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = (
            vehicle.target_lane_index[2]
            if isinstance(vehicle, ControlledVehicle)
            else vehicle.lane_index[2]
        )
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": float(np.clip(scaled_speed, 0, 1)),
            "on_road_reward": float(vehicle.on_road),
        }

    def _agent_reward(self, action, vehicle) -> float:
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(self.config.get(name, 0) * r for name, r in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        return reward * rewards["on_road_reward"]

    def _reward(self, action) -> float:
        return float(np.mean([self._agent_reward(action, v) for v in self.controlled_vehicles]))

    def _rewards(self, action) -> Dict[str, float]:
        per = [self._agent_rewards(action, v) for v in self.controlled_vehicles]
        return {k: float(np.mean([p[k] for p in per])) for k in per[0]}

    def _is_terminated(self) -> bool:
        return any(v.crashed for v in self.controlled_vehicles) or (
            self.config["offroad_terminal"] and any(not v.on_road for v in self.controlled_vehicles)
        )

    def _info(self, obs, action=None) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(self._agent_reward(action, v) for v in self.controlled_vehicles)
        info["agents_terminated"] = tuple(bool(v.crashed) for v in self.controlled_vehicles)
        return info


class MultiAgentHighwayFastEnv(MultiAgentHighwayEnv):
    """highway-fast-v0 execution on the multi-agent Highway: uncontrolled vehicles skip collision checks among
    themselves (as stock ``HighwayEnvFast``); simulation frequency is set by the scenario config."""

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


class TestbedIncompatibilityError(RuntimeError):
    """The stock simulator cannot realise the protocol declared in the paper (never patched silently)."""


class StaggeredIntersectionEnv(IntersectionEnv):
    """intersection-v0 with a second spawn row for controlled vehicles k >= 4.

    Stock ``IntersectionEnv._make_vehicles`` (v1.10.1) places controlled vehicle k on approach
    ``k mod 4`` at 60 + 5*N(1,1) m and then deletes EVERY vehicle within 20 m of the new ego
    from ``road.vehicles`` -- including previously created CONTROLLED vehicles.  With N = 5,
    vehicle 4 shares approach o0 with vehicle 0, so vehicle 0 is removed from the road in
    practically every episode (50/50 in our probe): it stays in ``controlled_vehicles`` but is
    never simulated (frozen, cannot collide, cannot arrive).

    This subclass is used ONLY when ``ScenarioConfig.intersection_second_row_offset_m`` is set
    explicitly.  It is identical to the stock method (same RNG call sequence) except that
    (i) vehicles k >= 4 are spawned ``offset * (k // 4)`` metres further upstream and
    (ii) the early-collision clean-up never deletes controlled vehicles.  It changes the
    testbed and must be described in the paper (see compatibility report).
    """

    second_row_offset_m: float = 30.0
    #: report #57: exit = (entry + k) mod 4, k ~ U{1,2,3} (one RNG call, as the stock destination draw)
    random_turn: bool = False

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (self.road.act(), self.road.step(1 / self.config["simulation_frequency"]))
                for _ in range(self.config["simulation_frequency"])
            ]
        self._spawn_vehicle(60, spawn_probability=1, go_straight=True, position_deviation=0.1, speed_deviation=0)
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane((f"o{ego_id % 4}", f"ir{ego_id % 4}", 0))
            if self.random_turn:
                destination = "o" + str((ego_id % 4 + int(self.np_random.integers(1, 4))) % 4)
            else:
                destination = self.config["destination"] or "o" + str(self.np_random.integers(1, 4))
            longitudinal = 60 + 5 * self.np_random.normal(1) - self.second_row_offset_m * (ego_id // 4)
            ego_vehicle = self.action_type.vehicle_class(
                self.road, ego_lane.position(longitudinal, 0), speed=ego_lane.speed_limit, heading=ego_lane.heading_at(60)
            )
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass
            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in list(self.road.vehicles):  # prevent early collisions, but never delete a controlled vehicle
                if (
                    v is not ego_vehicle
                    and v not in self.controlled_vehicles
                    and np.linalg.norm(v.position - ego_vehicle.position) < 20
                ):
                    self.road.vehicles.remove(v)


ROUNDABOUT_RING_NODES = {"se", "ex", "ee", "nx", "ne", "wx", "we", "sx"}
# entry lane (start node) and the destination exit (opposite side, "straight through")
ROUNDABOUT_ENTRIES = [
    (("ser", "ses", 0), "nxr"),
    (("eer", "ees", 0), "wxr"),
    (("ner", "nes", 0), "sxr"),
    (("wer", "wes", 0), "exr"),
]


class MultiAgentRoundaboutEnv(RoundaboutEnv):
    """roundabout-v0 extended to N controlled vehicles (entries S, E, N, W, repeated).

    Background vehicles are spawned on the ring until ``total_vehicles`` is reached.
    Reward: stock roundabout per-vehicle formula, averaged over controlled vehicles.
    Termination: any controlled vehicle crashed, or all controlled vehicles arrived.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({"controlled_vehicles": 2, "total_vehicles": 10, "exit_distance": 20.0})
        return cfg

    def _make_vehicles(self) -> None:
        other_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        points: List[np.ndarray] = []
        for k in range(self.config["controlled_vehicles"]):
            lane_index, dest = ROUNDABOUT_ENTRIES[k % 4]
            lane = self.road.network.get_lane(lane_index)
            longitudinal = min(lane.length - 5.0, 115.0) - 30.0 * (k // 4) + self.np_random.normal() * 2.0
            v = self.action_type.vehicle_class(
                self.road, lane.position(longitudinal, 0.0), speed=8.0, heading=lane.heading_at(longitudinal)
            )
            v.plan_route_to(dest)
            self.road.vehicles.append(v)
            self.controlled_vehicles.append(v)
            points.append(v.position.copy())
        destinations = ["exr", "sxr", "nxr", "wxr"]
        ring = [("se", "ex"), ("ex", "ee"), ("ee", "nx"), ("nx", "ne"), ("ne", "wx"), ("wx", "we"), ("we", "sx"), ("sx", "se")]
        n_background = max(0, int(self.config["total_vehicles"]) - len(self.controlled_vehicles))
        tries = 0
        while n_background > 0 and tries < 200:
            tries += 1
            f, t = ring[int(self.np_random.integers(0, len(ring)))]
            lane_id = int(self.np_random.integers(0, len(self.road.network.graph[f][t])))
            lane = self.road.network.get_lane((f, t, lane_id))
            s = float(self.np_random.uniform(0, lane.length))
            pos = lane.position(s, 0)
            if any(np.linalg.norm(pos - p) < 15.0 for p in points):
                continue
            v = other_type.make_on_lane(self.road, (f, t, lane_id), longitudinal=s, speed=14.0 + self.np_random.normal() * 2.0)
            v.plan_route_to(destinations[int(self.np_random.integers(0, 4))])
            v.randomize_behavior()
            self.road.vehicles.append(v)
            points.append(pos)
            n_background -= 1

    def has_arrived(self, vehicle, exit_distance: Optional[float] = None) -> bool:
        exit_distance = self.config["exit_distance"] if exit_distance is None else exit_distance
        f, t, _ = vehicle.lane_index
        return f.endswith("xs") and t.endswith("xr") and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def _agent_rewards(self, action_i, vehicle) -> Dict[str, float]:
        lane_change_ids = (
            self.action_type.agents_action_types[0].actions_indexes["LANE_LEFT"],
            self.action_type.agents_action_types[0].actions_indexes["LANE_RIGHT"],
        )
        return {
            "collision_reward": float(vehicle.crashed),
            "high_speed_reward": MDPVehicle.get_speed_index(vehicle) / max(len(vehicle.target_speeds) - 1, 1),
            "lane_change_reward": float(action_i in lane_change_ids),
            "on_road_reward": float(vehicle.on_road),
        }

    def _agent_reward(self, action_i, vehicle) -> float:
        rewards = self._agent_rewards(action_i, vehicle)
        reward = sum(self.config.get(name, 0) * r for name, r in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["high_speed_reward"]], [0, 1])
        return reward * rewards["on_road_reward"]

    def _per_agent_actions(self, action):
        if isinstance(action, (tuple, list)):
            return list(action)
        return [action] * len(self.controlled_vehicles)

    def _reward(self, action) -> float:
        acts = self._per_agent_actions(action)
        return float(np.mean([self._agent_reward(a, v) for a, v in zip(acts, self.controlled_vehicles)]))

    def _rewards(self, action) -> Dict[str, float]:
        acts = self._per_agent_actions(action)
        per = [self._agent_rewards(a, v) for a, v in zip(acts, self.controlled_vehicles)]
        return {k: float(np.mean([p[k] for p in per])) for k in per[0]}

    def _is_terminated(self) -> bool:
        return any(v.crashed for v in self.controlled_vehicles) or all(
            self.has_arrived(v) for v in self.controlled_vehicles
        )

    def _info(self, obs, action=None) -> dict:
        info = super()._info(obs, action)
        acts = self._per_agent_actions(action)
        info["agents_rewards"] = tuple(self._agent_reward(a, v) for a, v in zip(acts, self.controlled_vehicles))
        info["agents_terminated"] = tuple(
            bool(v.crashed or self.has_arrived(v)) for v in self.controlled_vehicles
        )
        return info


# ---------------------------------------------------------------------------
# Scenario geometry models
# ---------------------------------------------------------------------------
class ScenarioModel:
    name = "base"

    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg

    def has_arrived(self, raw_env, vehicle, traveled: float) -> bool:
        raise NotImplementedError

    def is_arrival_lane(self, lane_index) -> bool:
        return False

    def arrival_offset(self) -> float:
        return 0.0

    def conflict_zone(self, raw_env, vehicle) -> Tuple[float, bool]:
        return math.inf, False

    def rule_priority(self, raw_env, vehicle) -> float:
        return 0.0

    def intends_merge_or_crossing(self, raw_env, vehicle) -> bool:
        d, in_zone = self.conflict_zone(raw_env, vehicle)
        return in_zone or math.isfinite(d)

    def route_remaining(self, raw_env, vehicle, traveled: float) -> float:
        """Distance [m] still to travel until the task objective along the planned route."""
        net = raw_env.road.network
        lane_index = vehicle.lane_index
        s = vehicle.lane.local_coordinates(vehicle.position)[0]
        if self.is_arrival_lane(lane_index):
            return max(0.0, self.arrival_offset() - s)
        remaining = max(0.0, vehicle.lane.length - s)
        route = list(vehicle.route or [])
        idx = next((k for k, r in enumerate(route) if r[0] == lane_index[0] and r[1] == lane_index[1]), None)
        if idx is None:
            return remaining + self.arrival_offset()
        for r in route[idx + 1 :]:
            li = (r[0], r[1], r[2] if r[2] is not None else 0)
            if self.is_arrival_lane(li):
                return remaining + self.arrival_offset()
            remaining += net.get_lane(li).length
        return remaining + self.arrival_offset()


class IntersectionModel(ScenarioModel):
    name = "intersection"

    def is_arrival_lane(self, lane_index) -> bool:
        return lane_index[0].startswith("il") and lane_index[1].startswith("o")

    def arrival_offset(self) -> float:
        return self.cfg.intersection_exit_distance_m

    def has_arrived(self, raw_env, vehicle, traveled: float) -> bool:
        return bool(raw_env.has_arrived(vehicle, exit_distance=self.cfg.intersection_exit_distance_m))

    def conflict_zone(self, raw_env, vehicle) -> Tuple[float, bool]:
        f, t, _ = vehicle.lane_index
        if f.startswith("ir"):
            return 0.0, True
        if f.startswith("o") and t.startswith("ir"):
            s = vehicle.lane.local_coordinates(vehicle.position)[0]
            return max(0.0, vehicle.lane.length - s), False
        return math.inf, False

    def rule_priority(self, raw_env, vehicle) -> float:
        return float(getattr(vehicle.lane, "priority", 0)) / 3.0


class RoundaboutModel(ScenarioModel):
    name = "roundabout"

    def is_arrival_lane(self, lane_index) -> bool:
        return lane_index[0].endswith("xs") and lane_index[1].endswith("xr")

    def arrival_offset(self) -> float:
        return self.cfg.roundabout_exit_distance_m

    def has_arrived(self, raw_env, vehicle, traveled: float) -> bool:
        return bool(raw_env.has_arrived(vehicle, exit_distance=self.cfg.roundabout_exit_distance_m))

    def conflict_zone(self, raw_env, vehicle) -> Tuple[float, bool]:
        f, t, _ = vehicle.lane_index
        if f in ROUNDABOUT_RING_NODES and t in ROUNDABOUT_RING_NODES:
            return 0.0, True
        s = vehicle.lane.local_coordinates(vehicle.position)[0]
        net = raw_env.road.network
        if f.endswith("er"):  # straight access lane, followed by the sine entry lane
            nxt = list(net.graph[t].keys())[0]
            return max(0.0, vehicle.lane.length - s) + net.get_lane((t, nxt, 0)).length, False
        if f.endswith("es"):
            return max(0.0, vehicle.lane.length - s), False
        return math.inf, False

    def rule_priority(self, raw_env, vehicle) -> float:
        _, in_zone = self.conflict_zone(raw_env, vehicle)
        return 1.0 if in_zone else 0.0  # vehicles on the ring have right of way


class HighwayModel(ScenarioModel):
    name = "highway"

    def has_arrived(self, raw_env, vehicle, traveled: float) -> bool:
        """Highway has no destination: the objective is to drive the FULL horizon and cover at
        least ``highway_target_distance_m``.  The objective is therefore only completed at the
        time limit, so collisions and off-road events at any time before it are counted."""
        at_horizon = float(raw_env.time) >= float(raw_env.config["duration"]) - 1e-9
        return at_horizon and traveled >= self.cfg.highway_target_distance_m

    def intends_merge_or_crossing(self, raw_env, vehicle) -> bool:
        return False

    def route_remaining(self, raw_env, vehicle, traveled: float) -> float:
        return max(0.0, self.cfg.highway_target_distance_m - traveled)


SCENARIO_MODELS = {"intersection": IntersectionModel, "roundabout": RoundaboutModel, "highway": HighwayModel}


# ---------------------------------------------------------------------------
# Raw env construction
# ---------------------------------------------------------------------------
def build_env_config(sc: ScenarioConfig) -> dict:
    cfg = {
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "vehicles_count": int(sc.observation_vehicles_count),
                "features": list(OBS_FEATURES),
                "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]},
                "absolute": False,
                "order": "sorted",
                "normalize": False,  # physical units: metres, m/s
                "see_behind": False,
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
                "longitudinal": bool(sc.longitudinal),
                "lateral": bool(sc.lateral),
                "target_speeds": list(sc.target_speeds),
            },
        },
        "controlled_vehicles": int(sc.n_agents),
        "duration": float(sc.duration_s),
        "policy_frequency": int(sc.policy_frequency),
        "simulation_frequency": int(sc.simulation_frequency),
    }
    cfg.update(sc.extra_env_config)
    if sc.name == "highway":
        cfg["vehicles_count"] = max(0, int(sc.total_vehicles or 30) - sc.n_agents)
        cfg["background_collisions"] = bool(sc.background_collisions)
    if sc.name == "roundabout":
        cfg["total_vehicles"] = int(sc.total_vehicles or 10)
        cfg["exit_distance"] = float(sc.roundabout_exit_distance_m)
    return cfg


def make_raw_env(sc: ScenarioConfig):
    cls = {"intersection": IntersectionEnv, "roundabout": MultiAgentRoundaboutEnv, "highway": MultiAgentHighwayEnv}[sc.name]
    if sc.name == "highway" and not sc.background_vehicle_collisions:
        cls = MultiAgentHighwayFastEnv
    elif not sc.background_vehicle_collisions:
        raise ValueError("background_vehicle_collisions=False is defined for the Highway testbed only")
    if sc.name == "intersection" and (sc.intersection_second_row_offset_m is not None or sc.intersection_destinations == "random_turn"):
        # same spawn code as the stock env for N <= 4 (offset only applies to vehicles k >= 4)
        cls = type("StaggeredIntersectionEnvConfigured", (StaggeredIntersectionEnv,),
                   {"second_row_offset_m": float(sc.intersection_second_row_offset_m or 0.0),
                    "random_turn": sc.intersection_destinations == "random_turn"})
    return cls(config=build_env_config(sc), render_mode=None)


# ---------------------------------------------------------------------------
# Outcome evaluation
# ---------------------------------------------------------------------------
OUTCOME_KEYS = ("success", "collision", "offroad", "timeout")


def outcome_from_flags(
    crashed: Sequence[bool], offroad: Sequence[bool], arrived: Sequence[bool], terminated: bool, truncated: bool
) -> Dict[str, bool]:
    """Mutually exclusive outcome with precedence collision > offroad > success > timeout.

    * success  : every controlled agent completed its task objective, no collision, no off-road;
    * collision: at least one controlled agent crashed;
    * offroad  : no collision and at least one controlled agent left the valid road region
                 before completing its own objective (see PnALEnv._update_latches);
    * timeout  : time-limit truncation without collision/off-road and objective NOT completed.
    A time-limit truncation is never counted as success by itself.
    """
    if not (terminated or truncated):
        raise RuntimeError("evaluate_task_outcome called before the episode ended")
    collision = any(crashed)
    off = (not collision) and any(offroad)
    success = (not collision) and (not off) and len(arrived) > 0 and all(arrived)
    timeout = (not collision) and (not off) and (not success) and bool(truncated)
    out = {"success": success, "collision": collision, "offroad": off, "timeout": timeout}
    out["incomplete"] = not any(out.values())  # terminated for another reason (should not happen)
    return out


def evaluate_task_outcome(env, terminated: Optional[bool] = None, truncated: Optional[bool] = None) -> Dict[str, bool]:
    """Unified outcome interface for Intersection, Roundabout and Highway.

    ``env`` is a ``PnALEnv`` (uses latched per-agent events tracked during the episode) or
    a raw highway-env env (uses the env's own ``has_arrived`` / crash / on-road state at the
    current step; ``terminated``/``truncated`` must then be passed).
    """
    if isinstance(env, PnALEnv):
        return env.evaluate_task_outcome()
    raw = getattr(env, "unwrapped", env)
    if terminated is None or truncated is None:
        raise ValueError("pass terminated/truncated for a raw environment")
    vs = raw.controlled_vehicles
    arrived = [bool(raw.has_arrived(v)) if hasattr(raw, "has_arrived") else False for v in vs]
    return outcome_from_flags([v.crashed for v in vs], [not v.on_road for v in vs], arrived, terminated, truncated)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    observations: List[np.ndarray]
    shared_reward: float
    agent_rewards: List[float]
    terminated: bool
    truncated: bool
    info: dict
    newly_crashed: List[bool] = field(default_factory=list)
    newly_offroad: List[bool] = field(default_factory=list)
    arrived_before: List[bool] = field(default_factory=list)  # per-agent arrival status BEFORE this step


class PnALEnv:
    """Multi-agent wrapper with explicit action semantics and outcome tracking."""

    def __init__(self, scenario: ScenarioConfig):
        self.scenario = scenario
        self.raw = make_raw_env(scenario)
        self.model: ScenarioModel = SCENARIO_MODELS[scenario.name](scenario)
        self.raw.reset(seed=0)
        self._check_controlled_on_road()
        types = getattr(self.raw.action_type, "agents_action_types", None)
        if not types:
            raise ActionProtocolError("expected a MultiAgentAction action type")
        self.action_map = ActionMap.from_action_type(types[0])
        for t in types[1:]:
            if ActionMap.from_action_type(t).canonical_to_env != self.action_map.canonical_to_env:
                raise ActionProtocolError("agents have different action maps")
        if len(types) != scenario.n_agents:
            raise ActionProtocolError(f"env created {len(types)} controlled vehicles, expected {scenario.n_agents}")
        if tuple(self.action_map.target_speeds) != tuple(float(s) for s in scenario.target_speeds):
            raise ActionProtocolError("env target speeds differ from the configured target speeds")
        self.n_agents = scenario.n_agents
        self.obs_dim = int(scenario.observation_vehicles_count * len(OBS_FEATURES))
        self._reset_tracking()

    # ------------------------------------------------------------------
    def _reset_tracking(self):
        n = self.scenario.n_agents
        self.crashed = [False] * n
        self.offroad = [False] * n
        self.arrived = [False] * n
        self.last_terminated = False
        self.last_truncated = False
        self.decision_step = 0
        self.start_x = [float(v.position[0]) for v in self.raw.controlled_vehicles] if self.raw.controlled_vehicles else [0.0] * n

    @property
    def vehicles(self):
        return self.raw.controlled_vehicles

    def traveled(self, i: int) -> float:
        return float(self.vehicles[i].position[0]) - self.start_x[i]

    def _remove_background(self):
        controlled = list(self.raw.controlled_vehicles)
        self.raw.road.vehicles = [v for v in self.raw.road.vehicles if v in controlled]

    def _observe(self) -> List[np.ndarray]:
        obs = self.raw.observation_type.observe()
        return [np.asarray(o, dtype=np.float32).reshape(-1) for o in obs]

    def _check_controlled_on_road(self) -> None:
        """Every controlled vehicle must be simulated.  Stock intersection-v0 with N >= 5 deletes
        controlled vehicle 0 from ``road.vehicles`` (see StaggeredIntersectionEnv); such a
        'ghost' agent never moves or collides and would silently corrupt every metric."""
        detached = [i for i, v in enumerate(self.raw.controlled_vehicles) if v not in self.raw.road.vehicles]
        if detached:
            raise TestbedIncompatibilityError(
                f"{self.scenario.name} N={self.scenario.n_agents}: controlled vehicle(s) {detached} were removed from "
                "the road by the simulator's spawn clean-up and would never be simulated. Stock intersection-v0 "
                "cannot host N>=5 controlled vehicles. Either restrict the paper to N<=4 or set "
                "ScenarioConfig.intersection_second_row_offset_m (CLI --intersection-second-row-offset) and "
                "describe the modified spawn protocol in Sec. 5.1 / Table 2 and the N=5 rows of Table 3."
            )

    def reset(self, seed: int) -> Tuple[List[np.ndarray], dict]:
        obs, info = self.raw.reset(seed=int(seed))
        self._check_controlled_on_road()
        if not self.scenario.enable_background:
            self._remove_background()
            obs = self.raw.observation_type.observe()  # re-observe AFTER removal (original obs was stale)
        self._reset_tracking()
        self._update_latches()
        return [np.asarray(o, dtype=np.float32).reshape(-1) for o in obs], info

    def _update_latches(self) -> Tuple[List[bool], List[bool]]:
        """Latch per-agent events.

        * collision: always counted (a crash is a safety event whenever it happens);
        * off-road : counted only while the agent has NOT yet completed its objective.  An agent
          that has already arrived keeps driving and eventually runs past the end of the modelled
          exit lane (127.5 m in Roundabout, ~100 m in Intersection), where highway-env reports
          ``on_road == False``; without this rule a successful episode was relabelled 'offroad'
          (observed in 1/40 random Roundabout episodes);
        * arrival  : latched the first time the scenario objective holds.
        """
        new_c, new_o = [], []
        for i, v in enumerate(self.vehicles):
            c = bool(v.crashed)
            o = (not bool(v.on_road)) and not self.arrived[i]
            new_c.append(c and not self.crashed[i])
            new_o.append(o and not self.offroad[i])
            self.crashed[i] |= c
            self.offroad[i] |= o
            if not self.arrived[i] and self.model.has_arrived(self.raw, v, self.traveled(i)):
                self.arrived[i] = True
        return new_c, new_o

    def step(self, canonical_actions: Sequence[str]) -> StepResult:
        if len(canonical_actions) != self.n_agents:
            raise ValueError("one action per controlled agent is required")
        for a in canonical_actions:
            if a not in CANONICAL_ACTIONS:
                raise ValueError(f"non-canonical action {a!r} reached the env interface")
        env_action = self.action_map.to_env_tuple(canonical_actions)
        arrived_before = list(self.arrived)
        obs, reward, terminated, truncated, info = self.raw.step(env_action)
        if not self.scenario.enable_background:
            self._remove_background()
        self.decision_step += 1
        new_c, new_o = self._update_latches()
        agent_rewards = [float(r) for r in info.get("agents_rewards", [reward] * self.n_agents)]
        if self.scenario.reward_arrival_once:
            # arrival reward paid once: an agent that had already arrived before this step earns 0
            agent_rewards = [0.0 if arrived_before[i] else r for i, r in enumerate(agent_rewards)]
            reward = float(np.mean(agent_rewards))
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)
        return StepResult(
            observations=[np.asarray(o, dtype=np.float32).reshape(-1) for o in obs],
            shared_reward=float(reward),
            agent_rewards=agent_rewards,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
            newly_crashed=new_c,
            newly_offroad=new_o,
            arrived_before=arrived_before,
        )

    def evaluate_task_outcome(self) -> Dict[str, bool]:
        return outcome_from_flags(self.crashed, self.offroad, self.arrived, self.last_terminated, self.last_truncated)

    # ------------------------------------------------------------------
    def compatibility_report(self) -> List[Dict[str, str]]:
        """Protocol issues that must be reflected in the paper (never silently patched)."""
        issues: List[Dict[str, str]] = []
        net = self.raw.road.network
        multi_lane = any(len(lanes) > 1 for to in net.graph.values() for lanes in to.values())
        if not multi_lane:
            issues.append(
                {
                    "testbed": self.scenario.name,
                    "issue": "LANE_LEFT/LANE_RIGHT exist in the action space but every road segment has a "
                    "single lane, so lateral actions are dynamically identical to MAINTAIN. The "
                    "SafetyShield marks them inadmissible (no adjacent lane).",
                    "paper_location": "Sec. 5.1 action set; Table 2 'Action space' row",
                }
            )
        if self.scenario.name == "roundabout":
            issues.append(
                {
                    "testbed": "roundabout",
                    "issue": "stock roundabout-v0 supports one controlled vehicle; results use the custom "
                    "MultiAgentRoundaboutEnv (entries S/E/N/W, straight-through destinations, averaged stock reward).",
                    "paper_location": "Sec. 5.1 testbed description; Table 2",
                }
            )
        if self.scenario.name == "highway":
            issues.append(
                {
                    "testbed": "highway",
                    "issue": "highway-v0 has no destination; success is defined as every controlled vehicle "
                    f"driving the full {self.scenario.duration_s} s horizon without collision/off-road and "
                    f"travelling >= {self.scenario.highway_target_distance_m} m (with target speeds >= 20 m/s the "
                    "distance criterion is almost always met, so success is essentially collision-free survival). "
                    "Stock reward/termination only consider controlled_vehicles[0]; MultiAgentHighwayEnv averages "
                    "the stock reward over all controlled vehicles and terminates on any crash.",
                    "paper_location": "Sec. 5.1 metric definitions; Table 2",
                }
            )
        if self.scenario.name == "intersection" and self.scenario.n_agents > 4:
            issues.append(
                {
                    "testbed": "intersection",
                    "issue": "stock intersection-v0 places controlled vehicle k on approach k mod 4 and its spawn "
                    "clean-up deletes every vehicle within 20 m of a new ego, including controlled vehicle 0 when "
                    "vehicle 4 spawns on the same approach (observed in 50/50 resets with v1.10.1). In the first-"
                    "version experiments agent 0 of every N=5 episode was therefore a frozen, non-colliding ghost. "
                    f"This run uses the explicit StaggeredIntersectionEnv (vehicles k>=4 spawned "
                    f"{self.scenario.intersection_second_row_offset_m} m further upstream; controlled vehicles never deleted).",
                    "paper_location": "Sec. 5.1 / Table 2 (spawn protocol); Table 3 N=5 rows must be re-run and re-described",
                }
            )
        return issues


# ===========================
# Observation Processing: 物理单位、ego 坐标系安全状态与符号上下文
# ===========================
# Physical-unit safety state x_i^t, NF feature vector and symbolic context c_i^t.
#
# Observation processing (revision item H)
# ----------------------------------------
# The Kinematics observation is configured with ``normalize=False`` and ``absolute=False``:
# row 0 is the ego vehicle in absolute world coordinates, rows 1.. are other vehicles with
# position / velocity DIFFERENCES in the WORLD frame (metres, m/s) and absolute headings.
# No ad-hoc scale factors are applied.  Relative quantities are rotated into the ego frame:
#
#     e_long = [cos h, sin h],  e_lat = [-sin h, cos h]
#     s = dp . e_long,  d = dp . e_lat,  v_s = dv . e_long
#
# A vehicle is *leading* iff s > 0 and |d| < lane_width / 2 (ego frame, not world x).
#
# Lane availability / adjacent-lane gaps / conflict-zone distance come from the road
# topology and simulator vehicle state (``road.network`` and ``road.neighbour_vehicles``).

VEHICLE_LENGTH = 5.0
N_F = len(OBS_FEATURES)


@dataclass
class OtherVehicleEgoFrame:
    s: float        # longitudinal position in ego frame [m]
    d: float        # lateral position in ego frame [m]
    v_s: float      # relative longitudinal speed (other - ego) [m/s]
    v_d: float      # relative lateral speed [m/s]
    dist: float     # centre distance [m]
    heading_diff: float  # |heading_other - heading_ego| in [0, pi]
    t_closest: float     # time of closest approach under constant velocity [s]
    d_closest: float     # predicted closest centre distance [m]


def ego_frame_from_observation(obs_flat: np.ndarray) -> Tuple[Dict[str, float], List[OtherVehicleEgoFrame]]:
    obs = np.asarray(obs_flat, dtype=np.float64).reshape(-1, N_F)
    ego = obs[0]
    cos_h, sin_h = float(ego[5]), float(ego[6])
    norm = math.hypot(cos_h, sin_h) or 1.0
    cos_h, sin_h = cos_h / norm, sin_h / norm
    e_long = np.array([cos_h, sin_h])
    e_lat = np.array([-sin_h, cos_h])
    ego_v = np.array([ego[3], ego[4]])
    ego_info = {
        "x": float(ego[1]),
        "y": float(ego[2]),
        "speed_long": float(ego_v @ e_long),
        "speed": float(np.linalg.norm(ego_v)),
        "heading": math.atan2(sin_h, cos_h),
    }
    others: List[OtherVehicleEgoFrame] = []
    for row in obs[1:]:
        if row[0] <= 0.5:  # presence
            continue
        dp = np.array([row[1], row[2]])
        dv = np.array([row[3], row[4]])
        h_other = math.atan2(row[6], row[5])
        dh = abs((h_other - ego_info["heading"] + math.pi) % (2 * math.pi) - math.pi)
        vv = float(dv @ dv)
        t_star = 0.0 if vv < 1e-9 else max(0.0, -float(dp @ dv) / vv)
        others.append(
            OtherVehicleEgoFrame(
                s=float(dp @ e_long),
                d=float(dp @ e_lat),
                v_s=float(dv @ e_long),
                v_d=float(dv @ e_lat),
                dist=float(np.linalg.norm(dp)),
                heading_diff=dh,
                t_closest=t_star,
                d_closest=float(np.linalg.norm(dp + dv * t_star)),
            )
        )
    return ego_info, others


@dataclass
class SafetyState:
    """x_i^t in physical units (inf = not applicable / nothing observed)."""

    ego_speed: float
    ego_target_speed: float
    speed_limit: float
    lane_offset: float
    lane_width: float
    lead_gap: float
    lead_closing_speed: float
    lead_speed_long: float
    ttc_lead: float
    ttc_conflict: float
    nearest_distance: float
    n_vehicles_density: int
    n_vehicles_close: int
    n_crossing: int
    approaching_interactor: bool
    left_lane_exists: bool
    right_lane_exists: bool
    left_front_gap: float
    left_rear_gap: float
    left_front_closing: float
    left_rear_closing: float
    right_front_gap: float
    right_rear_gap: float
    right_front_closing: float
    right_rear_closing: float
    rear_gap: float
    rear_closing_speed: float
    dist_to_conflict_zone: float
    in_conflict_zone: bool
    intends_merge_or_crossing: bool
    neighbor_has_priority: bool
    rule_priority: float
    route_remaining: float
    time_remaining: float
    speed_index: int
    n_target_speeds: int
    near_miss: bool
    on_road: bool
    next_target_speed: Optional[float] = None  # target speed after ACCELERATE (speed constraint)

    def left_space(self, cap: float) -> float:
        return min(self.left_front_gap, self.left_rear_gap, cap) if self.left_lane_exists else 0.0

    def right_space(self, cap: float) -> float:
        return min(self.right_front_gap, self.right_rear_gap, cap) if self.right_lane_exists else 0.0

    def as_dict(self) -> Dict:
        return asdict(self)


#: order of the neuro-fuzzy input features x_i^t (all physical, capped; standardised inside the NF)
NF_FEATURE_NAMES = (
    "lead_gap_m",
    "lead_closing_speed_mps",
    "ttc_lead_s",
    "ttc_conflict_s",
    "nearest_distance_m",
    "lane_offset_m",
    "ego_speed_mps",
    "n_vehicles_density",
    "left_lane_available",
    "right_lane_available",
    "left_lane_space_m",
    "right_lane_space_m",
)


@dataclass
class SymbolicContext:
    """c_i^t: typed symbolic variables referenced by semantic templates."""

    dist_to_conflict_zone: float
    in_conflict_zone: bool
    ttc_min: float
    n_crossing: int
    intends_merge_or_crossing: bool
    approaching_interactor: bool
    front_gap: float
    neighbor_has_priority: bool
    interaction_active: bool
    lead_slower: bool
    adjacent_lane_feasible: bool
    target_rear_gap: float
    n_vehicles_density: int
    n_vehicles_close: int
    lateral_gap_limited: bool
    left_space: float
    right_space: float

    NUMERIC = {
        "dist_to_conflict_zone",
        "ttc_min",
        "n_crossing",
        "front_gap",
        "target_rear_gap",
        "n_vehicles_density",
        "n_vehicles_close",
        "left_space",
        "right_space",
    }

    @classmethod
    def field_types(cls) -> Dict[str, str]:
        return {f.name: ("numeric" if f.name in cls.NUMERIC else "bool") for f in fields(cls)}

    def get(self, name: str):
        return getattr(self, name)


class SafetyStateExtractor:
    def __init__(self, env: PnALEnv, cfg: FeatureConfig):
        self.env = env
        self.cfg = cfg

    # ------------------------------------------------------------------
    def _adjacent(self, vehicle, lane_index, side: int):
        """(exists, front_gap, rear_gap, front_closing, rear_closing) for lane id + side."""
        net = self.env.raw.road.network
        f, t, i = lane_index
        j = i + side
        lanes = net.graph[f][t]
        if j < 0 or j >= len(lanes):
            return False, 0.0, 0.0, 0.0, 0.0
        target = (f, t, j)
        lane = net.get_lane(target)
        if not lane.is_reachable_from(vehicle.position):
            return False, 0.0, 0.0, 0.0, 0.0
        front, rear = self.env.raw.road.neighbour_vehicles(vehicle, target)
        s_ego = lane.local_coordinates(vehicle.position)[0]
        v_ego = vehicle.speed * math.cos(vehicle.heading - lane.heading_at(s_ego))
        # No vehicle in the target lane -> gap = +inf (NOT a phantom vehicle at the sensing
        # radius: at highway speeds the RSS distance exceeds 50 m, so a capped gap made every
        # lane change inadmissible even on an empty lane).  Vehicles beyond the sensing radius
        # are also treated as absent.  Capping to the sensing radius happens only in the NF
        # feature vector / symbolic context.
        fg = rg = math.inf
        fc = rc = 0.0
        R = self.cfg.sensing_radius_m
        if front is not None:
            s_f = lane.local_coordinates(front.position)[0]
            if s_f - s_ego - VEHICLE_LENGTH <= R:
                fg = s_f - s_ego - VEHICLE_LENGTH
                fc = v_ego - front.speed * math.cos(front.heading - lane.heading_at(s_f))
        if rear is not None:
            s_r = lane.local_coordinates(rear.position)[0]
            if s_ego - s_r - VEHICLE_LENGTH <= R:
                rg = s_ego - s_r - VEHICLE_LENGTH
                rc = rear.speed * math.cos(rear.heading - lane.heading_at(s_r)) - v_ego
        return True, float(fg), float(rg), float(fc), float(rc)

    def _rear_same_lane(self, vehicle) -> Tuple[float, float]:
        """(gap, closing speed) to the rear vehicle in the ego lane; (+inf, 0) if none within range."""
        lane = vehicle.lane
        _, rear = self.env.raw.road.neighbour_vehicles(vehicle, vehicle.lane_index)
        if rear is None:
            return math.inf, 0.0
        s_e = lane.local_coordinates(vehicle.position)[0]
        s_r = lane.local_coordinates(rear.position)[0]
        gap = s_e - s_r - VEHICLE_LENGTH
        if gap > self.cfg.sensing_radius_m:
            return math.inf, 0.0
        closing = rear.speed * math.cos(rear.heading - lane.heading_at(s_r)) - vehicle.speed * math.cos(
            vehicle.heading - lane.heading_at(s_e)
        )
        return float(gap), float(closing)

    def _neighbor_has_priority(self, i: int, my_priority: float) -> bool:
        """Some vehicle within 30 m is inside the conflict zone or has higher right-of-way."""
        me = self.env.vehicles[i]
        model = self.env.model
        for v in self.env.raw.road.vehicles:
            if v is me or np.linalg.norm(v.position - me.position) > self.cfg.density_radius_m:
                continue
            _, in_zone = model.conflict_zone(self.env.raw, v)
            if in_zone or model.rule_priority(self.env.raw, v) > my_priority + 1e-9:
                return True
        return False

    # ------------------------------------------------------------------
    def extract(self, i: int, obs_flat: np.ndarray) -> SafetyState:
        cfg = self.cfg
        env = self.env
        vehicle = env.vehicles[i]
        ego, others = ego_frame_from_observation(obs_flat)
        lane_width = float(vehicle.lane.width_at(vehicle.lane.local_coordinates(vehicle.position)[0]))
        _, lat = vehicle.lane.local_coordinates(vehicle.position)

        lead = [o for o in others if o.s > 0 and abs(o.d) < lane_width / 2.0]
        # no lead vehicle within the sensing radius -> gap +inf, lead speed +inf (checks skipped)
        lead = [o for o in lead if o.s - VEHICLE_LENGTH <= cfg.sensing_radius_m]
        lead_gap, lead_closing, lead_speed, ttc_lead = math.inf, 0.0, math.inf, math.inf
        if lead:
            L = min(lead, key=lambda o: o.s)
            lead_gap = L.s - VEHICLE_LENGTH
            lead_closing = -L.v_s
            lead_speed = ego["speed_long"] + L.v_s
            if lead_closing > cfg.closing_speed_eps:
                ttc_lead = max(0.0, lead_gap) / lead_closing
        ttc_conflict = math.inf
        for o in others:
            if o.dist < cfg.conflict_radius_m:
                ttc_conflict = 0.0
            elif o.d_closest < cfg.conflict_radius_m and o.t_closest <= cfg.conflict_horizon_s:
                ttc_conflict = min(ttc_conflict, o.t_closest)
        nearest = min([o.dist for o in others], default=cfg.sensing_radius_m)
        n_density = sum(o.dist <= cfg.density_radius_m for o in others)
        n_close = sum(o.dist <= cfg.close_radius_m for o in others)
        n_crossing = sum(
            o.dist <= cfg.crossing_radius_m and o.heading_diff > math.radians(cfg.crossing_heading_deg) for o in others
        )
        approaching = any(o.dist <= cfg.crossing_radius_m and (o.s * o.v_s + o.d * o.v_d) < 0 for o in others)

        le, lfg, lrg, lfc, lrc = self._adjacent(vehicle, vehicle.lane_index, -1)
        re, rfg, rrg, rfc, rrc = self._adjacent(vehicle, vehicle.lane_index, +1)
        rear_gap, rear_closing = self._rear_same_lane(vehicle)
        dz, in_zone = env.model.conflict_zone(env.raw, vehicle)
        prio = env.model.rule_priority(env.raw, vehicle)
        speeds = getattr(vehicle, "target_speeds", np.array([vehicle.target_speed]))
        speed_index = int(getattr(vehicle, "speed_index", 0))
        near_miss = (
            min(ttc_lead, ttc_conflict) < cfg.near_miss_ttc_s or (lead and lead_gap < cfg.near_miss_gap_m)
        )
        return SafetyState(
            ego_speed=float(vehicle.speed),
            ego_target_speed=float(getattr(vehicle, "target_speed", vehicle.speed)),
            speed_limit=float(getattr(vehicle.lane, "speed_limit", math.inf) or math.inf),
            lane_offset=float(lat),
            lane_width=lane_width,
            lead_gap=float(lead_gap),
            lead_closing_speed=float(lead_closing),
            lead_speed_long=float(lead_speed),
            ttc_lead=float(ttc_lead),
            ttc_conflict=float(ttc_conflict),
            nearest_distance=float(nearest),
            n_vehicles_density=int(n_density),
            n_vehicles_close=int(n_close),
            n_crossing=int(n_crossing),
            approaching_interactor=bool(approaching),
            left_lane_exists=le,
            right_lane_exists=re,
            left_front_gap=lfg,
            left_rear_gap=lrg,
            left_front_closing=lfc,
            left_rear_closing=lrc,
            right_front_gap=rfg,
            right_rear_gap=rrg,
            right_front_closing=rfc,
            right_rear_closing=rrc,
            rear_gap=rear_gap,
            rear_closing_speed=rear_closing,
            dist_to_conflict_zone=float(dz),
            in_conflict_zone=bool(in_zone),
            intends_merge_or_crossing=bool(env.model.intends_merge_or_crossing(env.raw, vehicle)),
            neighbor_has_priority=self._neighbor_has_priority(i, prio),
            rule_priority=float(prio),
            route_remaining=float(env.model.route_remaining(env.raw, vehicle, env.traveled(i))),
            time_remaining=float(max(0.0, env.scenario.duration_s - env.raw.time)),
            speed_index=speed_index,
            n_target_speeds=int(len(speeds)),
            near_miss=bool(near_miss),
            on_road=bool(vehicle.on_road),
            next_target_speed=float(speeds[min(speed_index + 1, len(speeds) - 1)]) if len(speeds) else None,
        )

    # ------------------------------------------------------------------
    def predict_state(self, i: int, obs_flat: np.ndarray, action: str, dt: float, tau: float) -> SafetyState:
        """One-step prediction x'_a of executing canonical ``action`` for ``dt`` seconds (report #58).

        Ego: new target speed (speed index +1 / -1 / unchanged), first-order speed response with time
        constant ``tau`` (highway-env ControlledVehicle speed controller), displacement along the current
        heading with the mean speed over ``dt``.  Neighbours: constant velocity.  The predicted relative
        observation is passed through the SAME ``extract`` as the current state; lane geometry (offset,
        adjacent lanes) is kept at its current value (lateral motion of lane changes is not predicted)."""
        veh = self.env.vehicles[i]
        speeds = np.asarray(getattr(veh, "target_speeds", [getattr(veh, "target_speed", veh.speed)]), dtype=np.float64)
        idx = int(getattr(veh, "speed_index", 0))
        idx2 = min(idx + 1, len(speeds) - 1) if action == "ACCELERATE" else max(idx - 1, 0) if action == "DECELERATE" else idx
        v0, vt = float(veh.speed), float(speeds[idx2])
        k = math.exp(-dt / tau)
        v1 = vt + (v0 - vt) * k
        v_mean = vt + (v0 - vt) * tau / dt * (1.0 - k)
        obs = np.asarray(obs_flat, dtype=np.float64).reshape(-1, N_F).copy()
        ego = obs[0]
        n = math.hypot(ego[5], ego[6]) or 1.0
        e = np.array([ego[5] / n, ego[6] / n])
        v_ego0 = np.array([ego[3], ego[4]])
        v_ego1, v_egom = v1 * e, v_mean * e
        ego[1:3] += v_egom * dt
        ego[3:5] = v_ego1
        for row in obs[1:]:
            if row[0] <= 0.5:
                continue
            v_other = row[3:5] + v_ego0           # observation stores velocities relative to the ego
            row[1:3] = row[1:3] + (v_other - v_egom) * dt
            row[3:5] = v_other - v_ego1
        x = self.extract(i, obs.reshape(-1).astype(np.float32))
        return replace(x, ego_speed=float(v1), ego_target_speed=vt, speed_index=idx2)

    def nf_features(self, x: SafetyState) -> np.ndarray:
        cap_d, cap_t = self.cfg.sensing_radius_m, self.cfg.ttc_cap_s
        lane_ok = lambda exists, fg, rg: float(exists and fg >= 8.0 and rg >= 8.0)
        return np.array(
            [
                min(x.lead_gap, cap_d),
                float(np.clip(x.lead_closing_speed, -20.0, 20.0)),
                min(x.ttc_lead, cap_t),
                min(x.ttc_conflict, cap_t),
                min(x.nearest_distance, cap_d),
                x.lane_offset,
                x.ego_speed,
                float(x.n_vehicles_density),
                lane_ok(x.left_lane_exists, x.left_front_gap, x.left_rear_gap),
                lane_ok(x.right_lane_exists, x.right_front_gap, x.right_rear_gap),
                x.left_space(cap_d),
                x.right_space(cap_d),
            ],
            dtype=np.float32,
        )

    def symbolic_context(self, x: SafetyState) -> SymbolicContext:
        cap = self.cfg.sensing_radius_m
        left_ok = x.left_lane_exists and x.left_front_gap >= 8.0 and x.left_rear_gap >= 8.0
        right_ok = x.right_lane_exists and x.right_front_gap >= 8.0 and x.right_rear_gap >= 8.0
        target_rear = max(x.left_rear_gap if left_ok else 0.0, x.right_rear_gap if right_ok else 0.0)
        return SymbolicContext(
            dist_to_conflict_zone=float(min(x.dist_to_conflict_zone, 1e6)),
            in_conflict_zone=x.in_conflict_zone,
            ttc_min=float(min(x.ttc_lead, x.ttc_conflict, self.cfg.ttc_cap_s)),
            n_crossing=x.n_crossing,
            intends_merge_or_crossing=x.intends_merge_or_crossing,
            approaching_interactor=x.approaching_interactor,
            front_gap=float(min(x.lead_gap, cap)),
            neighbor_has_priority=x.neighbor_has_priority,
            interaction_active=x.n_vehicles_density > 0,
            lead_slower=bool(x.lead_gap < 40.0 and x.lead_closing_speed > 1.0),
            adjacent_lane_feasible=bool(left_ok or right_ok),
            target_rear_gap=float(min(target_rear, cap)),
            n_vehicles_density=x.n_vehicles_density,
            n_vehicles_close=x.n_vehicles_close,
            lateral_gap_limited=not (left_ok or right_ok),
            left_space=x.left_space(cap),
            right_space=x.right_space(cap),
        )


# ===========================
# ADAR-SOFFEN 核心组件: 状态级风险源 s = f_NF(x) 与独立拟合阶段
# ===========================
# Neuro-fuzzy STATE-risk source  s = f_NF(x)  (paper Eq. 26).
#
# The module is intentionally state-level (not action-conditioned).  Action dependence of
# the SafetyShield comes from A_hard(x), the per-action feasibility checks and the
# deterministic fallback rule.
#
# Output semantics: ``s in (0,1)``, larger = safer.  It is a *state-safety score* fitted by
# maximum-likelihood binary cross-entropy to the short-horizon label "no collision / off-road /
# near-miss in the next H decision steps".  It should only be called a probability if its
# reliability on the held-out calibration split C (ECE / Brier / reliability bins, stored in
# the run metadata as ``neuro_fuzzy_holdout_C``) supports that; otherwise report it as a score.
#
# Fixes relative to the original ADARSOFFENSafetyModule
# ------------------------------------------------------
# * parameters were initialised from KMeans on ``np.random.randn`` data and never trained
#   (only ``sigma`` was rescaled heuristically) -> explicit fitting stage on calibration
#   trajectories, then frozen;
# * pruning wrote ``requires_grad`` on advanced-index copies (no effect) and replaced
#   ``nn.Parameter`` objects (breaking optimisers) -> boolean masks + gradient hooks;
# * input features had ad-hoc divisions -> documented z-score standardisation fitted on
#   the calibration data and stored as buffers.

class ADARSOFFENStateRisk(nn.Module):
    """First-order TSK fuzzy system with attribute/rule attention and mask-based pruning."""

    def __init__(self, n_inputs: int, n_rules: int = 7):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules_total = n_rules
        self.mu = nn.Parameter(torch.zeros(n_rules, n_inputs))
        self.log_sigma = nn.Parameter(torch.zeros(n_rules, n_inputs))
        self.attention_weights = nn.Parameter(torch.full((n_rules, n_inputs), 2.0))
        self.rule_attention_weights = nn.Parameter(torch.full((n_rules,), 2.0))
        self.consequents = nn.Parameter(torch.zeros(n_rules, n_inputs))
        self.consequent_bias = nn.Parameter(torch.zeros(n_rules))
        self.register_buffer("attribute_mask", torch.ones(n_rules, n_inputs))
        self.register_buffer("rule_mask", torch.ones(n_rules))
        self.register_buffer("feature_mean", torch.zeros(n_inputs))
        self.register_buffer("feature_std", torch.ones(n_inputs))
        self.register_buffer("fitted", torch.tensor(0, dtype=torch.uint8))
        # gradient hooks keep pruned entries frozen (no requires_grad games on slices)
        self.attention_weights.register_hook(lambda g: g * self.attribute_mask)
        self.consequents.register_hook(lambda g: g * self.attribute_mask)
        self.mu.register_hook(lambda g: g * self.attribute_mask)
        self.log_sigma.register_hook(lambda g: g * self.attribute_mask)
        self.rule_attention_weights.register_hook(lambda g: g * self.rule_mask)
        self.consequent_bias.register_hook(lambda g: g * self.rule_mask)

    # ------------------------------------------------------------------
    @property
    def n_rules(self) -> int:
        return int(self.rule_mask.sum().item())

    def initialise_from_data(self, X: np.ndarray, seed: int) -> None:
        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(0)
        std = X.std(0)
        std[std < 1e-6] = 1.0
        Z = (X - mean) / std
        km = KMeans(n_clusters=self.n_rules_total, random_state=seed, n_init=10).fit(Z)
        sig = np.ones((self.n_rules_total, self.n_inputs))
        for k in range(self.n_rules_total):
            members = Z[km.labels_ == k]
            if len(members) > 1:
                sig[k] = np.maximum(members.std(0), 0.3)
        with torch.no_grad():
            self.feature_mean.copy_(torch.tensor(mean, dtype=torch.float32))
            self.feature_std.copy_(torch.tensor(std, dtype=torch.float32))
            self.mu.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32))
            self.log_sigma.copy_(torch.log(torch.tensor(sig, dtype=torch.float32)))

    def standardise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean) / self.feature_std

    def forward(self, x: torch.Tensor):
        """x: raw physical features (B, n_inputs) -> (score (B,), firing, attention, rule_attention)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.standardise(x)
        attention = torch.sigmoid(self.attention_weights) * self.attribute_mask
        rule_attention = torch.sigmoid(self.rule_attention_weights) * self.rule_mask
        sigma = torch.exp(self.log_sigma).clamp(min=1e-2)
        diff = (z.unsqueeze(1) - self.mu.unsqueeze(0)) / sigma.unsqueeze(0)
        log_firing = (-0.5 * diff.pow(2) * attention.unsqueeze(0)).sum(-1)  # (B, R)
        # log-sum-exp normalisation (numerically stable); masked rules get -inf
        log_w = log_firing + torch.log(rule_attention.clamp(min=1e-12)).unsqueeze(0)
        log_w = log_w.masked_fill(self.rule_mask.unsqueeze(0) == 0, float("-inf"))
        norm_w = torch.softmax(log_w, dim=1)
        rule_out = (z.unsqueeze(1) * (self.consequents * attention).unsqueeze(0)).sum(-1) + self.consequent_bias
        logit = (norm_w * rule_out).sum(1)
        return torch.sigmoid(logit), torch.exp(log_firing), attention, rule_attention

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        s = self.forward(x)[0].clamp(1e-6, 1 - 1e-6)
        return torch.log(s) - torch.log1p(-s)

    @torch.no_grad()
    def score(self, features: np.ndarray) -> float:
        was_training = self.training
        self.eval()
        s = float(self.forward(torch.as_tensor(features, dtype=torch.float32))[0].item())
        self.train(was_training)
        return s

    # ------------------------------------------------------------------
    @torch.no_grad()
    def prune_attributes(self, threshold: float) -> int:
        att = torch.sigmoid(self.attention_weights)
        newly = (att < threshold) & (self.attribute_mask == 1)
        # keep at least one active attribute per rule
        for r in range(self.n_rules_total):
            if (self.attribute_mask[r] * (~newly[r]).float()).sum() < 1:
                newly[r] = False
        self.attribute_mask[newly] = 0.0
        return int(newly.sum().item())

    @torch.no_grad()
    def prune_rules(self, threshold: float) -> int:
        att = torch.sigmoid(self.rule_attention_weights)
        newly = (att < threshold) & (self.rule_mask == 1)
        if (self.rule_mask.sum() - newly.sum()) < 1:
            return 0
        self.rule_mask[newly] = 0.0
        return int(newly.sum().item())

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()


@dataclass
class NFFitReport:
    n_samples: int
    positive_rate: float
    train_bce: float
    val_bce: float
    val_brier: float
    val_auroc: float
    epochs_run: int
    rules_active: int
    attributes_active: int


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class StateRiskFitter:
    """Independent state-risk fitting stage on calibration trajectories (item E)."""

    def __init__(self, cfg: NeuroFuzzyConfig, seed: int):
        self.cfg = cfg
        self.seed = seed

    def fit(self, model: ADARSOFFENStateRisk, X: np.ndarray, y: np.ndarray) -> NFFitReport:
        cfg = self.cfg
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if len(X) < 20 or len(np.unique(y)) < 2:
            raise ValueError(
                f"state-risk fitting needs >=20 samples with both classes; got n={len(X)}, "
                f"classes={np.unique(y).tolist()}. Collect more calibration episodes."
            )
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        n_val = max(1, int(cfg.val_fraction * len(X)))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        model.initialise_from_data(X[tr_idx], seed=self.seed)
        Xt, yt = torch.tensor(X[tr_idx]), torch.tensor(y[tr_idx])
        Xv, yv = torch.tensor(X[val_idx]), torch.tensor(y[val_idx])
        # maximum-likelihood BCE by default; optional class-balanced weights (cfg.class_balanced_loss)
        pos = float(yt.mean())
        if cfg.class_balanced_loss:
            w_pos, w_neg = 0.5 / max(pos, 1e-3), 0.5 / max(1 - pos, 1e-3)
        else:
            w_pos = w_neg = 1.0
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad, epoch = float("inf"), None, 0, 0
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            perm = torch.randperm(len(Xt), generator=g)
            for s in range(0, len(Xt), cfg.batch_size):
                b = perm[s : s + cfg.batch_size]
                logit = model.logit(Xt[b])
                w = torch.where(yt[b] > 0.5, torch.full_like(yt[b], w_pos), torch.full_like(yt[b], w_neg))
                loss = F.binary_cross_entropy_with_logits(logit, yt[b], weight=w)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if cfg.prune_every_epochs and epoch % cfg.prune_every_epochs == 0:
                model.prune_attributes(0.1)
                model.prune_rules(0.05)
            model.eval()
            with torch.no_grad():
                vl = float(F.binary_cross_entropy_with_logits(model.logit(Xv), yv).item())
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.fitted.fill_(1)
        model.freeze()
        with torch.no_grad():
            pt = model(Xt)[0].numpy()
            pv = model(Xv)[0].numpy()
        eps = 1e-6
        bce = lambda yy, pp: float(-np.mean(yy * np.log(pp + eps) + (1 - yy) * np.log(1 - pp + eps)))
        return NFFitReport(
            n_samples=len(X),
            positive_rate=float(y.mean()),
            train_bce=bce(yt.numpy(), pt),
            val_bce=bce(yv.numpy(), pv),
            val_brier=float(np.mean((pv - yv.numpy()) ** 2)),
            val_auroc=_auroc(yv.numpy(), pv),
            epochs_run=epoch,
            rules_active=model.n_rules,
            attributes_active=int(model.attribute_mask.sum().item()),
        )


# ===========================
# SafetyShield: A_hard(x) 离散可行性筛选 + 确定性 fallback
# ===========================
# SafetyShield: neuro-fuzzy state-risk assessment + action-specific discrete feasibility.
#
# Implements paper Eqs. (27)-(30):
#
#     A_hard(x) = { a : C_lon(x,a) <= 0 and C_lat(x,a) <= 0 }
#     a_safe = a_nom                          if a_nom in A_hard(x) and s >= delta_eff
#            = Pi_safe(a_nom; A_hard(x), s)   otherwise (fixed conservative ranking)
#            = emergency action               if A_hard(x) is empty
#
# All checks are **CBF-inspired / RSS-inspired discrete feasibility screening** of the five
# primitive actions over one decision interval.  They are not a formal CBF or RSS guarantee.
#
# This class replaces BOTH ``EnhancedADARAnfisSafetyLayer`` (whose
# ``apply_safety_constraints`` was never called) and the simplified
# ``HierarchicalFusionController._apply_safety_constraints`` (which was the only code on the
# execution path and only mapped FASTER->SLOWER when the untrained score < 0.3).

@dataclass
class ShieldDecision:
    nominal_action: str
    safe_action: str
    admissible: Tuple[str, ...]
    state_score: float
    effective_threshold: float
    triggered: bool          # keep-condition failed (nominal inadmissible or s < delta_eff)
    intervened: bool         # safe_action != nominal_action
    reason: str              # "kept" | "inadmissible:<checks>" | "risk_below_threshold" | "emergency_empty_set" | "topology_noop:MAINTAIN"
    failed_checks: Dict[str, List[str]] = field(default_factory=dict)


class SafetyShield:
    def __init__(self, nf_model: ADARSOFFENStateRisk, feature_fn, cfg: ShieldConfig, dt: float):
        self.nf = nf_model
        self.feature_fn = feature_fn  # SafetyState -> np.ndarray (NF input)
        self.cfg = cfg
        self.dt = float(dt)
        self.rss_min_gap = cfg.rss_min_gap_m
        self._last_update_episode: Optional[int] = None
        self.update_calls = 0

    # ------------------------------------------------------------------
    # neuro-fuzzy state score
    # ------------------------------------------------------------------
    def state_score(self, x: SafetyState) -> float:
        if int(self.nf.fitted.item()) != 1:
            raise RuntimeError("neuro-fuzzy state-risk model has not been fitted; run the calibration stage")
        return self.nf.score(self.feature_fn(x))

    # ------------------------------------------------------------------
    # RSS-inspired distance and CBF-inspired barrier
    # ------------------------------------------------------------------
    def rss_distance(self, v_rear: float, v_front: float) -> float:
        c, rho = self.cfg, self.cfg.rss_reaction_time_s
        v_rear, v_front = max(0.0, v_rear), max(0.0, v_front)
        d = (
            v_rear * rho
            + 0.5 * c.rss_a_max_accel * rho**2
            + (v_rear + rho * c.rss_a_max_accel) ** 2 / (2 * c.rss_b_min_brake)
            - v_front**2 / (2 * c.rss_b_max_brake)
        )
        return max(self.rss_min_gap, d)

    def _accel_for(self, action: str, x: SafetyState) -> float:
        if action == "ACCELERATE":
            return self.cfg.comfort_accel
        if action == "DECELERATE":
            return -self.cfg.comfort_decel
        return 0.0

    def _cbf_ok(self, gap: float, v_ego: float, v_front: float, accel: float) -> bool:
        """Discrete-time barrier h = gap - d_RSS(v_ego, v_front); require h' >= (1-alpha) h."""
        if not math.isfinite(gap) or not math.isfinite(v_front):
            return True
        h = gap - self.rss_distance(v_ego, v_front)
        dt = self.dt
        v_next = max(0.0, v_ego + accel * dt)
        gap_next = gap + (v_front - v_ego) * dt - 0.5 * accel * dt**2
        h_next = gap_next - self.rss_distance(v_next, v_front)
        return h_next >= (1.0 - self.cfg.cbf_alpha) * h

    # ------------------------------------------------------------------
    # A_hard(x)
    # ------------------------------------------------------------------
    def check_action(self, action: str, x: SafetyState) -> List[str]:
        """Return the list of failed checks for ``action`` in state ``x`` (empty = admissible)."""
        c = self.cfg
        failed: List[str] = []
        accel = self._accel_for(action, x)
        has_lead = math.isfinite(x.lead_gap) and math.isfinite(x.lead_speed_long)

        # ---- environment boundary / lane feasibility (C_lat)
        if action == "LANE_LEFT" and not x.left_lane_exists:
            failed.append("lane_topology")
        if action == "LANE_RIGHT" and not x.right_lane_exists:
            failed.append("lane_topology")
        if action in ("LANE_LEFT", "LANE_RIGHT") and x.in_conflict_zone:
            failed.append("no_lane_change_in_conflict_zone")
        if abs(x.lane_offset) > c.max_lateral_offset_m and action in ("LANE_LEFT", "LANE_RIGHT", "ACCELERATE"):
            failed.append("boundary_lateral_offset")
        if action in ("LANE_LEFT", "LANE_RIGHT") and "lane_topology" not in failed:
            fg, rg, fc, rc = (
                (x.left_front_gap, x.left_rear_gap, x.left_front_closing, x.left_rear_closing)
                if action == "LANE_LEFT"
                else (x.right_front_gap, x.right_rear_gap, x.right_front_closing, x.right_rear_closing)
            )
            if fg < c.lane_change_front_gap_m or rg < c.lane_change_rear_gap_m:
                failed.append("target_lane_gap")
            v_front_target = x.ego_speed - fc
            v_rear_target = x.ego_speed + rc
            if fg < self.rss_distance(x.ego_speed, v_front_target):
                failed.append("rss_target_front")
            if rg < self.rss_distance(v_rear_target, x.ego_speed):
                failed.append("rss_target_rear")
            if fc > 0.1 and fg / fc < c.ttc_min_lane_change_s:
                failed.append("ttc_target_front")
            if rc > 0.1 and rg / rc < c.ttc_min_lane_change_s:
                failed.append("ttc_target_rear")

        # ---- longitudinal (C_lon): distance / TTC / RSS / CBF
        if action in ("ACCELERATE", "MAINTAIN", "LANE_LEFT", "LANE_RIGHT"):
            ttc_min = c.ttc_min_accelerate_s if action == "ACCELERATE" else c.ttc_min_maintain_s
            if min(x.ttc_lead, x.ttc_conflict) < ttc_min:
                failed.append("ttc")
            if has_lead and action in ("ACCELERATE", "MAINTAIN"):
                v_after = x.ego_speed + accel * self.dt
                if x.lead_gap < self.rss_distance(v_after, x.lead_speed_long):
                    failed.append("rss_lead")
                if not self._cbf_ok(x.lead_gap, x.ego_speed, x.lead_speed_long, accel):
                    failed.append("cbf_lead")
        if action == "DECELERATE":
            if x.rear_closing_speed > 0.1 and x.rear_gap / x.rear_closing_speed < c.rear_ttc_min_s:
                failed.append("rear_ttc")

        # ---- speed constraint
        if action == "ACCELERATE":
            # target speed after the action must respect the lane speed limit
            if x.ego_speed > x.speed_limit + c.speed_limit_tolerance or (
                x.next_target_speed is not None and x.next_target_speed > x.speed_limit + c.speed_limit_tolerance
            ):
                failed.append("speed_limit")
        return failed

    def admissible_with_reasons(self, x: SafetyState) -> Tuple[Tuple[str, ...], Dict[str, List[str]]]:
        reasons = {a: self.check_action(a, x) for a in CANONICAL_ACTIONS}
        adm = tuple(a for a in CANONICAL_ACTIONS if not reasons[a])
        return adm, reasons

    def get_admissible_actions(self, x: SafetyState) -> Tuple[str, ...]:
        """A_hard(x): discrete feasibility screening over ALL valid primitive actions."""
        return self.admissible_with_reasons(x)[0]

    def fallback(self, admissible: Tuple[str, ...]) -> Optional[str]:
        ranked = order_by_ranking(admissible, self.cfg.fallback_ranking)
        return ranked[0] if ranked else None

    def ranked_admissible(self, admissible: Tuple[str, ...]) -> List[str]:
        return order_by_ranking(admissible, self.cfg.fallback_ranking)

    # ------------------------------------------------------------------
    def apply_safety_constraints(
        self, nominal_action: str, x: SafetyState, state_score: float, effective_threshold: float
    ) -> ShieldDecision:
        admissible, reasons = self.admissible_with_reasons(x)
        is_adm = nominal_action in admissible
        risk_ok = state_score >= effective_threshold
        if is_adm and risk_ok:
            return ShieldDecision(nominal_action, nominal_action, admissible, state_score, effective_threshold,
                                  False, False, "kept", reasons)
        if not admissible:
            a = self.cfg.emergency_action
            return ShieldDecision(nominal_action, a, admissible, state_score, effective_threshold,
                                  True, a != nominal_action, "emergency_empty_set", reasons)
        if (
            self.cfg.fallback_mode == "topology_noop_to_maintain"
            and nominal_action in ("LANE_LEFT", "LANE_RIGHT")
            and "lane_topology" in reasons[nominal_action]  # no adjacent lane -> env executes keep-lane/keep-speed
            and "MAINTAIN" in admissible
            and risk_ok
        ):
            # the env would execute this lateral action as keep-lane/keep-speed: use its physical equivalent
            return ShieldDecision(nominal_action, "MAINTAIN", admissible, state_score, effective_threshold,
                                  True, True, "topology_noop:MAINTAIN", reasons)
        a = self.fallback(admissible)
        reason = "inadmissible:" + ",".join(reasons[nominal_action]) if not is_adm else "risk_below_threshold"
        return ShieldDecision(nominal_action, a, admissible, state_score, effective_threshold,
                              True, a != nominal_action, reason, reasons)

    def apply_safety_constraints_action_conditioned(
        self, nominal_action: str, x: SafetyState, scores: Dict[str, float], effective_threshold: float
    ) -> ShieldDecision:
        """Pi_safe with the action-conditioned risk gate (ShieldConfig.risk_mode = "action_conditioned").

        ``scores[a]`` = f_NF(x'_a) for the one-step prediction of executing a.  A_hard, the fallback ranking
        and the topology no-op rule are unchanged; only the risk test becomes action-specific."""
        admissible, reasons = self.admissible_with_reasons(x)
        ok = lambda a: a in admissible and scores.get(a, -math.inf) >= effective_threshold
        s_nom = float(scores.get(nominal_action, float("nan")))
        if ok(nominal_action):
            return ShieldDecision(nominal_action, nominal_action, admissible, s_nom, effective_threshold,
                                  False, False, "kept", reasons)
        if not admissible:
            a = self.cfg.emergency_action
            return ShieldDecision(nominal_action, a, admissible, s_nom, effective_threshold,
                                  True, a != nominal_action, "emergency_empty_set", reasons)
        if (
            self.cfg.fallback_mode == "topology_noop_to_maintain"
            and nominal_action in ("LANE_LEFT", "LANE_RIGHT")
            and "lane_topology" in reasons[nominal_action]
            and ok("MAINTAIN")
        ):
            return ShieldDecision(nominal_action, "MAINTAIN", admissible, s_nom, effective_threshold,
                                  True, True, "topology_noop:MAINTAIN", reasons)
        safe = [a for a in self.ranked_admissible(admissible) if ok(a)]
        why = "inadmissible:" + ",".join(reasons[nominal_action]) if nominal_action not in admissible else "action_risk_below_threshold"
        if safe:
            return ShieldDecision(nominal_action, safe[0], admissible, s_nom, effective_threshold,
                                  True, safe[0] != nominal_action, why, reasons)
        a = self.fallback(admissible)
        return ShieldDecision(nominal_action, a, admissible, s_nom, effective_threshold,
                              True, a != nominal_action, why + "|no_action_meets_risk", reasons)

    def ranked_admissible_action_conditioned(self, admissible: Tuple[str, ...], scores: Dict[str, float], thr: float) -> List[str]:
        """Fallback list for coordination: admissible actions meeting the risk threshold first (ranking order)."""
        ranked = self.ranked_admissible(admissible)
        return [a for a in ranked if scores.get(a, -math.inf) >= thr] + [a for a in ranked if scores.get(a, -math.inf) < thr]

    # ------------------------------------------------------------------
    # optional online adaptation (flag adapt_safety_layer_online; never during evaluation)
    # ------------------------------------------------------------------
    def end_of_episode_update(self, episode_count: int, collision: bool, near_miss: bool, success: bool) -> Dict:
        """Exactly ONE call per training episode; ``episode_count`` must be passed explicitly."""
        if self._last_update_episode == episode_count:
            raise RuntimeError(f"safety layer updated twice in episode {episode_count}")
        self._last_update_episode = episode_count
        self.update_calls += 1
        c = self.cfg
        with torch.no_grad():
            if collision:
                self.nf.log_sigma.add_(np.log(0.95))
                self.rss_min_gap = min(c.rss_min_gap_max_m, self.rss_min_gap * 1.1)
            elif near_miss:
                self.nf.log_sigma.add_(np.log(0.98))
            elif success:
                self.nf.log_sigma.add_(np.log(1.02))
        pruned = {}
        if episode_count > 0 and episode_count % c.prune_every_episodes == 0:
            pruned = {
                "attributes": self.nf.prune_attributes(c.attribute_prune_threshold),
                "rules": self.nf.prune_rules(c.rule_prune_threshold),
            }
        return {"episode": episode_count, "rss_min_gap": self.rss_min_gap, "pruned": pruned}

    def statistics(self) -> Dict:
        return {
            "rules_active": self.nf.n_rules,
            "attributes_active": int(self.nf.attribute_mask.sum().item()),
            "rss_min_gap": self.rss_min_gap,
            "update_calls": self.update_calls,
        }


# ===========================
# 离线语义先验源: 冻结模板库、匹配、置信度校准
# ===========================
# Offline semantic-prior source, confidence calibration and the frozen template library.
#
# Online path (paper Eqs. 20-25)
#     T, r      = Match(c, L_sem)                 (template, retrieval-match quality)
#     psi(a|c)  = RankScore(T, a)
#     a_sp      = argmax_a psi(a|c)
#     raw_kappa = l_T * template_confidence + l_M * match_reliability + l_V * validation_reliability
#                 (match_reliability = empirical reliability of T at the observed match level, report #53;
#                  raw_confidence_mode="paper_eq22" uses the fraction of satisfied conditions instead)
#     kappa     = Calib(raw_kappa)                 (Platt scaling, fitted offline, frozen)
#     admitted  = kappa >= tau_c                   (the ONLY admission gate)
#
# Calibration label (semantic_acceptability), recorded in metadata:
#     For a calibration-rollout step where the matched semantic recommendation a_sp was
#     EXECUTED by agent i:  y = 1  iff  a_sp in A_hard(x)  and agent i has no collision,
#     no off-road and no near-miss (FeatureConfig definition) in the next H decision steps
#     (H = SemanticConfig.label_horizon_steps); otherwise y = 0.

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq}


class FrozenLibraryError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    value: object

    def validate(self) -> None:
        types = SymbolicContext.field_types()
        if self.feature not in types:
            raise ValueError(f"template condition references unknown context variable {self.feature!r}")
        if self.op not in _OPS:
            raise ValueError(f"invalid operator {self.op!r}")
        if types[self.feature] == "bool":
            if self.op != "==" or not isinstance(self.value, bool):
                raise ValueError(f"boolean variable {self.feature} must use '==' with a bool value")
        else:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError(f"numeric variable {self.feature} requires a numeric threshold, got {self.value!r}")

    def holds(self, ctx: SymbolicContext) -> bool:
        return bool(_OPS[self.op](ctx.get(self.feature), self.value))


@dataclass(frozen=True)
class DecisionTemplate:
    key: str
    description: str
    conditions: Tuple[Condition, ...]
    #: ranked groups of canonical actions, e.g. (("DECELERATE",), ("MAINTAIN",), ("LANE_LEFT","LANE_RIGHT"), ...)
    ranking: Tuple[Tuple[str, ...], ...]
    confidence: float  # nominal template confidence field in [0,1]
    min_safety_score: float = 0.0  # delta_temp (0 = none)

    def validate(self) -> None:
        for c in self.conditions:
            c.validate()
        flat = [a for g in self.ranking for a in g]
        for a in flat:
            if a not in CANONICAL_ACTIONS:
                raise ValueError(f"template {self.key}: unknown action {a!r}")
        if sorted(flat) != sorted(CANONICAL_ACTIONS):
            raise ValueError(f"template {self.key}: ranking must cover each canonical action exactly once")
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.min_safety_score < 1.0:
            raise ValueError(f"template {self.key}: confidence / threshold out of range")

    def match_score(self, ctx: SymbolicContext) -> float:
        return sum(c.holds(ctx) for c in self.conditions) / len(self.conditions)

    def rank_score(self, action: str) -> float:
        """psi(a|c) = (G - g(a)) / G with g the 0-based group index (ties share a score)."""
        G = len(self.ranking)
        for g, group in enumerate(self.ranking):
            if action in group:
                return (G - g) / G
        raise KeyError(action)

    def top_action(self, ctx: SymbolicContext) -> str:
        """argmax psi; ties inside the top group broken by larger lateral space, then fixed order."""
        group = self.ranking[0]
        if len(group) == 1:
            return group[0]
        if set(group) == {"LANE_LEFT", "LANE_RIGHT"}:
            return "LANE_LEFT" if ctx.left_space > ctx.right_space else "LANE_RIGHT" if ctx.right_space > ctx.left_space else group[0]
        return group[0]

    def to_json(self) -> Dict:
        d = asdict(self)
        d["conditions"] = [asdict(c) for c in self.conditions]
        d["ranking"] = [list(g) for g in self.ranking]
        return d

    @classmethod
    def from_json(cls, d: Dict) -> "DecisionTemplate":
        return cls(
            key=d["key"],
            description=d["description"],
            conditions=tuple(Condition(**c) for c in d["conditions"]),
            ranking=tuple(tuple(g) for g in d["ranking"]),
            confidence=float(d["confidence"]),
            min_safety_score=float(d.get("min_safety_score", 0.0)),
        )


C = Condition
LANES = ("LANE_LEFT", "LANE_RIGHT")

#: The five templates of paper Table 1, with executable symbolic conditions.
#: Numeric thresholds are the implementation of the qualitative conditions in Table 1 and
#: must be listed in the revised paper (they were absent from the first version).
PAPER_TABLE1_TEMPLATES: Tuple[DecisionTemplate, ...] = (
    DecisionTemplate(
        "conflict_zone_approach",
        "Near shared interaction zone; low predicted time margin; cross-agent interaction present",
        (C("dist_to_conflict_zone", "<", 25.0), C("ttc_min", "<", 4.0), C("n_crossing", ">=", 1)),
        (("DECELERATE",), ("MAINTAIN",), LANES, ("ACCELERATE",)),
        0.8,
        0.70,
    ),
    DecisionTemplate(
        "gap_acceptance",
        "Intended crossing or merge; interacting agent approaching; available gap below nominal preference",
        (C("intends_merge_or_crossing", "==", True), C("approaching_interactor", "==", True), C("front_gap", "<", 20.0)),
        (("DECELERATE",), ("MAINTAIN",), LANES, ("ACCELERATE",)),
        0.8,
        0.75,
    ),
    DecisionTemplate(
        "yield_before_merge",
        "Entry or merge context; neighbouring agent has local priority; interaction range is active",
        (C("intends_merge_or_crossing", "==", True), C("neighbor_has_priority", "==", True), C("interaction_active", "==", True)),
        (("DECELERATE",), ("MAINTAIN",), ("ACCELERATE",), LANES),
        0.8,
        0.65,
    ),
    DecisionTemplate(
        "opportunity_lane_change",
        "Slow leading agent; adjacent lane feasible; rear gap sufficient; local risk favourable",
        (C("lead_slower", "==", True), C("adjacent_lane_feasible", "==", True), C("target_rear_gap", ">=", 10.0), C("ttc_min", ">=", 4.0)),
        (LANES, ("MAINTAIN",), ("ACCELERATE",), ("DECELERATE",)),
        0.8,
        0.55,
    ),
    DecisionTemplate(
        "dense_interaction_keep_lane",
        "High local density; frequent neighbouring interactions; limited lateral gap availability",
        (C("n_vehicles_density", ">=", 3), C("n_vehicles_close", ">=", 2), C("lateral_gap_limited", "==", True)),
        (("MAINTAIN",), ("DECELERATE",), LANES, ("ACCELERATE",)),
        0.8,
        0.70,
    ),
)

#: The four templates that the ORIGINAL script actually contained (for provenance only),
#: rewritten with canonical names according to the ORIGINAL AUTHOR INTENT stated in the
#: comments (e.g. "[4, 0, 1, 2, 3]  # 优先减速").  NOTE: at runtime the original integers were
#: interpreted by highway-env as 0=LANE_LEFT, 1=IDLE, so the executed behaviour differed.
LEGACY_CODE_TEMPLATES: Tuple[DecisionTemplate, ...] = (
    DecisionTemplate(
        "legacy_intersection_approach", "original: intersection_distance<30 & density 'high' (never matchable)",
        (C("dist_to_conflict_zone", "<", 30.0), C("n_vehicles_density", ">=", 7)),
        (("DECELERATE",), ("MAINTAIN",), ("LANE_LEFT",), ("LANE_RIGHT",), ("ACCELERATE",)), 0.8, 0.8),
    DecisionTemplate(
        "legacy_high_density_merge", "original: density>0.7 & merge_required",
        (C("n_vehicles_density", ">=", 7), C("intends_merge_or_crossing", "==", True)),
        (("DECELERATE",), ("MAINTAIN",), ("LANE_LEFT",), ("LANE_RIGHT",), ("ACCELERATE",)), 0.8, 0.85),
    DecisionTemplate(
        "legacy_emergency_avoidance", "original: ttc<1.5 & collision_risk high",
        (C("ttc_min", "<", 1.5),),
        (("DECELERATE",), ("LANE_LEFT",), ("LANE_RIGHT",), ("MAINTAIN",), ("ACCELERATE",)), 0.8, 0.9),
    DecisionTemplate(
        "legacy_normal_highway", "original: density<0.3 & collision_risk low",
        (C("n_vehicles_density", "<", 3), C("ttc_min", ">=", 3.0)),
        (("MAINTAIN",), ("ACCELERATE",), ("LANE_LEFT",), ("LANE_RIGHT",), ("DECELERATE",)), 0.8, 0.6),
)

def _reversed_ranking_library(templates: Sequence[DecisionTemplate]) -> Tuple[DecisionTemplate, ...]:
    """STRESS TEST ONLY (Reviewer 1: 'mismatched or misleading symbolic knowledge').

    Same symbolic conditions, nominal confidence and delta_temp as Table 1, but every ranking is
    reversed, so the semantic recommendation is the action the template considers WORST.  The
    library is calibrated like any other library (its own calibration split); a working
    confidence gate should then admit it far less often than the correct library."""
    return tuple(
        DecisionTemplate(t.key + "_reversed", "STRESS (reversed ranking): " + t.description, t.conditions,
                         tuple(reversed(t.ranking)), t.confidence, t.min_safety_score)
        for t in templates
    )


LIBRARIES = {
    "paper_table1": PAPER_TABLE1_TEMPLATES,
    "legacy_code_v1": LEGACY_CODE_TEMPLATES,
    "stress_reversed_table1": _reversed_ranking_library(PAPER_TABLE1_TEMPLATES),
}


@dataclass
class TemplateReliability:
    successes: int = 0
    trials: int = 0


class SemanticLibrary:
    """Finite template library.  After ``freeze()`` nothing can be added or modified."""

    def __init__(self, templates: Sequence[DecisionTemplate], name: str):
        self.name = name
        self._templates: List[DecisionTemplate] = []
        self.validation_reliability: Dict[str, float] = {}
        #: empirical reliability per (template key, match level), key "<template>|<level>" (report #53)
        self.level_reliability: Dict[str, float] = {}
        self._frozen = False
        self.version = 0
        for t in templates:
            self.add_template(t)

    @classmethod
    def from_config(cls, cfg: SemanticConfig) -> "SemanticLibrary":
        return cls(LIBRARIES[cfg.library_version], cfg.library_version)

    # ------------------------------------------------------------------
    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def templates(self) -> Tuple[DecisionTemplate, ...]:
        return tuple(self._templates)

    def _check_mutable(self, what: str) -> None:
        if self._frozen:
            raise FrozenLibraryError(f"semantic library is frozen: cannot {what}")

    def add_template(self, t: DecisionTemplate) -> None:
        self._check_mutable("add a template")
        t.validate()
        if any(x.key == t.key for x in self._templates):
            raise ValueError(f"duplicate template key {t.key}")
        self._templates.append(t)
        self.version += 1

    def replace_template(self, key: str, new: DecisionTemplate) -> None:
        self._check_mutable("modify ranking/confidence")
        new.validate()
        self._templates = [new if t.key == key else t for t in self._templates]
        self.version += 1

    def set_validation_reliability(self, rel: Dict[str, float]) -> None:
        self._check_mutable("modify validation reliability")
        for k, v in rel.items():
            if not 0.0 <= v <= 1.0:
                raise ValueError("reliability must be in [0,1]")
        self.validation_reliability = dict(rel)
        self.version += 1

    def set_level_reliability(self, rel: Dict[str, float]) -> None:
        self._check_mutable("modify match-level reliability")
        for k, v in rel.items():
            if not 0.0 <= v <= 1.0:
                raise ValueError("reliability must be in [0,1]")
        self.level_reliability = dict(rel)
        self.version += 1

    def remove_template(self, key: str) -> None:
        self._check_mutable("remove a template")
        if not any(t.key == key for t in self._templates):
            raise KeyError(key)
        self._templates = [t for t in self._templates if t.key != key]
        self.version += 1

    def freeze(self) -> None:
        self._frozen = True

    # ------------------------------------------------------------------
    def find_matching_template(self, ctx: SymbolicContext, min_score: float) -> Tuple[Optional[DecisionTemplate], float]:
        """Returns (matched_template, match_score); matched iff best score > min_score.

        Ties are broken by library order (deterministic)."""
        best, best_score = None, 0.0
        for t in self._templates:
            s = t.match_score(ctx)
            if s > best_score:
                best, best_score = t, s
        if best is None or best_score <= min_score:
            return None, best_score
        return best, best_score

    def predicate_signature(self, ctx: SymbolicContext) -> Tuple[bool, ...]:
        """Exact cache key: the truth value of every condition in the library."""
        return tuple(c.holds(ctx) for t in self._templates for c in t.conditions)

    def to_json(self) -> Dict:
        return {
            "name": self.name,
            "templates": [t.to_json() for t in self._templates],
            "validation_reliability": self.validation_reliability,
            "level_reliability": self.level_reliability,
        }

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "SemanticLibrary":
        with open(path) as f:
            d = json.load(f)
        lib = cls([DecisionTemplate.from_json(t) for t in d["templates"]], d["name"])
        lib.set_validation_reliability(d.get("validation_reliability", {}))
        lib.set_level_reliability(d.get("level_reliability", {}))
        return lib


def reversed_ranking_copy(library: "SemanticLibrary", keep_reliability: bool) -> "SemanticLibrary":
    """STRESS TEST ONLY (Reviewer 2 comment 7, Reviewer 1: misleading semantic knowledge).

    Same template keys, conditions, nominal confidence and delta_temp, every ranking reversed (the
    recommendation becomes the action the template ranks WORST).
    keep_reliability=True : the copy keeps the reliabilities q estimated for the CORRECT library and is frozen
                            -> the library is corrupted AFTER calibration (the gate cannot know);
    keep_reliability=False: an unfrozen copy without reliabilities -> must go through the calibration stage
                            (q from split A, Platt on split B) like any library before deployment."""
    lib = SemanticLibrary([DecisionTemplate(t.key, "STRESS (reversed ranking): " + t.description, t.conditions,
                                            tuple(reversed(t.ranking)), t.confidence, t.min_safety_score)
                           for t in library.templates], library.name + "+reversed")
    if keep_reliability:
        lib.set_validation_reliability(library.validation_reliability)
        lib.set_level_reliability(library.level_reliability)
        lib.freeze()
    return lib


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------
class SemanticConfidenceCalibrator:
    """Maps raw semantic confidence to calibrated confidence.

    method="platt"   : p = sigmoid(A * raw + B), A >= 0, fitted by Newton-Raphson on Platt's
                       prior-corrected targets (Platt 1999) -> main calibration.  The A >= 0
                       constraint keeps the map non-decreasing; the unconstrained slope is kept
                       in ``params()`` because a negative value means the raw confidence is not
                       informative about acceptability on the calibration data.
    method="isotonic": non-decreasing step function (sklearn IsotonicRegression) ->
                       sensitivity control.
    After ``freeze()`` any further ``fit`` raises.
    """

    def __init__(self, method: str = "platt"):
        if method not in ("platt", "isotonic"):
            raise ValueError(method)
        self.method = method
        self.A: Optional[float] = None
        self.B: Optional[float] = None
        self._iso_x: Optional[np.ndarray] = None
        self._iso_y: Optional[np.ndarray] = None
        self.n_fit = 0
        self.positive_rate = float("nan")
        self.unconstrained_A: Optional[float] = None  # Platt slope before the monotonicity constraint
        self._frozen = False

    @property
    def fitted(self) -> bool:
        return (self.A is not None) if self.method == "platt" else (self._iso_x is not None)

    def freeze(self) -> None:
        if not self.fitted:
            raise RuntimeError("cannot freeze an unfitted calibrator")
        self._frozen = True

    def fit(self, raw_confidence: Sequence[float], binary_labels: Sequence[int]) -> "SemanticConfidenceCalibrator":
        if self._frozen:
            raise FrozenLibraryError("calibrator is frozen; refitting (e.g. during evaluation) is forbidden")
        x = np.asarray(raw_confidence, dtype=np.float64)
        y = np.asarray(binary_labels, dtype=np.float64)
        if x.shape != y.shape or x.ndim != 1 or len(x) == 0:
            raise ValueError("raw_confidence and labels must be equal-length 1-D arrays")
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("labels must be binary")
        self.n_fit = len(x)
        self.positive_rate = float(y.mean())
        if self.method == "platt":
            self.A, self.B, self.unconstrained_A = self._fit_platt_monotone(x, y)
        else:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(x, y)
            self._iso_x = np.asarray(iso.X_thresholds_, dtype=np.float64)
            self._iso_y = np.asarray(iso.y_thresholds_, dtype=np.float64)
        return self

    @staticmethod
    def _fit_platt(x: np.ndarray, y: np.ndarray, iters: int = 100) -> Tuple[float, float]:
        n_pos, n_neg = y.sum(), len(y) - y.sum()
        # Platt's prior-corrected targets avoid infinite parameters under separation.
        t = np.where(y > 0.5, (n_pos + 1) / (n_pos + 2), 1.0 / (n_neg + 2))
        A, B = 0.0, float(np.log((n_pos + 1) / (n_neg + 1)))  # p = sigmoid(A*x + B)
        if np.ptp(x) < 1e-12:  # a single raw value: only the intercept is identifiable
            m = float(np.clip(t.mean(), 1e-6, 1 - 1e-6))
            return 0.0, float(np.log(m / (1 - m)))

        def nll(a, b):
            z = a * x + b
            return float(np.sum(np.logaddexp(0, z) - t * z))

        f = nll(A, B)
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(A * x + B)))
            g = np.array([np.sum((p - t) * x), np.sum(p - t)])
            w = p * (1 - p) + 1e-12
            H = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]]) + 1e-10 * np.eye(2)
            step = np.linalg.solve(H, g)
            lr = 1.0
            while lr > 1e-8:  # backtracking line search
                A2, B2 = A - lr * step[0], B - lr * step[1]
                f2 = nll(A2, B2)
                if f2 <= f + 1e-12:
                    break
                lr *= 0.5
            A, B, f_old, f = A2, B2, f, f2
            if abs(f_old - f) < 1e-12 or np.max(np.abs(lr * step)) < 1e-10:
                break
        return float(A), float(B)

    @classmethod
    def _fit_platt_monotone(cls, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Platt scaling constrained to a NON-DECREASING map (A >= 0).

        A calibration map must not reverse the order of raw confidences (a more confident template
        would otherwise be admitted less often).  If the unconstrained optimum has A < 0 -- i.e. the
        raw confidence is not positively related to acceptability on the calibration data -- the
        constrained optimum is A = 0 with the intercept fitted alone (calibrated confidence = the
        prior-corrected base rate).  The unconstrained slope is returned for reporting."""
        A, B = cls._fit_platt(x, y)
        if A >= 0:
            return A, B, A
        n_pos, n_neg = y.sum(), len(y) - y.sum()
        t = np.where(y > 0.5, (n_pos + 1) / (n_pos + 2), 1.0 / (n_neg + 2))
        m = float(np.clip(t.mean(), 1e-6, 1 - 1e-6))
        return 0.0, float(np.log(m / (1 - m))), A

    def predict(self, raw_confidence) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator used before fit()")
        x = np.asarray(raw_confidence, dtype=np.float64)
        if self.method == "platt":
            return 1.0 / (1.0 + np.exp(-(self.A * x + self.B)))
        return np.interp(x, self._iso_x, self._iso_y, left=self._iso_y[0], right=self._iso_y[-1])

    def params(self) -> Dict:
        d = {"method": self.method, "n_fit": self.n_fit, "positive_rate": self.positive_rate}
        if self.method == "platt":
            d.update({"A": self.A, "B": self.B, "unconstrained_A": self.unconstrained_A,
                      "monotonicity_constraint_active": bool(self.unconstrained_A is not None and self.unconstrained_A < 0)})
        else:
            d.update({"x": self._iso_x.tolist(), "y": self._iso_y.tolist()})
        return d

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.params(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SemanticConfidenceCalibrator":
        with open(path) as f:
            d = json.load(f)
        c = cls(d["method"])
        c.n_fit, c.positive_rate = d["n_fit"], d["positive_rate"]
        if c.method == "platt":
            c.A, c.B = d["A"], d["B"]
            c.unconstrained_A = d.get("unconstrained_A")
        else:
            c._iso_x, c._iso_y = np.array(d["x"]), np.array(d["y"])
        c.freeze()
        return c


def calibration_diagnostics(prob: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> Dict:
    """ECE, Brier, NLL and reliability-diagram bins (for Reviewer 2, comment 1)."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            conf, acc = float(p[m].mean()), float(y[m].mean())
            ece += m.mean() * abs(conf - acc)
            rows.append({"bin": [float(bins[b]), float(bins[b + 1])], "count": int(m.sum()), "mean_conf": conf, "accuracy": acc})
    return {
        "n": int(len(y)),
        "ece": float(ece),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "reliability_bins": rows,
    }


# ---------------------------------------------------------------------------
# Online semantic source
# ---------------------------------------------------------------------------
@dataclass
class SemanticQuery:
    matched: bool
    template_key: Optional[str]
    match_score: float
    semantic_action: Optional[str]
    raw_confidence: float
    calibrated_confidence: float
    delta_temp: float


class SemanticSource:
    """Retrieval + confidence. Caches ONLY retrieval results (never a final decision)."""

    def __init__(self, library: SemanticLibrary, cfg: SemanticConfig, calibrator: Optional[SemanticConfidenceCalibrator]):
        self.library = library
        self.cfg = cfg
        self.calibrator = calibrator
        self._cache: Dict[Tuple, Tuple[Optional[str], float]] = {}
        self._cache_version = library.version
        self.cache_hits = 0
        self.queries = 0

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_version = self.library.version

    def raw_confidence(self, t: DecisionTemplate, match_score: float) -> float:
        c = self.cfg
        prior = c.reliability_prior_alpha / (c.reliability_prior_alpha + c.reliability_prior_beta)
        q = self.library.validation_reliability.get(t.key, prior)
        if c.raw_confidence_mode == "paper_eq22":
            m = match_score  # fraction of satisfied conditions (first submission)
        else:
            m = self.library.level_reliability.get(match_level_key(t.key, match_score), prior)
        raw = c.lambda_template * t.confidence + c.lambda_match * m + c.lambda_validation * q
        return float(np.clip(raw, 0.0, 1.0))

    def _retrieve(self, ctx: SymbolicContext) -> Tuple[Optional[DecisionTemplate], float]:
        if self._cache_version != self.library.version:
            self.clear_cache()
        key = self.library.predicate_signature(ctx)
        self.queries += 1
        if key in self._cache:
            self.cache_hits += 1
            tkey, score = self._cache[key]
            t = next((x for x in self.library.templates if x.key == tkey), None) if tkey else None
            return t, score
        t, score = self.library.find_matching_template(ctx, self.cfg.match_min_score)
        self._cache[key] = (t.key if t else None, score)
        return t, score

    def query(self, ctx: SymbolicContext) -> SemanticQuery:
        t, score = self._retrieve(ctx)
        if t is None:
            return SemanticQuery(False, None, score, None, 0.0, 0.0, 0.0)
        raw = self.raw_confidence(t, score)
        cal = float(self.calibrator.predict([raw])[0]) if (self.calibrator is not None and self.calibrator.fitted) else float("nan")
        return SemanticQuery(True, t.key, score, t.top_action(ctx), raw, cal, t.min_safety_score)


def match_level_key(template_key: str, match_score: float) -> str:
    """Key of a (template, match level) cell; match levels are fractions k/n of satisfied conditions."""
    return f"{template_key}|{round(float(match_score), 4):.4f}"


def estimate_level_reliability(records: Sequence[Tuple[str, float, int]], cfg: SemanticConfig) -> Dict[str, float]:
    """Beta-smoothed reliability per (template, match level) from CALIBRATION split A only."""
    cells: Dict[str, List[int]] = {}
    for key, score, y in records:
        cells.setdefault(match_level_key(key, score), []).append(int(y))
    a, b = cfg.reliability_prior_alpha, cfg.reliability_prior_beta
    return {k: (sum(v) + a) / (len(v) + a + b) for k, v in sorted(cells.items())}


def estimate_validation_reliability(records: Sequence[Tuple[str, int]], cfg: SemanticConfig, keys: Sequence[str]) -> Dict[str, float]:
    """q_bar per template from CALIBRATION split A only (Beta-smoothed success rate)."""
    out = {}
    for k in keys:
        ys = [y for kk, y in records if kk == k]
        out[k] = (sum(ys) + cfg.reliability_prior_alpha) / (len(ys) + cfg.reliability_prior_alpha + cfg.reliability_prior_beta)
    return out


# ===========================
# Priority-Based Local Conflict Coordination (safe 候选)
# ===========================
# Priority-based local conflict coordination over SAFETY-FILTERED candidates (Eqs. 31-35).
#
# Pass 2 of the two-pass execution flow: it receives the complete set {a_i^safe} of all
# agents (never raw PPO intentions), builds the local conflict graph, computes
#
#     P_i = w_r R_i + w_u U_i + w_g G_i - w_m M_i
#
# and resolves conflicts greedily in descending priority (ties: lower agent id first).
# The highest-priority agent of a conflict keeps its safety-filtered action; each
# lower-priority agent keeps its action if it no longer conflicts with the already
# accepted higher-priority actions, otherwise it takes the first action of its admissible
# fallback list that does not conflict.  A strict total order is used, so two agents can
# never both yield because each considers itself lower priority.
#
# Quantities (all in [0,1]):
#     R_i  risk urgency           = 1 - s_i   (neuro-fuzzy state-safety score)
#     U_i  task/route urgency     = clip(route_remaining / (time_remaining * v_ref), 0, 1)
#     G_i  rule-based priority    = scenario right-of-way (intersection lane priority / 3,
#                                   roundabout: 1 on the ring, highway: 0)
#     M_i  maneuver aggressiveness = table in CoordinationConfig (of the SAFE action)

@dataclass
class CoordCandidate:
    agent_id: int
    safe_action: str
    fallback_actions: Sequence[str]  # admissible alternatives in fallback order
    state: SafetyState
    state_score: float
    vehicle: object  # highway-env ControlledVehicle (for route-based prediction)


@dataclass
class CoordResult:
    agent_id: int
    safe_action: str
    executed_action: str
    priority: float
    intervened: bool
    reason: str
    conflict_with: List[int] = field(default_factory=list)


class PriorityCoordinator:
    def __init__(self, cfg: CoordinationConfig):
        self.cfg = cfg
        self._traj_cache: Dict[Tuple[int, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    def priority(self, c: CoordCandidate, action: Optional[str] = None) -> Tuple[float, Dict[str, float]]:
        cfg = self.cfg
        x = c.state
        R = float(np.clip(1.0 - c.state_score, 0.0, 1.0))
        denom = max(x.time_remaining, 1e-3) * cfg.reference_speed_mps
        U = float(np.clip(x.route_remaining / denom, 0.0, 1.0))
        G = float(np.clip(x.rule_priority, 0.0, 1.0))
        M = float(cfg.maneuver_aggressiveness[action or c.safe_action])
        P = cfg.w_r * R + cfg.w_u * U + cfg.w_g * G - cfg.w_m * M
        return P, {"R": R, "U": U, "G": G, "M": M}

    # ------------------------------------------------------------------
    def predict(self, c: CoordCandidate, action: str) -> np.ndarray:
        """Short-horizon positions (T, 2) along the planned route under ``action``."""
        key = (c.agent_id, action)
        if key in self._traj_cache:
            return self._traj_cache[key]
        cfg = self.cfg
        v = c.vehicle
        net = v.road.network
        speeds = np.asarray(getattr(v, "target_speeds", [v.target_speed]), dtype=float)
        idx = int(getattr(v, "speed_index", 0))
        if action == "ACCELERATE":
            idx = min(idx + 1, len(speeds) - 1)
        elif action == "DECELERATE":
            idx = max(idx - 1, 0)
        v_target = float(speeds[idx]) if len(speeds) else float(v.target_speed)
        lane_index = v.lane_index
        lateral_target = 0.0
        if action in ("LANE_LEFT", "LANE_RIGHT"):
            f, t, i = lane_index
            j = i - 1 if action == "LANE_LEFT" else i + 1
            if 0 <= j < len(net.graph[f][t]):
                lateral_target = (j - i) * v.lane.width_at(0)
        s0, d0 = v.lane.local_coordinates(v.position)
        route = _explicit_route(net, v.route or [lane_index], lane_index)
        kp = getattr(v, "KP_A", 1.0 / 0.6)
        n = int(round(cfg.horizon_s / cfg.dt_s))
        pts = []
        s, speed = s0, float(v.speed)
        for k in range(1, n + 1):
            acc = float(np.clip(kp * (v_target - speed), -5.0, 5.0))
            s += speed * cfg.dt_s + 0.5 * acc * cfg.dt_s**2
            speed = max(0.0, speed + acc * cfg.dt_s)
            frac = min(1.0, k * cfg.dt_s / cfg.lane_change_duration_s)
            d = d0 + (lateral_target - d0) * frac if lateral_target != 0.0 else d0 * (1 - frac)
            pts.append(_position_along(net, route, s, d))
        arr = np.asarray(pts)
        self._traj_cache[key] = arr
        return arr

    def conflict(self, ci: CoordCandidate, ai: str, cj: CoordCandidate, aj: str) -> bool:
        cfg = self.cfg
        pi, pj = self.predict(ci, ai), self.predict(cj, aj)
        if float(np.min(np.linalg.norm(pi - pj, axis=1))) < cfg.conflict_distance_m:
            return True
        # both change into the same target lane with small longitudinal separation
        if ai in ("LANE_LEFT", "LANE_RIGHT") and aj in ("LANE_LEFT", "LANE_RIGHT"):
            li, lj = ci.vehicle.lane_index, cj.vehicle.lane_index
            if li[:2] == lj[:2]:
                ti = li[2] + (-1 if ai == "LANE_LEFT" else 1)
                tj = lj[2] + (-1 if aj == "LANE_LEFT" else 1)
                if ti == tj and ti != li[2]:
                    lane = ci.vehicle.lane
                    si = lane.local_coordinates(ci.vehicle.position)[0]
                    sj = lane.local_coordinates(cj.vehicle.position)[0]
                    if abs(si - sj) < cfg.same_target_lane_gap_m:
                        return True
        return False

    # ------------------------------------------------------------------
    def coordinate(self, cands: Sequence[CoordCandidate]) -> Tuple[List[CoordResult], Dict]:
        self._traj_cache = {}
        cfg = self.cfg
        n = len(cands)
        by_id = {c.agent_id: c for c in cands}
        neighbours = {c.agent_id: set() for c in cands}
        for a in range(n):
            for b in range(a + 1, n):
                ca, cb = cands[a], cands[b]
                if np.linalg.norm(ca.vehicle.position - cb.vehicle.position) <= cfg.neighbour_radius_m:
                    neighbours[ca.agent_id].add(cb.agent_id)
                    neighbours[cb.agent_id].add(ca.agent_id)
        # conflict graph among safety-filtered candidates
        edges = set()
        for i in neighbours:
            for j in neighbours[i]:
                if i < j and self.conflict(by_id[i], by_id[i].safe_action, by_id[j], by_id[j].safe_action):
                    edges.add((i, j))
        comps = _components([c.agent_id for c in cands], edges)
        prios = {c.agent_id: self.priority(c) for c in cands}
        order = sorted(cands, key=lambda c: (-prios[c.agent_id][0], c.agent_id))  # strict total order
        final: Dict[int, str] = {}
        results: Dict[int, CoordResult] = {}
        involved = {i for e in edges for i in e}
        for c in order:
            i = c.agent_id
            P = prios[i][0]
            accepted = [j for j in neighbours[i] if j in final]
            # always check against already-accepted (higher-priority) neighbours, so that a
            # re-selected action of a higher-priority agent is also respected by agents that
            # were not part of the original conflict graph
            clash = [j for j in accepted if self.conflict(c, c.safe_action, by_id[j], final[j])]
            if not clash:
                final[i] = c.safe_action
                reason = "kept_highest_priority_in_conflict" if i in involved else "no_conflict"
                results[i] = CoordResult(i, c.safe_action, c.safe_action, P, False, reason)
                continue
            chosen, reason = None, "fallback"
            for alt in c.fallback_actions:
                if alt == c.safe_action:
                    continue
                if not any(self.conflict(c, alt, by_id[j], final[j]) for j in accepted):
                    chosen = alt
                    break
            if chosen is None:
                chosen, reason = "DECELERATE", "fallback_exhausted_emergency_decelerate"
            final[i] = chosen
            results[i] = CoordResult(i, c.safe_action, chosen, P, chosen != c.safe_action, reason, clash)
        info = {
            "conflict_edges": sorted(edges),
            "conflict_sets": [sorted(s) for s in comps if len(s) > 1],
            "priorities": {i: {"P": p[0], **p[1]} for i, p in prios.items()},
        }
        return [results[c.agent_id] for c in cands], info


def _explicit_route(net, route, current):
    """Route starting at the current lane with explicit, valid lane ids (clipped per segment)."""
    start = next((k for k, r in enumerate(route) if r[0] == current[0] and r[1] == current[1]), None)
    rest = route[start + 1 :] if start is not None else []
    out = [current]
    lane_id = current[2]
    for f, t, i in rest:
        n = len(net.graph[f][t])
        lane_id = min(i if i is not None else lane_id, n - 1)
        out.append((f, t, lane_id))
    return out


def _position_along(net, route, s, d):
    for k, li in enumerate(route):
        lane = net.get_lane(li)
        if s <= lane.length or k == len(route) - 1:
            return lane.position(s, d)
        s -= lane.length
    return net.get_lane(route[-1]).position(s, d)


def _components(nodes, edges):
    parent = {v: v for v in nodes}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: Dict[int, set] = {}
    for v in nodes:
        groups.setdefault(find(v), set()).add(v)
    return list(groups.values())


# ===========================
# Learned Decision Source: PathNet / IPPO + PPO (proposal log-prob)
# ===========================
# Learned decision source: PathNet feature extractor + PPO (proposal action/log-prob).
#
# Revision item C
# ---------------
# Each step stores   proposal_action, proposal_log_prob   (the sampled PPO action and its
# behaviour log-probability) and SEPARATELY the executed_action.  The clipped surrogate is
#
#     ratio = exp( log pi_new(proposal_action) - proposal_log_prob )
#
# i.e. numerator and denominator always refer to the SAME action.  Semantic admission,
# SafetyShield and coordination are treated as part of the environment transition from the
# point of view of the learned source.  ``executed_action`` is kept for logging /
# intervention analysis and an optional, separate auxiliary imitation loss.
#
# Curriculum replay of the original code recomputed the "old" log-probability with the
# CURRENT policy for executed actions (no behaviour log-prob) and mixed it into the PPO
# surrogate.  That is not a valid importance ratio; it is removed from the surrogate.  An
# optional auxiliary objective (``use_curriculum_aux``) is provided instead.

class ObservationNormalizer:
    """Running mean/std (Welford).  Frozen during evaluation."""

    def __init__(self, dim: int, clip: float = 5.0):
        self.dim, self.clip = dim, clip
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
        self.frozen = False

    def update(self, x: np.ndarray) -> None:
        if self.frozen:
            return
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.dim)
        for row in x:
            self.count += 1
            d = row - self.mean
            self.mean += d / self.count
            self.M2 += d * (row - self.mean)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return np.asarray(x, dtype=np.float32)
        std = np.sqrt(self.M2 / (self.count - 1)) + 1e-8
        return np.clip((x - self.mean) / std, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> Dict:
        return {"count": self.count, "mean": self.mean.tolist(), "M2": self.M2.tolist()}

    def load_state_dict(self, d: Dict) -> None:
        self.count, self.mean, self.M2 = int(d["count"]), np.array(d["mean"]), np.array(d["M2"])


# ---------------------------------------------------------------------------
# PathNet
# ---------------------------------------------------------------------------
class PathNetModule(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, module_type: str, dropout: float):
        super().__init__()
        self.module_type = module_type
        if module_type == "linear":
            self.layer = nn.Linear(input_dim, output_dim)
        elif module_type == "residual":
            self.layer = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
            self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        elif module_type == "attention":
            self.projection = nn.Linear(input_dim, output_dim)
            self.layer = nn.MultiheadAttention(output_dim, num_heads=4, batch_first=True)
        elif module_type == "lstm":
            self.layer = nn.LSTM(input_dim, output_dim, batch_first=True, bidirectional=True)
            self.projection = nn.Linear(output_dim * 2, output_dim)
        elif module_type == "gru":
            self.layer = nn.GRU(input_dim, output_dim, batch_first=True)
        else:
            raise ValueError(module_type)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.module_type == "residual":
            out = self.layer(x) + self.shortcut(x)
        elif self.module_type == "attention":
            h = self.projection(x).unsqueeze(1)
            out = self.layer(h, h, h)[0].squeeze(1)
        elif self.module_type == "lstm":
            out = self.projection(self.layer(x.unsqueeze(1))[0]).squeeze(1)
        elif self.module_type == "gru":
            out = self.layer(x.unsqueeze(1))[0].squeeze(1)
        else:
            out = self.layer(x)
        return self.dropout(F.relu(self.norm(out)))


class PathNet(nn.Module):
    """Modular PathNet with ONE fixed pathway (index 0).

    Audit: in the original code ``pathway_idx`` was always 0 and ``evolve_pathways`` was
    never called, so no pathway evolution took place.  The revision keeps that (actual)
    behaviour, draws the pathway from a seeded generator and stores it as a buffer so that
    checkpoints are self-contained (the original pathway list was not in the state_dict).
    """

    TYPES = ("linear", "residual", "lstm", "gru", "attention")

    def __init__(self, input_dim, output_dim, num_layers, modules_per_layer, active_modules, hidden, dropout, seed):
        super().__init__()
        self.layers = nn.ModuleList()
        for li in range(num_layers):
            mods = nn.ModuleList()
            for mi in range(modules_per_layer):
                t = self.TYPES[mi % len(self.TYPES)]
                if li == 0:
                    mods.append(PathNetModule(input_dim, hidden, t, dropout))
                elif li == num_layers - 1:
                    mods.append(PathNetModule(hidden, output_dim, "linear", dropout))
                else:
                    mods.append(PathNetModule(hidden, hidden, t, dropout))
            self.layers.append(mods)
        rng = np.random.default_rng(seed)
        pathway = np.stack([rng.choice(modules_per_layer, active_modules, replace=False) for _ in range(num_layers)])
        self.register_buffer("pathway", torch.as_tensor(pathway, dtype=torch.long))
        self.pathway_attention = nn.Linear(hidden, 1)
        self.num_layers = num_layers

    def forward(self, x):
        for li, mods in enumerate(self.layers):
            outs = torch.stack([mods[int(m)](x) for m in self.pathway[li]])  # (K, B, H)
            if li < self.num_layers - 1:
                w = F.softmax(self.pathway_attention(outs).squeeze(-1), dim=0)
                x = (outs * w.unsqueeze(-1)).sum(0)
            else:
                x = outs.sum(0)
        return x


class MLPBackbone(nn.Module):
    """Plain IPPO backbone for the 'IPPO + same SafetyShield' baseline."""

    def __init__(self, input_dim, output_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, output_dim), nn.Tanh())

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    obs: np.ndarray
    proposal_idx: int
    proposal_log_prob: float
    value: float
    reward: float
    terminated: bool
    truncated: bool
    next_obs: np.ndarray
    executed_idx: int


@dataclass
class Proposal:
    action: str
    index: int
    log_prob: float
    value: float
    entropy: float
    #: pi(a | o) over CANONICAL_ACTIONS (masked actions = 0).  Only read by the evaluation-time comparison
    #: rules of ``HierarchicalFusionController.admission_rule`` (Reviewer 2 comments 3/4); never by training.
    probs: Optional[Tuple[float, ...]] = None


class PPOAgent(nn.Module):
    def __init__(self, agent_id: int, obs_dim: int, cfg: PPOConfig, seed: int, backbone: str = "pathnet", device="cpu",
                 action_set: Optional[Tuple[str, ...]] = None):
        super().__init__()
        self.agent_id = agent_id
        self.cfg = cfg
        self.device = torch.device(device)
        act_dim = len(CANONICAL_ACTIONS)
        feat = 256
        if backbone == "pathnet":
            self.feature_extractor = PathNet(
                obs_dim, feat, cfg.pathnet_layers, cfg.pathnet_modules_per_layer, cfg.pathnet_active_modules,
                cfg.pathnet_hidden, cfg.policy_dropout, seed=seed * 1000 + agent_id,
            )
        elif backbone == "mlp":
            self.feature_extractor = MLPBackbone(obs_dim, feat)
        else:
            raise ValueError(backbone)
        self.backbone = backbone
        self.policy_head = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(), nn.Linear(128, act_dim))
        allowed = torch.ones(act_dim, dtype=torch.bool)
        if action_set is not None:
            allowed = torch.tensor([a in action_set for a in CANONICAL_ACTIONS], dtype=torch.bool)
        self.register_buffer("action_mask", allowed)  # same mask for sampling, log-probs and the PPO update
        self.value_head = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(), nn.Linear(128, 1))
        self.to(self.device)
        self.actor_params = list(self.feature_extractor.parameters()) + list(self.policy_head.parameters())
        self.critic_params = list(self.value_head.parameters())
        self.actor_optimizer = torch.optim.AdamW(self.actor_params, lr=cfg.lr_actor)
        self.critic_optimizer = torch.optim.AdamW(self.critic_params, lr=cfg.lr_critic)
        self.base_lr_actor, self.base_lr_critic = cfg.lr_actor, cfg.lr_critic
        self.entropy_coef = cfg.entropy_coef
        self.beta_aux = cfg.beta_aux
        self.curriculum_ratio = cfg.curriculum_ratio
        self.normalizer = ObservationNormalizer(obs_dim)
        self.buffer: List[Transition] = []
        self.curriculum: deque = deque(maxlen=cfg.curriculum_buffer_size)
        self.update_count = 0
        self.last_stats: Dict[str, float] = {}
        self.entropy_history: deque = deque(maxlen=200)
        self.meta_hparams_applied = 0  # counts set_hyperparameters calls (proves meta output is consumed)

    # ------------------------------------------------------------------
    def forward(self, obs: torch.Tensor):
        f = self.feature_extractor(obs)
        logits = self.policy_head(f).masked_fill(~self.action_mask, -1e9)
        return logits, self.value_head(f).squeeze(-1)

    @torch.no_grad()
    def propose(self, norm_obs: np.ndarray, deterministic: bool, generator: Optional[torch.Generator] = None) -> Proposal:
        was = self.training
        self.eval()  # no stochastic layers when producing behaviour log-probs
        logits, value = self.forward(torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        self.train(was)
        dist = Categorical(logits=logits[0])
        if deterministic:
            idx = int(torch.argmax(logits[0]).item())
        else:
            # sample on the CPU with the seeded CPU generator (works for CPU and CUDA models and
            # keeps the sampled sequence identical across devices)
            probs = dist.probs.detach().to("cpu")
            idx = int(torch.multinomial(probs, 1, generator=generator).item())
        lp = float(dist.log_prob(torch.tensor(idx, device=self.device)).item())
        ent = float(dist.entropy().item())
        self.entropy_history.append(ent)
        probs = tuple(float(v) for v in dist.probs.detach().to("cpu").tolist())
        return Proposal(index_to_name(idx), idx, lp, float(value[0].item()), ent, probs)

    def log_prob_of(self, norm_obs: np.ndarray, action_idx: int) -> float:
        with torch.no_grad():
            logits, _ = self.forward(torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        return float(Categorical(logits=logits[0]).log_prob(torch.tensor(action_idx, device=self.device)).item())

    def store(self, t: Transition) -> None:
        self.buffer.append(t)

    def ready(self) -> bool:
        return len(self.buffer) >= self.cfg.buffer_size

    # ------------------------------------------------------------------
    def _gae(self, rewards, values, next_values, terminated, ended):
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * next_values[t] * (1.0 - terminated[t]) - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - ended[t]) * gae
            adv[t] = gae
        return adv

    def update(self) -> Optional[Dict[str, float]]:
        if not self.ready():
            return None
        batch, self.buffer = self.buffer, []
        cfg = self.cfg
        dev = self.device
        obs = torch.as_tensor(np.stack([b.obs for b in batch]), dtype=torch.float32, device=dev)
        nxt = torch.as_tensor(np.stack([b.next_obs for b in batch]), dtype=torch.float32, device=dev)
        acts = torch.as_tensor([b.proposal_idx for b in batch], dtype=torch.long, device=dev)
        old_lp = torch.as_tensor([b.proposal_log_prob for b in batch], dtype=torch.float32, device=dev)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        term = np.array([b.terminated for b in batch], dtype=np.float32)
        ended = np.array([b.terminated or b.truncated for b in batch], dtype=np.float32)
        ended[-1] = 1.0  # segment boundary: bootstrap through next_values only
        with torch.no_grad():
            _, values = self.forward(obs)
            _, next_values = self.forward(nxt)
        adv = self._gae(rewards, values.cpu().numpy(), next_values.cpu().numpy(), term, ended)
        returns = torch.as_tensor(adv, device=dev) + values
        adv_t = torch.as_tensor(adv, device=dev)
        if len(adv_t) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "aux_loss": 0.0}
        n_mb = 0
        g = torch.Generator().manual_seed(1_000_003 * (self.agent_id + 1) + self.update_count)
        # ratio sanity: before any gradient step pi_new == pi_old on the stored proposal actions
        with torch.no_grad():
            logits0, _ = self.forward(obs)
            lp0 = Categorical(logits=logits0).log_prob(acts)
            stats["initial_max_abs_log_ratio"] = float((lp0 - old_lp).abs().max().item())
        for _ in range(cfg.epochs):
            perm = torch.randperm(len(batch), generator=g)
            for s in range(0, len(batch), cfg.minibatch):
                mb = perm[s : s + cfg.minibatch].to(dev)
                logits, v = self.forward(obs[mb])
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(acts[mb])  # SAME action as the stored behaviour log-prob
                ratio = torch.exp(new_lp - old_lp[mb])
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_t[mb]
                ent = dist.entropy().mean()
                actor_loss = -torch.min(s1, s2).mean() - self.entropy_coef * ent
                aux = torch.zeros((), device=dev)
                if self.aux_enabled and len(self.curriculum) > 0:
                    k = max(1, int(round(self.curriculum_ratio * len(mb))))
                    ids = torch.randint(len(self.curriculum), (k,), generator=g).tolist()
                    co = torch.as_tensor(np.stack([self.curriculum[j][0] for j in ids]), dtype=torch.float32, device=dev)
                    ca = torch.as_tensor([self.curriculum[j][1] for j in ids], dtype=torch.long, device=dev)
                    aux = F.cross_entropy(self.forward(co)[0], ca)
                    actor_loss = actor_loss + self.beta_aux * aux
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_params, cfg.max_grad_norm)
                self.actor_optimizer.step()
                _, v2 = self.forward(obs[mb])
                critic_loss = F.mse_loss(v2, returns[mb])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_params, cfg.max_grad_norm)
                self.critic_optimizer.step()
                with torch.no_grad():
                    stats["approx_kl"] += float((old_lp[mb] - new_lp).mean().item())
                    stats["clip_frac"] += float(((ratio - 1).abs() > cfg.clip).float().mean().item())
                stats["actor_loss"] += float(actor_loss.item())
                stats["critic_loss"] += float(critic_loss.item())
                stats["entropy"] += float(ent.item())
                stats["aux_loss"] += float(aux.item())
                n_mb += 1
        for k in ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_frac", "aux_loss"):
            stats[k] /= max(1, n_mb)
        stats["batch_size"] = len(batch)
        stats["intervened_fraction"] = float(np.mean([b.proposal_idx != b.executed_idx for b in batch]))
        self.update_count += 1
        self.last_stats = stats
        return stats

    aux_enabled: bool = False

    def add_curriculum(self, norm_obs: np.ndarray, executed_idx: int) -> None:
        self.curriculum.append((np.asarray(norm_obs, dtype=np.float32), int(executed_idx)))

    # ------------------------------------------------------------------
    def set_hyperparameters(self, hp: Dict[str, float]) -> None:
        """Consumes every active meta-action dimension (asserted by the meta learner)."""
        for g in self.actor_optimizer.param_groups:
            g["lr"] = self.base_lr_actor * float(hp["lr_actor_factor"])
        for g in self.critic_optimizer.param_groups:
            g["lr"] = self.base_lr_critic * float(hp["lr_critic_factor"])
        self.entropy_coef = max(float(hp["entropy_coef"]), 1e-4)
        if "beta_aux" in hp:
            self.beta_aux = float(hp["beta_aux"])
        if "curriculum_ratio" in hp:
            self.curriculum_ratio = float(np.clip(hp["curriculum_ratio"], 0.0, 0.5))
        self.meta_hparams_applied += 1


# ===========================
# Hierarchical Fusion Controller: 两阶段在线融合 + 干预统计
# ===========================
# Two-pass online fusion pathway (paper Algorithm 1).
#
# Pass 1 (per agent):  proposal -> confidence-gated semantic admission -> SafetyShield
# Pass 2 (all agents): priority-based coordination over the complete set {a_i^safe}
#
# The SafetyShield instance is injected (dependency injection); the controller has no
# private safety rule of its own.

#: Semantic-admission rules.  "hard_gate" is PnAL (paper Eq. 25) and the ONLY rule used for training and for
#: every reported main result.  The other rules are evaluation-time comparisons on the SAME trained learned
#: source (Reviewer 2 comment 3: soft / product-of-experts fusion; comment 4: relative source reliability):
#:   soft_mixture        a_nom = argmax_a  kappa * 1[a = a_sp] + (1 - kappa) * pi(a | o)
#:   product_of_experts  a_nom = argmax_a  pi(a | o) * (kappa * 1[a = a_sp] + (1 - kappa) / |A|)
#:   relative_reliability admit a_sp iff kappa >= tau_c AND kappa >= max_a pi(a | o)
#: kappa is the confidence the preset uses (calibrated unless use_calibration=False); ties keep the proposal.
ADMISSION_RULES = ("hard_gate", "soft_mixture", "product_of_experts", "relative_reliability")


@dataclass
class AgentInput:
    agent_id: int
    proposal: Proposal
    state: SafetyState
    context: SymbolicContext
    vehicle: object
    #: one-step predictions x'_a for every canonical action (only for ShieldConfig.risk_mode="action_conditioned")
    predicted: Optional[Dict[str, SafetyState]] = None


@dataclass
class StepRecord:
    agent_id: int
    proposal_action: str
    proposal_log_prob: float
    semantic_candidate: Optional[str]
    semantic_template: Optional[str]
    semantic_match_score: float
    semantic_confidence_raw: float
    semantic_confidence_calibrated: float
    semantic_confidence_used: float
    semantic_matched: bool
    semantic_admitted: bool
    semantic_changed_action: bool
    nominal_action: str
    state_score: float
    effective_threshold: float
    admissible_actions: tuple
    safe_action: str
    shield_triggered: bool
    shield_intervened: bool
    shield_reason: str
    coordination_intervened: bool
    coordination_reason: str
    priority: float
    executed_action: str
    speed_floor_applied: bool = False  # nominal action changed by the nominal-speed floor

    def to_dict(self) -> Dict:
        return asdict(self)


class HierarchicalFusionController:
    def __init__(
        self,
        flags: ExperimentFlags,
        tau_c: float,
        risk_threshold: float,
        semantic_source: Optional[SemanticSource],
        safety_shield: SafetyShield,
        coordinator: PriorityCoordinator,
        speed_floor_index: Optional[int] = None,
        admission_rule: str = "hard_gate",
        delta_temp_rule: str = "paper_max",
        delta_temp_cap: float = 1.0,
    ):
        if admission_rule not in ADMISSION_RULES:
            raise ValueError(f"admission_rule must be one of {ADMISSION_RULES}")
        self.admission_rule = admission_rule
        if delta_temp_rule not in ("paper_max", "none", "cap"):
            raise ValueError("delta_temp_rule must be 'paper_max', 'none' or 'cap'")
        self.delta_temp_rule, self.delta_temp_cap = delta_temp_rule, float(delta_temp_cap)
        self.flags = flags
        self.speed_floor_index = speed_floor_index
        self.tau_c = float(tau_c)  # the single admission threshold
        self.delta_t = float(risk_threshold)
        self.semantic = semantic_source
        self.shield = safety_shield  # shared instance (dependency injection)
        self.coordinator = coordinator
        #: wall-clock seconds spent per online stage (Reviewer 1: latency / overhead report)
        self.timing: Dict[str, float] = defaultdict(float)
        self.timing_calls: Dict[str, int] = defaultdict(int)
        if flags.use_semantic and semantic_source is None:
            raise ValueError("use_semantic=True requires a SemanticSource")
        if flags.use_semantic and flags.use_calibration and not flags.always_trust_semantic:
            cal = semantic_source.calibrator
            if cal is None or not cal.fitted:
                raise ValueError("use_calibration=True requires a fitted SemanticConfidenceCalibrator")

    # ------------------------------------------------------------------
    def _admit(self, q) -> tuple:
        if not q.matched:
            return False, 0.0
        used = q.calibrated_confidence if self.flags.use_calibration else q.raw_confidence
        if self.flags.always_trust_semantic:
            return True, used
        return bool(used >= self.tau_c), used  # the ONLY admission gate

    def _nominal(self, q, p) -> tuple:
        """(nominal action, admitted, confidence used) under ``self.admission_rule``."""
        admitted, used = self._admit(q) if q is not None else (False, 0.0)
        rule = self.admission_rule
        if rule == "hard_gate" or q is None or not q.matched or self.flags.always_trust_semantic:
            return (q.semantic_action if admitted else p.action), admitted, used
        if rule not in ADMISSION_RULES:
            raise ValueError(f"unknown admission_rule {rule}")
        if p.probs is None:
            raise RuntimeError(f"admission_rule={rule} needs the policy distribution (Proposal.probs)")
        pi = np.asarray(p.probs, dtype=float)
        k = float(used)
        if rule == "relative_reliability":
            adm = bool(k >= self.tau_c and k >= float(pi.max()))
            return (q.semantic_action if adm else p.action), adm, k
        sem = np.zeros(len(CANONICAL_ACTIONS))
        sem[CANONICAL_ACTIONS.index(q.semantic_action)] = 1.0
        if rule == "soft_mixture":
            score = k * sem + (1.0 - k) * pi
        else:  # product_of_experts
            score = pi * (k * sem + (1.0 - k) / len(CANONICAL_ACTIONS))
        best = float(score.max())
        p_idx = CANONICAL_ACTIONS.index(p.action)
        a = p.action if score[p_idx] >= best - 1e-12 else CANONICAL_ACTIONS[int(np.argmax(score))]
        return a, bool(a == q.semantic_action), k

    def _template_threshold(self, delta_temp: float) -> float:
        """delta_temp' of an admitted template (SemanticConfig.delta_temp_rule)."""
        if self.delta_temp_rule == "none":
            return 0.0
        if self.delta_temp_rule == "cap":
            return min(float(delta_temp), self.delta_temp_cap)
        return float(delta_temp)

    def _apply_speed_floor(self, nominal: str, inp) -> tuple:
        """Nominal-speed floor (shared implementation: ``actions.apply_nominal_speed_floor``)."""
        x = inp.state
        return apply_nominal_speed_floor(nominal, int(getattr(inp.vehicle, "speed_index", 0)), self.speed_floor_index,
                                         x.left_lane_exists, x.right_lane_exists)
        idx = int(getattr(inp.vehicle, "speed_index", 0))
        f = self.speed_floor_index
        x = inp.state
        lateral_noop = (nominal == "LANE_LEFT" and not x.left_lane_exists) or (nominal == "LANE_RIGHT" and not x.right_lane_exists)
        if idx < f and (nominal in ("MAINTAIN", "DECELERATE") or lateral_noop):
            return "ACCELERATE", True  # resume towards the floor (the shield still checks it)
        if idx <= f and nominal == "DECELERATE":
            return "MAINTAIN", True    # the nominal action cannot command a speed below the floor
        return nominal, False

    def _tick(self, key: str, t0: float) -> float:
        t1 = time.perf_counter()
        self.timing[key] += t1 - t0
        self.timing_calls[key] += 1
        return t1

    def latency_summary_ms(self) -> Dict[str, float]:
        """Mean wall-clock milliseconds per call of each online stage."""
        return {k: 1e3 * self.timing[k] / max(1, self.timing_calls[k]) for k in sorted(self.timing)}

    def reset_timing(self) -> None:
        self.timing.clear()
        self.timing_calls.clear()

    def decide(self, inputs: Sequence[AgentInput]) -> List[StepRecord]:
        partial = []
        # ---------------- Pass 1 -----------------
        for inp in inputs:
            p = inp.proposal
            t0 = time.perf_counter()
            q = self.semantic.query(inp.context) if self.flags.use_semantic else None
            nominal, admitted, used = self._nominal(q, p)
            nominal, floored = self._apply_speed_floor(nominal, inp)
            # delta_temp is part of the semantic prior: by default it applies only when the template is
            # ADMITTED by the calibrated gate (a rejected, unreliable source must not tighten the shield).
            # flags.delta_temp_when_matched=True reproduces the literal Algorithm 1 (matched => delta_temp).
            if q is not None and q.matched and (admitted or self.flags.delta_temp_when_matched):
                delta_temp = self._template_threshold(q.delta_temp)
            else:
                delta_temp = 0.0
            delta_eff = max(self.delta_t, delta_temp)
            if self.flags.use_semantic:
                t0 = self._tick("semantic_match_and_gate", t0)
            s = self.shield.state_score(inp.state)
            t0 = self._tick("neuro_fuzzy_state_score", t0)
            if self.flags.use_safety_shield and self.shield.cfg.risk_mode == "action_conditioned":
                if inp.predicted is None:
                    raise RuntimeError("risk_mode='action_conditioned' needs AgentInput.predicted")
                scores = {a: self.shield.state_score(xa) for a, xa in inp.predicted.items()}
                d = self.shield.apply_safety_constraints_action_conditioned(nominal, inp.state, scores, delta_eff)
                safe, adm, trig, interv, reason = d.safe_action, d.admissible, d.triggered, d.intervened, d.reason
                fallback_list = self.shield.ranked_admissible_action_conditioned(adm, scores, delta_eff)
                self._tick("safety_shield_A_hard_and_fallback", t0)
            elif self.flags.use_safety_shield:
                d = self.shield.apply_safety_constraints(nominal, inp.state, s, delta_eff)
                safe, adm, trig, interv, reason = d.safe_action, d.admissible, d.triggered, d.intervened, d.reason
                fallback_list = self.shield.ranked_admissible(adm)
                self._tick("safety_shield_A_hard_and_fallback", t0)
            else:
                safe, adm, trig, interv, reason = nominal, tuple(CANONICAL_ACTIONS), False, False, "shield_disabled"
                fallback_list = list(self.shield.cfg.fallback_ranking)
            partial.append((inp, q, admitted, used, nominal, s, delta_eff, safe, adm, trig, interv, reason, fallback_list, floored))
        # ---------------- Pass 2 -----------------
        if self.flags.use_coordination:
            cands = [
                CoordCandidate(t[0].agent_id, t[7], t[12], t[0].state, t[5], t[0].vehicle) for t in partial
            ]
            t0 = time.perf_counter()
            coord, _info = self.coordinator.coordinate(cands)
            self._tick("coordination_per_joint_step", t0)
            self.last_coordination_info = _info
        else:
            coord = None
            self.last_coordination_info = {}
        records = []
        for k, (inp, q, admitted, used, nominal, s, delta_eff, safe, adm, trig, interv, reason, _, floored) in enumerate(partial):
            if coord is not None:
                executed, c_int, c_reason, prio = coord[k].executed_action, coord[k].intervened, coord[k].reason, coord[k].priority
            else:
                executed, c_int, c_reason, prio = safe, False, "coordination_disabled", float("nan")
            records.append(
                StepRecord(
                    agent_id=inp.agent_id,
                    proposal_action=inp.proposal.action,
                    proposal_log_prob=inp.proposal.log_prob,
                    semantic_candidate=q.semantic_action if q is not None else None,
                    semantic_template=q.template_key if q is not None else None,
                    semantic_match_score=q.match_score if q is not None else 0.0,
                    semantic_confidence_raw=q.raw_confidence if q is not None else 0.0,
                    semantic_confidence_calibrated=q.calibrated_confidence if q is not None else 0.0,
                    semantic_confidence_used=used,
                    semantic_matched=bool(q is not None and q.matched),
                    semantic_admitted=admitted,
                    semantic_changed_action=bool(admitted and q is not None and q.semantic_action != inp.proposal.action),
                    nominal_action=nominal,
                    state_score=s,
                    effective_threshold=delta_eff,
                    admissible_actions=tuple(adm),
                    safe_action=safe,
                    shield_triggered=trig,
                    shield_intervened=interv,
                    shield_reason=reason,
                    coordination_intervened=c_int,
                    coordination_reason=c_reason,
                    priority=prio,
                    executed_action=executed,
                    speed_floor_applied=floored,
                )
            )
        return records


def _physically_unchanged(r: "StepRecord") -> bool:
    if r.proposal_action == r.executed_action:
        return True
    return (r.proposal_action in ("LANE_LEFT", "LANE_RIGHT") and r.executed_action == "MAINTAIN"
            and r.shield_reason.startswith("topology_noop") and not r.semantic_changed_action
            and not r.coordination_intervened and not r.speed_floor_applied)


class InterventionStatistics:
    """Aggregates per-step records into the rates requested by Reviewer 1 / Reviewer 2."""

    def __init__(self):
        self.records: List[StepRecord] = []

    def add(self, recs: Sequence[StepRecord]) -> None:
        self.records.extend(recs)

    def summary(self) -> Dict[str, float]:
        R = self.records
        n = len(R)
        if n == 0:
            return {"agent_steps": 0}
        m = lambda f: float(np.mean([f(r) for r in R]))
        n_changes = [int(r.semantic_changed_action) + int(r.shield_intervened) + int(r.coordination_intervened) for r in R]
        matched = [r for r in R if r.semantic_matched]
        out = {
            "agent_steps": n,
            "semantic_match_rate": m(lambda r: r.semantic_matched),
            "semantic_admission_rate": m(lambda r: r.semantic_admitted),
            "semantic_admission_rate_given_match": float(np.mean([r.semantic_admitted for r in matched])) if matched else None,  # undefined without matches
            "semantic_action_change_rate": m(lambda r: r.semantic_changed_action),
            "shield_trigger_rate": m(lambda r: r.shield_triggered),
            "shield_intervention_rate": m(lambda r: r.shield_intervened),
            "shield_emergency_rate": m(lambda r: r.shield_reason == "emergency_empty_set"),
            # lateral no-op on a single-lane road remapped to MAINTAIN: changes the action LABEL but not
            # the executed physics -> report separately from safety-relevant interventions
            "shield_topology_remap_rate": m(lambda r: r.shield_reason.startswith("topology_noop")),
            "speed_floor_rate": m(lambda r: r.speed_floor_applied),
            "coordination_intervention_rate": m(lambda r: r.coordination_intervened),
            "proposal_executed_agreement": m(lambda r: r.proposal_action == r.executed_action),
            # shield changes that are not the physically-neutral lateral no-op relabelling
            "shield_safety_intervention_rate": m(lambda r: r.shield_intervened and not r.shield_reason.startswith("topology_noop")),
            # executed action physically equal to the proposal (a lateral no-op relabelled MAINTAIN counts as equal)
            "proposal_executed_physical_agreement": m(_physically_unchanged),
            "double_intervention_rate": float(np.mean([c >= 2 for c in n_changes])),
            "triple_intervention_rate": float(np.mean([c == 3 for c in n_changes])),
            "mean_state_score": m(lambda r: r.state_score),
        }
        reasons: Dict[str, int] = {}
        for r in R:
            if r.shield_triggered:
                key = r.shield_reason.split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
        out["shield_reason_counts"] = reasons
        return out


# ===========================
# SAC Meta-learner Components (training-time adaptation only)
# ===========================
# Training-time adaptation interface: SAC meta-scheduler (never an online fusion source).
#
# Audit of the original implementation
# ------------------------------------
# * PATHNET_BATCH_SIZE = 128 while TOTAL_PATHNET_EPISODES = 100 with ONE meta transition per
#   meta episode -> the replay buffer never reached the batch size and
#   ``update_meta_learner`` never performed a gradient step (meta_update_count == 0).
# * 4 of the 9 meta-action dimensions (focus_intensity, update_ratio, exploration_bonus,
#   safety_weight) were never read by any computation; ``beta_aux`` was stored but no auxiliary
#   loss existed.  Only lr_actor_factor, lr_critic_factor, entropy_coef, curriculum_ratio
#   reached training.  The revision keeps only consumed dimensions.
# * The meta reward (orig. ``calculate_fused_meta_reward``) referenced a decision-cache
#   hit-rate ('efficiency'), curriculum growth/quality and a 'coordination_score' of the removed
#   coordination_matrix; these terms are removed together with the components they measured.
#   Its ``safety_violations`` counter was always 0 because of the ``'safety' in list`` bug; the
#   revision keeps that term with the SAME weights but feeds it the bug-fixed quantity
#   (per-step SafetyShield intervention RATE instead of a raw count/100).  The two bonus/penalty
#   rules that depended only on ``safety_violations`` (+80 / -50) are dropped because their
#   count-based thresholds (<10, >50) have no meaning for a rate.  Success, return, collision
#   and state-safety-score terms, phase weights, the +100 bonus, the -100 penalty and the
#   reward scale (1000) are unchanged.

LOG_STD_MIN, LOG_STD_MAX = -20, 2


def build_meta_action_space(flags: ExperimentFlags) -> List[Tuple[str, Tuple[float, float]]]:
    names = ["lr_actor_factor", "lr_critic_factor", "entropy_coef"]
    if flags.use_curriculum_aux:
        names += ["beta_aux", "curriculum_ratio"]
    return [(n, META_ACTION_RANGES[n]) for n in names]


META_STATE_FEATURES = (
    "success_rate",
    "collision_rate",
    "offroad_rate",
    "timeout_rate",
    "mean_team_return_scaled",
    "mean_state_safety_score",
    "shield_intervention_rate",
    "semantic_admission_rate",
    "coordination_intervention_rate",
    "mean_policy_entropy",
    "training_progress",
)


class SACActor(nn.Module):
    def __init__(self, obs_dim, act_dim, ranges, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        nn.init.xavier_uniform_(self.mean.weight, gain=0.01)
        nn.init.xavier_uniform_(self.log_std.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.bias)
        lo = torch.tensor([r[0] for r in ranges], dtype=torch.float32)
        hi = torch.tensor([r[1] for r in ranges], dtype=torch.float32)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    def sample(self, s, deterministic=False, generator=None):
        h = self.net(s)
        mean, log_std = self.mean(h), self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        if deterministic:
            x = mean
        elif generator is not None:
            x = mean + std * torch.randn(mean.shape, generator=generator)
        else:
            x = Normal(mean, std).rsample()
        y = torch.tanh(x)
        logp = (Normal(mean, std).log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)).sum(-1, keepdim=True)
        action = self.lo + (y + 1) * 0.5 * (self.hi - self.lo)  # environment scale
        y_mean = torch.tanh(mean)
        return action, y, logp, y_mean


def _critic(in_dim, dropout):
    return nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1))


class SACMetaScheduler:
    """Stores the NORMALISED action y in [-1,1]^d in the replay buffer and feeds y to the critics.
    (The original fed environment-scale actions consistently; normalising only removes the
    very different scales of lr factors [0.2,2] and entropy coefficients [5e-4,5e-2].)"""

    def __init__(self, state_dim: int, action_space, cfg: MetaConfig, seed: int, device="cpu"):
        self.cfg = cfg
        self.names = [n for n, _ in action_space]
        self.ranges = [r for _, r in action_space]
        self.state_dim, self.action_dim = state_dim, len(action_space)
        self.device = torch.device(device)
        self.actor = SACActor(state_dim, self.action_dim, self.ranges).to(self.device)
        self.q1 = _critic(state_dim + self.action_dim, cfg.critic_dropout).to(self.device)
        self.q2 = _critic(state_dim + self.action_dim, cfg.critic_dropout).to(self.device)
        self.q1_t, self.q2_t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.opt_a = torch.optim.AdamW(self.actor.parameters(), lr=cfg.lr_actor, weight_decay=1e-4)
        self.opt_q1 = torch.optim.AdamW(self.q1.parameters(), lr=cfg.lr_critic, weight_decay=1e-4)
        self.opt_q2 = torch.optim.AdamW(self.q2.parameters(), lr=cfg.lr_critic, weight_decay=1e-4)
        self.log_alpha = torch.tensor(np.log(cfg.init_temperature), dtype=torch.float32, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        self.target_entropy = -float(self.action_dim)
        self.replay: deque = deque(maxlen=cfg.replay_size)
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.meta_update_count = 0
        self.history: List[Dict[str, float]] = []

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select(self, state: np.ndarray, deterministic: bool) -> Tuple[Dict[str, float], np.ndarray]:
        self.actor.eval()
        with torch.no_grad():
            a, y, _, _ = self.actor.sample(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0), deterministic, self.gen)
        self.actor.train()
        hp = {n: float(v) for n, v in zip(self.names, a[0].tolist())}
        return hp, y[0].numpy()

    def store(self, s, y, r, s2, done) -> None:
        self.replay.append((np.asarray(s, np.float32), np.asarray(y, np.float32), float(r), np.asarray(s2, np.float32), float(done)))

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.replay) < self.cfg.batch_size:
            return None
        batch = self.rng.sample(list(self.replay), self.cfg.batch_size)
        s, y, r, s2, d = (torch.as_tensor(np.array(z), dtype=torch.float32) for z in zip(*batch))
        r, d = r.unsqueeze(1), d.unsqueeze(1)
        with torch.no_grad():
            _, y2, lp2, _ = self.actor.sample(s2, generator=self.gen)
            q_t = torch.min(self.q1_t(torch.cat([s2, y2], 1)), self.q2_t(torch.cat([s2, y2], 1))) - self.alpha * lp2
            target = r + self.cfg.gamma * (1 - d) * q_t
        l1 = F.mse_loss(self.q1(torch.cat([s, y], 1)), target)
        l2 = F.mse_loss(self.q2(torch.cat([s, y], 1)), target)
        self.opt_q1.zero_grad(); l1.backward(); nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0); self.opt_q1.step()
        self.opt_q2.zero_grad(); l2.backward(); nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0); self.opt_q2.step()
        _, yp, lp, _ = self.actor.sample(s)
        qp = torch.min(self.q1(torch.cat([s, yp], 1)), self.q2(torch.cat([s, yp], 1)))
        actor_loss = (self.alpha.detach() * lp - qp).mean()
        self.opt_a.zero_grad(); actor_loss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.opt_a.step()
        alpha_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()
        with torch.no_grad():
            for t, p in ((self.q1_t, self.q1), (self.q2_t, self.q2)):
                for tp, pp in zip(t.parameters(), p.parameters()):
                    tp.mul_(1 - self.cfg.tau).add_(self.cfg.tau * pp)
        self.meta_update_count += 1
        rec = {
            "meta_replay_size": len(self.replay),
            "meta_update_count": self.meta_update_count,
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float((l1.item() + l2.item()) / 2),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
        }
        self.history.append(rec)
        return rec

    def state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_t": self.q1_t.state_dict(), "q2_t": self.q2_t.state_dict(),
            "log_alpha": float(self.log_alpha.item()), "names": self.names, "ranges": self.ranges,
            "meta_update_count": self.meta_update_count,
        }


PHASE_WEIGHTS = {
    # retained terms of the original calculate_fused_meta_reward (same weights)
    "exploration": {"success": 50.0, "reward": 30.0, "collision": 100.0, "safety": 80.0, "violations": 50.0},
    "refinement": {"success": 100.0, "reward": 50.0, "collision": 150.0, "safety": 100.0, "violations": 80.0},
    "exploitation": {"success": 150.0, "reward": 100.0, "collision": 200.0, "safety": 120.0, "violations": 100.0},
}


def training_phase(progress: float) -> str:
    if progress < 0.01:
        return "exploration"
    if progress < 0.30:
        return "refinement"
    return "exploitation"


def meta_reward(before: Dict[str, float], after: Dict[str, float], phase: str, scale: float) -> float:
    w = PHASE_WEIGHTS[phase]
    r = (
        w["success"] * (after["success_rate"] - before["success_rate"])
        + w["reward"] * np.tanh((after["mean_team_return"] - before["mean_team_return"]) / 100.0)
        + w["collision"] * (before["collision_rate"] - after["collision_rate"])
        + w["safety"] * (after["mean_state_safety_score"] - before["mean_state_safety_score"])
        + w["violations"] * np.tanh(before["shield_intervention_rate"] - after["shield_intervention_rate"])
    )
    if after["success_rate"] > 0.8 and after["collision_rate"] < 0.05:
        r += 100.0
    if before["collision_rate"] - after["collision_rate"] < -0.1:
        r -= 100.0
    return float(r / scale)


# ===========================
# Pre-deployment Calibration Stage
# ===========================
# Pre-deployment calibration stage (run BEFORE MARL training; outputs are frozen).
#
# Data: dedicated calibration episodes whose environment seeds come from the reserved
# ``calibration`` seed range (disjoint from train / validation / test ranges).
#
# Behaviour policy during calibration rollouts: if a template matches, the agent executes
# the semantic recommendation a_sp (so that recommendations receive outcome labels);
# otherwise it samples a primitive action uniformly (exploration of risky states).
#
# Episode-level split of calibration episodes:
#     A (40%) -> validation reliability q_bar per template
#     B (40%) -> Platt scaling fit (raw_kappa computed with q_bar from A)
#     C (20%) -> calibration diagnostics (ECE / Brier / NLL / reliability bins)
# The neuro-fuzzy state-risk model is fitted on the states of A u B (its own internal
# held-out split reports AUROC / Brier).
#
# Deployment consistency (report #52): every nominal action in these rollouts passes through the same
# ``actions.apply_nominal_speed_floor`` as at deployment and then through the SafetyShield hard-feasibility
# screening (A_hard + fallback; the NF risk gate is not applied because the NF model is fitted from these
# rollouts).  A semantic recommendation is labelled on the nominal action deployment would produce from it.
#
# Labels
#     state_safe(t)            = 1 iff no collision, off-road or near-miss for the agent in
#                                decision steps t+1 .. t+H_nf
#     semantic_acceptability(t)= 1 iff a_sp in A_hard(x_t) and no collision, off-road or
#                                near-miss for the agent in steps t+1 .. t+H_sem

class CalibrationDataError(RuntimeError):
    pass


@dataclass
class SemanticRecord:
    episode: int
    agent: int
    step: int
    template_key: str
    match_score: float
    template_confidence: float
    semantic_action: str
    admissible: bool
    label: int = -1
    #: True iff the behaviour policy followed the recommendation at this matched step (randomised, p_exec)
    executed: bool = True
    #: nominal action actually executed (after the speed floor)
    executed_action: str = ""
    #: outcome WITHOUT the admissibility term; defined for followed and not-followed steps (refinement)
    outcome: int = -1


@dataclass
class CalibrationArtifacts:
    nf_model: ADARSOFFENStateRisk
    nf_report: Dict
    library: SemanticLibrary
    calibrator: Optional[SemanticConfidenceCalibrator]
    isotonic_calibrator: Optional[SemanticConfidenceCalibrator]
    diagnostics: Dict
    label_definitions: Dict
    counts: Dict
    refinement: Optional[Dict] = None


def _future_event(events: List[bool], t: int, H: int) -> bool:
    """events[k] refers to state k (after k transitions); check k in t+1..t+H."""
    return any(events[k] for k in range(t + 1, min(t + H, len(events) - 1) + 1))


def _base_behaviour_action(cfg: PnALConfig, vehicle, rng) -> str:
    """Behaviour for steps that do not execute a_sp (fixed heuristic, independent of the learned policy).

    "uniform"        : uniformly random primitive action.
    "cruise_epsilon" : with prob. epsilon a uniformly random action, otherwise progress-seeking
                       cruising (ACCELERATE below the highest target speed, else MAINTAIN).  This keeps
                       task completion possible, so that a task-aware acceptability label measures the
                       recommendation rather than the randomness of the calibration behaviour.
    """
    b = cfg.semantic.calibration_behaviour
    if b == "uniform" or (b == "cruise_epsilon" and rng.uniform() < cfg.semantic.calibration_epsilon):
        return CANONICAL_ACTIONS[int(rng.integers(0, len(CANONICAL_ACTIONS)))]
    if b != "cruise_epsilon":
        raise ValueError(f"unknown calibration behaviour {b!r}")
    speeds = getattr(vehicle, "target_speeds", [])
    return "ACCELERATE" if int(getattr(vehicle, "speed_index", 0)) < len(speeds) - 1 else "MAINTAIN"


def _floored(cfg: PnALConfig, action: str, vehicle, x) -> str:
    return apply_nominal_speed_floor(action, int(getattr(vehicle, "speed_index", 0)), cfg.scenario.nominal_speed_floor_index,
                                     x.left_lane_exists, x.right_lane_exists)[0]


def _executed(cfg: PnALConfig, nominal: str, x, shield: SafetyShield) -> str:
    """Deployment-consistent execution of a nominal action in the pre-deployment rollouts (report #52):
    hard-feasibility screening only (state score 1, threshold 0 => the NF risk gate never fires)."""
    if not cfg.semantic.rollout_hard_shield:
        return nominal
    return shield.apply_safety_constraints(nominal, x, 1.0, 0.0).safe_action


def collect_semantic_rollouts(cfg: PnALConfig, env: PnALEnv, extractor: SafetyStateExtractor, library: SemanticLibrary,
                              shield_checker: SafetyShield, n_episodes: int, split: str = "calibration", episode_offset: int = 0):
    """Randomised rollouts with a fixed behaviour policy (independent of the learned policy).

    At a matched step the recommendation a_sp is followed with probability p_exec (randomised), otherwise
    the base behaviour acts; unmatched steps use the base behaviour.  Every executed action passes the
    deployment speed floor.  Returns NF states/labels and one SemanticRecord per MATCHED step (followed or
    not); only followed records carry a calibration label."""
    rng = np.random.default_rng(cfg.seeds.env_seed(split, episode_offset))
    states: List[Tuple[int, np.ndarray, int, int]] = []  # (episode, features, agent, step)
    labels_nf: List[int] = []
    sem: List[SemanticRecord] = []
    H_nf, H_sem = cfg.neuro_fuzzy.label_horizon_steps, cfg.semantic.label_horizon_steps
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=cfg.seeds.env_seed(split, episode_offset + ep))
        n = env.n_agents
        feats: List[List[np.ndarray]] = [[] for _ in range(n)]
        events: List[List[bool]] = [[False] for _ in range(n)]  # state 0 has no event by construction
        ep_sem: List[SemanticRecord] = []
        t = 0
        while True:
            actions = []
            for i in range(n):
                x = extractor.extract(i, obs[i])
                feats[i].append(extractor.nf_features(x))
                ctx = extractor.symbolic_context(x)
                tmpl, score = library.find_matching_template(ctx, cfg.semantic.match_min_score)
                if tmpl is not None:
                    follow = bool(rng.uniform() < cfg.semantic.calibration_execute_prob)
                    a_sp = tmpl.top_action(ctx)
                    nominal = _floored(cfg, a_sp if follow else _base_behaviour_action(cfg, env.vehicles[i], rng), env.vehicles[i], x)
                    adm = shield_checker.check_action(nominal, x) == [] if follow else False
                    a = _executed(cfg, nominal, x, shield_checker)
                    ep_sem.append(SemanticRecord(ep, i, t, tmpl.key, score, tmpl.confidence, a_sp, adm,
                                                 executed=follow, executed_action=a))
                else:
                    a = _executed(cfg, _floored(cfg, _base_behaviour_action(cfg, env.vehicles[i], rng), env.vehicles[i], x), x, shield_checker)
                actions.append(a)
            res = env.step(actions)
            obs = res.observations
            t += 1
            for i in range(n):
                x2 = extractor.extract(i, obs[i])
                events[i].append(bool(res.newly_crashed[i] or env.crashed[i] or res.newly_offroad[i] or x2.near_miss))
            if res.terminated or res.truncated:
                break
        for i in range(n):
            for k, f in enumerate(feats[i]):
                states.append((episode_offset + ep, f, i, k))
                labels_nf.append(0 if _future_event(events[i], k, H_nf) else 1)
        completed = list(env.arrived)  # per-agent task completion (latched) at episode end
        for r in ep_sem:
            safe_h = not _future_event(events[r.agent], r.step, H_sem)
            if cfg.semantic.label_definition == "safety_only":
                r.outcome = int(safe_h)
            elif cfg.semantic.label_definition == "safety_and_task_completion":
                r.outcome = int(safe_h and completed[r.agent] and not env.crashed[r.agent])
            else:
                raise ValueError(f"unknown semantic label definition {cfg.semantic.label_definition!r}")
            r.label = int(r.admissible and r.outcome) if r.executed else -1
            r.episode = episode_offset + r.episode
        sem.extend(ep_sem)
    return states, np.array(labels_nf, dtype=np.int64), sem


def collect_calibration_rollouts(cfg: PnALConfig, env: PnALEnv, extractor: SafetyStateExtractor, library: SemanticLibrary, shield_checker: SafetyShield, n_episodes: int):
    """Calibration-split rollouts; returns only the FOLLOWED semantic records (the labelled ones)."""
    states, y, sem = collect_semantic_rollouts(cfg, env, extractor, library, shield_checker, n_episodes, "calibration")
    return states, y, [r for r in sem if r.executed]


def run_calibration_stage(cfg: PnALConfig, env: PnALEnv, extractor: SafetyStateExtractor, n_episodes: Optional[int] = None,
                          library: Optional[SemanticLibrary] = None, refinement: Optional[Dict] = None) -> CalibrationArtifacts:
    """``library``: the (refined, not yet frozen) library to calibrate; default = the configured initial library."""
    n_episodes = n_episodes or cfg.run.calibration_episodes
    library = library if library is not None else SemanticLibrary.from_config(cfg.semantic)
    if library.frozen:
        raise ValueError("calibration needs the unfrozen library (it sets the reliabilities and then freezes it)")
    # admissibility checks do not depend on the NF model; use an unfitted placeholder
    placeholder = SafetyShield(ADARSOFFENStateRisk(len(NF_FEATURE_NAMES), cfg.neuro_fuzzy.n_rules), extractor.nf_features, cfg.shield, 1.0 / cfg.scenario.policy_frequency)
    states, y_nf, sem = collect_calibration_rollouts(cfg, env, extractor, library, placeholder, n_episodes)
    n_a = int(0.4 * n_episodes)
    n_b = int(0.8 * n_episodes)
    split = lambda ep: "A" if ep < n_a else ("B" if ep < n_b else "C")

    # ---- neuro-fuzzy state-risk fitting on A u B
    X = np.stack([s[1] for s in states])
    ep_of = np.array([s[0] for s in states])
    fit_mask = np.array([split(e) in ("A", "B") for e in ep_of])
    nf = ADARSOFFENStateRisk(len(NF_FEATURE_NAMES), cfg.neuro_fuzzy.n_rules)
    nf_report = StateRiskFitter(cfg.neuro_fuzzy, seed=cfg.seeds.seed).fit(nf, X[fit_mask], y_nf[fit_mask])
    with np.errstate(all="ignore"):
        pc = np.array([nf.score(x) for x in X[~fit_mask]]) if (~fit_mask).any() else np.array([])
    nf_holdout = calibration_diagnostics(pc, y_nf[~fit_mask]) if len(pc) else {}

    # ---- semantic confidence
    recA = [r for r in sem if split(r.episode) == "A"]
    recB = [r for r in sem if split(r.episode) == "B"]
    recC = [r for r in sem if split(r.episode) == "C"]
    q_bar = estimate_validation_reliability([(r.template_key, r.label) for r in recA], cfg.semantic, [t.key for t in library.templates])
    library.set_validation_reliability(q_bar)
    q_level = estimate_level_reliability([(r.template_key, r.match_score, r.label) for r in recA], cfg.semantic)
    library.set_level_reliability(q_level)
    tmp_source = SemanticSource(library, cfg.semantic, calibrator=None)
    tmpl_by_key = {t.key: t for t in library.templates}
    raw = lambda recs: np.array([tmp_source.raw_confidence(tmpl_by_key[r.template_key], r.match_score) for r in recs])
    counts = {
        "calibration_episodes": n_episodes,
        "nf_states": int(len(X)),
        "nf_positive_rate": float(y_nf.mean()),
        "semantic_records": {"A": len(recA), "B": len(recB), "C": len(recC)},
        "semantic_label_rate_B": float(np.mean([r.label for r in recB])) if recB else float("nan"),
        "template_usage": {k: sum(r.template_key == k for r in sem) for k in tmpl_by_key},
        "q_bar": q_bar,
        "q_level": q_level,
        "raw_confidence_mode": cfg.semantic.raw_confidence_mode,
        "n_templates": len(library.templates),
    }
    calibrator = iso = None
    diagnostics: Dict = {"neuro_fuzzy_holdout_C": nf_holdout, "neuro_fuzzy_fit": asdict(nf_report)}
    yB = np.array([r.label for r in recB])
    if cfg.flags.use_semantic and not library.templates:
        # explicit, recorded outcome of the refinement stage (never a silent fallback): every template was
        # removed, so the semantic source can never match and PnAL runs without a semantic prior.
        diagnostics["semantic_source_inactive"] = "refinement removed every template"
    elif cfg.flags.use_semantic:
        if len(recB) < cfg.semantic.min_calibration_samples or len(np.unique(yB)) < 2:
            raise CalibrationDataError(
                f"Semantic calibration split B has {len(recB)} records with labels {np.unique(yB).tolist()} "
                f"(need >= {cfg.semantic.min_calibration_samples} and both classes). Increase "
                "run.calibration_episodes; do NOT fall back to an uncalibrated gate silently."
            )
        calibrator = SemanticConfidenceCalibrator("platt").fit(raw(recB), yB)
        iso = SemanticConfidenceCalibrator("isotonic").fit(raw(recB), yB)
        if recC:
            yC = np.array([r.label for r in recC])
            diagnostics["semantic_C_platt"] = calibration_diagnostics(calibrator.predict(raw(recC)), yC)
            diagnostics["semantic_C_isotonic"] = calibration_diagnostics(iso.predict(raw(recC)), yC)
            diagnostics["semantic_C_uncalibrated_raw"] = calibration_diagnostics(raw(recC), yC)
            diagnostics["semantic_C_per_template_accuracy"] = {
                k: float(np.mean([r.label for r in recC if r.template_key == k])) for k in tmpl_by_key if any(r.template_key == k for r in recC)
            }
        calibrator.freeze()
        iso.freeze()
    library.freeze()
    labels = {
        "state_safe": f"1 iff no collision/off-road/near-miss for the agent in the next {cfg.neuro_fuzzy.label_horizon_steps} decision steps",
        "semantic_acceptability": (
            f"[{cfg.semantic.label_definition}] 1 iff executed a_sp is in A_hard(x_t) and no collision/off-road/near-miss "
            f"of the agent in the next {cfg.semantic.label_horizon_steps} decision steps"
            + (" and the agent completes its task objective in that calibration episode"
               if cfg.semantic.label_definition == "safety_and_task_completion" else "")
        ),
        "behaviour_policy": (f"matched context: execute a_sp with prob {cfg.semantic.calibration_execute_prob}; otherwise "
                             f"'{cfg.semantic.calibration_behaviour}'"
                             + (f" (epsilon={cfg.semantic.calibration_epsilon})" if cfg.semantic.calibration_behaviour == "cruise_epsilon" else "")),
        "near_miss": f"min(TTC_lead, TTC_conflict) < {cfg.features.near_miss_ttc_s} s or lead gap < {cfg.features.near_miss_gap_m} m",
        "splits": "calibration episodes A 40% (q_bar), B 40% (Platt fit), C 20% (diagnostics); NF fitted on A u B",
        "seed_range": "calibration (disjoint from train/validation/test)",
    }
    labels["semantic_raw_confidence"] = (
        "l_T*conf_T + l_M*q(T, match level) + l_V*q(T); q from split A" if cfg.semantic.raw_confidence_mode == "conditional_reliability"
        else "l_T*conf_T + l_M*match_fraction + l_V*q(T) (paper Eq. 22)")
    labels["speed_floor_in_rollouts"] = cfg.scenario.nominal_speed_floor_index
    labels["hard_shield_in_rollouts"] = cfg.semantic.rollout_hard_shield
    return CalibrationArtifacts(nf, asdict(nf_report), library, calibrator, iso, diagnostics, labels, counts, refinement)


# ===========================
# Pre-deployment Semantic-Library Refinement (follow-vs-not benefit, rule / LLM revision, then freeze)
# ===========================
# Pre-deployment refinement of the semantic-template library (report #54).
#
# Position in the pipeline (all BEFORE calibration, MARL training and evaluation):
#
#     initial library (paper Table 1)
#       -> K refinement rounds on the dedicated ``refinement`` environment-seed range
#            each round: randomised rollouts (fixed behaviour policy, deployment speed floor)
#                        -> follow-vs-not benefit per template (randomised comparison, cluster-robust CI)
#                        -> revision (rule-based by default, or LLM-proposed) -> next library version
#       -> one report-only verification round of the final library (fresh refinement seeds, no changes)
#       -> calibration of the FINAL library on the calibration range -> freeze
#       -> MARL training / validation / test with the frozen library
#
# Why a randomised comparison: at a matched step the calibration behaviour follows the recommendation
# with probability p_exec, independently of everything else, so the difference between the outcome after
# following and after not following estimates the effect of the template's recommendation in the
# contexts where the template fires.  A per-template *success rate* (the original script's criterion) is
# confounded by the riskiness of those contexts (report #51).
#
# Rule-based revision (deterministic):
#     benefit significantly negative (upper bound < 0), first time  -> swap the first two ranking groups
#     benefit significantly negative again after a revision         -> remove the template
#     otherwise                                                      -> keep
# Templates are never edited during training or evaluation; the library is frozen by the calibration stage.

# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _cluster_mean_se(values: Sequence[int], clusters: Sequence[Tuple[int, int]]) -> Tuple[float, float]:
    """Mean and cluster-robust (CR0) standard error; clusters = (episode, agent) because steps of one
    agent in one episode share the episode-level outcome."""
    y = np.asarray(values, dtype=np.float64)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan")
    m = float(y.mean())
    sums: Dict[Tuple[int, int], float] = defaultdict(float)
    for v, c in zip(y, clusters):
        sums[c] += v - m
    var = sum(s * s for s in sums.values()) / (n * n)
    return m, math.sqrt(var)


def estimate_effects(records: Sequence[SemanticRecord], z: float) -> Dict[str, Dict]:
    """Follow-vs-not benefit per template (and per match level, for reporting)."""
    by_t: Dict[str, List[SemanticRecord]] = defaultdict(list)
    for r in records:
        by_t[r.template_key].append(r)
    out = {}
    for key, rs in sorted(by_t.items()):
        f = [r for r in rs if r.executed]
        nf = [r for r in rs if not r.executed]
        pf, sf = _cluster_mean_se([r.outcome for r in f], [(r.episode, r.agent) for r in f])
        pn, sn = _cluster_mean_se([r.outcome for r in nf], [(r.episode, r.agent) for r in nf])
        both = bool(f and nf)  # the comparison is undefined (None, not NaN) when one arm is empty
        d = pf - pn if both else None
        se = math.sqrt(sf * sf + sn * sn) if both else None
        pf, pn = (pf if f else None), (pn if nf else None)
        levels = {}
        for lv in sorted({round(r.match_score, 4) for r in rs}):
            lf = [r.outcome for r in f if round(r.match_score, 4) == lv]
            ln = [r.outcome for r in nf if round(r.match_score, 4) == lv]
            levels[f"{lv:.4f}"] = {"n_follow": len(lf), "n_not": len(ln),
                                   "p_follow": float(np.mean(lf)) if lf else None, "p_not": float(np.mean(ln)) if ln else None}
        out[key] = {"n_follow": len(f), "n_not": len(nf), "p_follow": pf, "p_not": pn, "benefit": d, "se": se,
                    "ci_low": d - z * se if both else None, "ci_high": d + z * se if both else None,
                    "recommended_actions": dict(sorted(_count(r.semantic_action for r in rs).items())), "levels": levels}
    return out


def _count(it) -> Dict[str, int]:
    c: Dict[str, int] = defaultdict(int)
    for x in it:
        c[x] += 1
    return dict(c)


# ---------------------------------------------------------------------------
# revisions
# ---------------------------------------------------------------------------
def _swap_top_groups(t: DecisionTemplate, round_idx: int) -> DecisionTemplate:
    r = list(t.ranking)
    r[0], r[1] = r[1], r[0]
    return dataclasses.replace(t, ranking=tuple(r),
                               description=t.description + f" [refinement round {round_idx}: top group {list(t.ranking[0])} -> {list(r[0])}]")


def rule_decisions(library: SemanticLibrary, effects: Dict[str, Dict], revised: Dict[str, int], cfg: PnALConfig) -> List[Dict]:
    sm = cfg.semantic
    out = []
    for t in library.templates:
        e = effects.get(t.key)
        if e is None or min(e["n_follow"], e["n_not"]) < sm.refinement_min_samples:
            out.append({"key": t.key, "decision": "keep", "reason": "insufficient data"})
        elif e["ci_high"] < 0:
            if revised.get(t.key, 0) == 0:
                out.append({"key": t.key, "decision": "revise", "reason": "significantly harmful",
                            "ranking": [list(g) for g in _swap_top_groups(t, 0).ranking]})
            else:
                out.append({"key": t.key, "decision": "remove", "reason": "significantly harmful after revision"})
        elif e["ci_low"] > 0:
            out.append({"key": t.key, "decision": "keep", "reason": "significantly beneficial"})
        else:
            out.append({"key": t.key, "decision": "keep", "reason": "no significant effect"})
    return out


LLM_PROMPT = """You maintain a finite library of driving decision templates used as semantic priors.
Each template has symbolic conditions and a ranking over the five canonical actions
{actions}. The table below reports, for each template, a RANDOMISED comparison of the outcome
(no collision/off-road/near-miss in the next steps AND task completion) when its top recommendation was
followed versus not followed, with a 95% interval. Revise ONLY templates whose interval is entirely below
zero: either give a new ranking (each canonical action exactly once, groups as lists) or remove the template.
Do not change conditions. Reply with a JSON list only:
[{{"key": "...", "decision": "keep|revise|remove", "ranking": [["..."], ...]}}]

LIBRARY:
{library}

EFFECTS:
{effects}
"""


def llm_decisions(library: SemanticLibrary, effects: Dict[str, Dict], call: Callable[[str], str]) -> Tuple[List[Dict], str, str]:
    """Ask an LLM for revisions; the reply is validated (canonical actions, full coverage) before use."""
    eff = {k: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in e.items() if kk != "levels"} for k, e in effects.items()}
    prompt = LLM_PROMPT.format(actions=list(CANONICAL_ACTIONS), library=json.dumps(library.to_json()["templates"], indent=1),
                               effects=json.dumps(eff, indent=1))
    reply = call(prompt)
    start, end = reply.find("["), reply.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("LLM reply contains no JSON list")
    items = json.loads(reply[start:end + 1])
    keys = {t.key for t in library.templates}
    out = []
    for d in items:
        if d.get("key") not in keys or d.get("decision") not in ("keep", "revise", "remove"):
            raise ValueError(f"invalid LLM decision {d!r}")
        if d["decision"] == "revise":
            ranking = tuple(tuple(g) for g in d["ranking"])
            dataclasses.replace(next(t for t in library.templates if t.key == d["key"]), ranking=ranking).validate()
        out.append({"key": d["key"], "decision": d["decision"], "reason": "llm", **({"ranking": d["ranking"]} if d["decision"] == "revise" else {})})
    return out, prompt, reply


def apply_decisions(library: SemanticLibrary, decisions: Sequence[Dict], round_idx: int, revised: Dict[str, int]) -> int:
    changes = 0
    for d in decisions:
        if d["decision"] == "remove":
            library.remove_template(d["key"])
            changes += 1
        elif d["decision"] == "revise":
            t = next(t for t in library.templates if t.key == d["key"])
            new = dataclasses.replace(t, ranking=tuple(tuple(g) for g in d["ranking"]),
                                      description=t.description + f" [refinement round {round_idx}: ranking -> {d['ranking']}]")
            library.replace_template(t.key, new)
            revised[t.key] = revised.get(t.key, 0) + 1
            changes += 1
    return changes


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------
def refine_library(cfg: PnALConfig, env, extractor, llm_call: Optional[Callable[[str], str]] = None,
                   log: Callable[[str], None] = lambda *_: None) -> Tuple[SemanticLibrary, Dict]:
    """Returns the refined (NOT frozen) library and a JSON-serialisable report of every round."""
    sm = cfg.semantic
    library = SemanticLibrary.from_config(sm)
    initial_sha = library.sha256()
    checker = SafetyShield(ADARSOFFENStateRisk(len(NF_FEATURE_NAMES), cfg.neuro_fuzzy.n_rules), extractor.nf_features, cfg.shield,
                           1.0 / cfg.scenario.policy_frequency)
    rounds, revised = [], {}
    E = sm.refinement_episodes
    if sm.refinement_rounds == 0:
        return library, {"mode": "disabled", "rounds_configured": 0, "initial_library_sha256": initial_sha,
                         "final_library_sha256": initial_sha, "final_templates": [t.key for t in library.templates], "rounds": []}
    if sm.refinement_mode == "llm" and llm_call is None:
        raise RuntimeError("refinement_mode='llm' needs an llm_call (offline_llm.make_llm_call)")

    # Evidence for a template version accumulates over the decision rounds in which it was unchanged; the
    # K looks are Bonferroni-corrected (two-sided family-wise level of the configured z over K rounds).
    alpha = 2.0 * (1.0 - NormalDist().cdf(sm.refinement_z))
    z_eff = NormalDist().inv_cdf(1.0 - alpha / (2.0 * sm.refinement_rounds))
    pooled: Dict[Tuple[str, Tuple], List[SemanticRecord]] = defaultdict(list)

    def _version(key: str) -> Tuple:
        return tuple(next(t for t in library.templates if t.key == key).ranking)

    def _fmt(effects: Dict[str, Dict]) -> str:
        return ", ".join(f"{key} {e['benefit']:+.3f} [{e['ci_low']:+.3f},{e['ci_high']:+.3f}]" if e["benefit"] is not None
                         else f"{key} n/a (follow n={e['n_follow']}, not n={e['n_not']})" for key, e in effects.items())

    for k in range(sm.refinement_rounds):  # decision rounds on seed blocks 0..K-1
        entry = {"round": k + 1, "verification_only": False, "seed_episodes": [k * E, (k + 1) * E - 1],
                 "library_sha256_before": library.sha256(), "templates_before": [t.key for t in library.templates]}
        _, _, recs = collect_semantic_rollouts(cfg, env, extractor, library, checker, E, split="refinement", episode_offset=k * E)
        for r in recs:
            pooled[(r.template_key, _version(r.template_key))].append(r)
        current = [r for t in library.templates for r in pooled[(t.key, tuple(t.ranking))]]
        entry["effects_this_round"] = estimate_effects(recs, z_eff)
        entry["effects"] = effects = estimate_effects(current, z_eff)  # pooled since the current version
        entry["z_effective"] = z_eff
        if sm.refinement_mode == "rule":
            decisions = rule_decisions(library, effects, revised, cfg)
        else:
            decisions, prompt, reply = llm_decisions(library, effects, llm_call)
            entry.update({"llm_prompt": prompt, "llm_reply": reply})
        entry["decisions"] = decisions
        entry["changes"] = apply_decisions(library, decisions, k + 1, revised)
        entry["library_sha256_after"] = library.sha256()
        rounds.append(entry)
        log(f"refinement round {k + 1} (pooled, z={z_eff:.2f}): {_fmt(effects)} -> "
            + ", ".join(f"{d['key']}={d['decision']}" for d in decisions if d['decision'] != 'keep' or d['reason'] != 'insufficient data'))
    # verification of the FINAL library on a fixed, fresh seed block (block K); report only, nominal 95% interval
    if library.templates:
        _, _, recs = collect_semantic_rollouts(cfg, env, extractor, library, checker, E, split="refinement",
                                               episode_offset=sm.refinement_rounds * E)
        effects = estimate_effects(recs, sm.refinement_z)
    else:
        effects = {}
    rounds.append({"round": len(rounds) + 1, "verification_only": True,
                   "seed_episodes": [sm.refinement_rounds * E, (sm.refinement_rounds + 1) * E - 1],
                   "library_sha256_before": library.sha256(), "templates_before": [t.key for t in library.templates],
                   "effects": effects})
    log(f"refinement verification (final library, fresh seeds): {_fmt(effects) if effects else 'no templates left'}")
    report = {
        "mode": sm.refinement_mode, "rounds_configured": sm.refinement_rounds, "episodes_per_round": E,
        "decision_rule": ("swap the first two ranking groups if the pooled follow-vs-not benefit interval (Bonferroni over "
                          "the K decision rounds) is entirely < 0; remove the template if this happens again after a revision"),
        "seed_range": "refinement (disjoint from calibration/train/validation/test)",
        "initial_library_sha256": initial_sha, "final_library_sha256": library.sha256(),
        "final_templates": [t.key for t in library.templates],
        "final_rankings": {t.key: [list(g) for g in t.ranking] for t in library.templates},
        "rounds": rounds,
    }
    return library, report


# ===========================
# Reproducibility Utilities
# ===========================
# Seeding, determinism and experiment metadata.

def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def git_commit_hash(path: str = ".") -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_versions() -> Dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("highway-env", "gymnasium", "torch", "numpy", "scikit-learn"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            try:
                mod = __import__({"highway-env": "highway_env", "scikit-learn": "sklearn"}.get(pkg, pkg))
                ver = getattr(mod, "__version__", None)
                if ver is None:  # source checkout: record the git tag / commit of the package
                    src = os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))
                    try:
                        ver = "source:" + subprocess.check_output(
                            ["git", "-C", src, "describe", "--tags", "--always"], stderr=subprocess.DEVNULL
                        ).decode().strip()
                    except Exception:
                        ver = "unknown (source install at %s)" % src
                out[pkg] = ver
            except Exception:
                out[pkg] = "not installed"
    return out


def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


# ===========================
# Main Training System: calibrate -> train -> evaluate -> save / restore
# ===========================
# End-to-end PnAL system: calibration -> training (with optional meta-scheduler) -> evaluation.

SELECTION_EPISODE_OFFSET = 5000  # validation episodes reserved for model selection

#: Evaluation-time replacements of the learned proposal (Reviewer 3 comment 2: how much of the result does the
#: learned source contribute once the rule-based layers are engaged?).  "learned" is the normal pathway; the
#: untrained-initialisation reference is obtained with ``PnALSystem.restore(..., load_policy=False)``.
PROPOSAL_SOURCES = ("learned", "uniform_random", "cruise", "maintain")

#: Checkpoint-selection criteria computed on the validation split during training (Reviewer 2 comment 10).
#: "success" is the pre-declared PnAL rule (max success, ties -> lower collision -> later checkpoint).
SELECTION_CRITERIA = {
    "success": lambda v: (v["success"], -v["collision"]),
    "team_return": lambda v: (v["team_return"], v["success"]),
    "success_minus_collision": lambda v: (v["success"] - v["collision"], v["success"]),
}


class PnALSystem:
    def __init__(self, cfg: PnALConfig):
        cfg.validate()
        self.cfg = cfg
        seed_everything(cfg.seeds.seed)
        self.env = PnALEnv(cfg.scenario)
        self.action_map = self.env.action_map
        # runtime assertions on action semantics
        self.action_map.assert_valid(cfg.shield.fallback_ranking, "fallback ranking")
        self.action_map.assert_valid([cfg.shield.emergency_action], "emergency action")
        self.extractor = SafetyStateExtractor(self.env, cfg.features)
        # baselines (pnal/baselines.py) replace self.agents by their own joint learner; the small MLP placeholder
        # only keeps the base-class constructor unchanged
        backbone = "mlp" if (cfg.preset_name in IPPO_BACKBONE_PRESETS or cfg.preset_name in BASELINE_PRESETS) else "pathnet"
        self.agents = [PPOAgent(i, self.env.obs_dim, cfg.ppo, cfg.seeds.seed, backbone, cfg.run.device, action_set=cfg.scenario.policy_action_set)
                       for i in range(self.env.n_agents)]
        for a in self.agents:
            a.aux_enabled = cfg.flags.use_curriculum_aux
        self.coordinator = PriorityCoordinator(cfg.coordination)
        self.artifacts: Optional[CalibrationArtifacts] = None
        self.shield: Optional[SafetyShield] = None
        self.fusion: Optional[HierarchicalFusionController] = None
        self.torch_gen = torch.Generator().manual_seed(cfg.seeds.seed)
        self.train_episode_count = 0
        self.meta: Optional[SACMetaScheduler] = None
        self.meta_log: List[Dict] = []
        self._selected_state, self.selected_meta_step, self.selection_applied, self.selection_log = None, None, False, []
        self._criterion_states: Dict[str, Dict] = {}  # criterion -> {"meta_step", "state"} (in memory only)
        self.criterion_steps: Dict[str, int] = {}      # criterion -> chosen meta-step (kept after release)
        self._final_state: Optional[Dict] = None
        #: evaluation-time analysis switches (never active during training)
        self.proposal_source = "learned"
        self.obs_noise: Optional[tuple] = None  # (position std [m], velocity std [m/s]) on perceived neighbours
        self.window: deque = deque(maxlen=cfg.meta.marl_episodes_per_meta_step)
        self.compatibility = self.env.compatibility_report()
        self.phase = "setup"

    # ------------------------------------------------------------------
    def calibrate(self, n_episodes: Optional[int] = None, llm_call=None, log=lambda *_: None) -> CalibrationArtifacts:
        """Pre-deployment stage: library refinement (report #54) -> calibration -> freeze.

        Refinement runs for every preset (also those without the semantic source) so that all presets of
        one seed calibrate the neuro-fuzzy model on identical rollouts; it uses only the refinement seeds."""
        self.phase = "refinement"
        library, refinement = refine_library(self.cfg, self.env, self.extractor, llm_call=llm_call, log=log)
        self.phase = "calibration"
        art = run_calibration_stage(self.cfg, self.env, self.extractor, n_episodes, library=library, refinement=refinement)
        self._install(art)
        return art

    def _install(self, art: CalibrationArtifacts) -> None:
        cfg = self.cfg
        assert art.library.frozen, "semantic library must be frozen before training/evaluation"
        self.artifacts = art
        self.shield = SafetyShield(art.nf_model, self.extractor.nf_features, cfg.shield, 1.0 / cfg.scenario.policy_frequency)
        for t in art.library.templates:  # every template action exists in the env
            self.action_map.assert_valid([a for g in t.ranking for a in g], f"template {t.key}")
        sem = SemanticSource(art.library, cfg.semantic, art.calibrator) if cfg.flags.use_semantic else None
        self.fusion = HierarchicalFusionController(cfg.flags, cfg.semantic.tau_c, cfg.shield.risk_threshold_delta, sem, self.shield, self.coordinator,
                                                   speed_floor_index=cfg.scenario.nominal_speed_floor_index,
                                                   delta_temp_rule=cfg.semantic.delta_temp_rule, delta_temp_cap=cfg.semantic.delta_temp_cap)

    # ------------------------------------------------------------------
    def set_mode(self, train: bool) -> None:
        for a in self.agents:
            a.train(train)
            a.normalizer.frozen = not train
        if self.shield is not None:
            self.shield.nf.eval()  # the NF model is always frozen after calibration

    def run_episode(self, split: str, episode: int, train: bool, deterministic: bool) -> Dict:
        assert self.fusion is not None, "call calibrate() first"
        self.set_mode(train)
        env = self.env
        env_seed = self.cfg.seeds.env_seed(split, episode)
        obs, _ = env.reset(seed=env_seed)
        if train and (self.proposal_source != "learned" or self.obs_noise is not None):
            raise RuntimeError("proposal_source / obs_noise are evaluation-time analyses; reset them before training")
        if self.proposal_source not in PROPOSAL_SOURCES:
            raise ValueError(f"proposal_source must be one of {PROPOSAL_SOURCES}")
        noise_rng = np.random.default_rng([env_seed, 7919]) if self.obs_noise is not None else None
        prop_rng = np.random.default_rng([env_seed, 104729])
        obs = self._perceive(obs, noise_rng)
        stats = InterventionStatistics()
        team_return, agent_returns, steps = 0.0, np.zeros(env.n_agents), 0
        near_miss = False
        ep_traj = [[] for _ in range(env.n_agents)]
        if train:
            for i, a in enumerate(self.agents):
                a.normalizer.update(obs[i])  # each observation enters the running stats exactly once
        while True:
            norm = [a.normalizer.normalize(obs[i]) for i, a in enumerate(self.agents)]
            inputs = []
            for i, a in enumerate(self.agents):
                t0 = time.perf_counter()
                if self.proposal_source == "learned":
                    prop = a.propose(norm[i], deterministic, self.torch_gen)
                else:
                    prop = self._scripted_proposal(i, prop_rng)
                t0 = self.fusion._tick("learned_policy_forward", t0)
                x = self.extractor.extract(i, obs[i])
                ctx = self.extractor.symbolic_context(x)
                self.fusion._tick("safety_state_and_context_extraction", t0)
                near_miss |= x.near_miss
                pred = None
                if self.cfg.flags.use_safety_shield and self.cfg.shield.risk_mode == "action_conditioned":
                    dt, tau = 1.0 / self.cfg.scenario.policy_frequency, self.cfg.shield.speed_response_tau_s
                    pred = {a: self.extractor.predict_state(i, obs[i], a, dt, tau) for a in CANONICAL_ACTIONS}
                inputs.append(AgentInput(i, prop, x, ctx, env.vehicles[i], pred))
            recs = self.fusion.decide(inputs)
            stats.add(recs)
            res = env.step([r.executed_action for r in recs])
            steps += 1
            team_return += res.shared_reward
            agent_returns += np.array(res.agent_rewards)
            if train:
                agent_terminal = self.cfg.scenario.reward_arrival_once
                for i, a in enumerate(self.agents):
                    if agent_terminal and res.arrived_before[i]:
                        continue  # this agent's trajectory already ended at its arrival
                    arrived_now = agent_terminal and env.arrived[i]
                    a.normalizer.update(res.observations[i])
                    nxt = a.normalizer.normalize(res.observations[i])
                    a.store(Transition(
                        obs=norm[i],
                        proposal_idx=inputs[i].proposal.index,
                        proposal_log_prob=inputs[i].proposal.log_prob,  # behaviour log-prob of the SAME action
                        value=inputs[i].proposal.value,
                        reward=(res.agent_rewards[i] if self.cfg.ppo.reward_mode == "individual" else res.shared_reward),
                        terminated=res.terminated or arrived_now,  # agent-level terminal at arrival
                        truncated=res.truncated and not (res.terminated or arrived_now),
                        next_obs=nxt,
                        executed_idx=name_to_index(recs[i].executed_action),
                    ))
                    ep_traj[i].append((norm[i], name_to_index(recs[i].executed_action)))
                    if a.ready():
                        a.update()
            obs = self._perceive(res.observations, noise_rng)
            if res.terminated or res.truncated:
                break
        outcome = env.evaluate_task_outcome()
        if train:
            self.train_episode_count += 1
            if self.cfg.flags.use_curriculum_aux and outcome["success"]:
                for i, a in enumerate(self.agents):
                    for o, ex in ep_traj[i]:
                        a.add_curriculum(o, ex)
            if self.cfg.flags.adapt_safety_layer_online:
                self.shield.end_of_episode_update(
                    episode_count=self.train_episode_count,
                    collision=outcome["collision"], near_miss=near_miss, success=outcome["success"],
                )
        summ = stats.summary()
        self._last_episode_records = stats.records  # pooled by evaluate()
        return {
            "split": split, "episode": episode, "steps": steps, **{k: bool(v) for k, v in outcome.items()},
            "team_return": float(team_return), "mean_agent_return": float(agent_returns.mean()),
            "near_miss": bool(near_miss), "interventions": summ,
            "entropy": float(np.mean([np.mean(a.entropy_history) if a.entropy_history else 0.0 for a in self.agents])),
        }

    def _perceive(self, obs, rng):
        """Evaluation-time perception noise (Reviewer 2 comment 7): i.i.d. Gaussian noise on the relative
        position / velocity of every OBSERVED neighbour (rows 1.. with presence = 1) in the observation that both
        the learned policy and the safety-state extractor receive.  The ego row (odometry), the lane topology
        read from the map, and the coordination layer's vehicle states are left exact."""
        if self.obs_noise is None:
            return obs
        sp, sv = (float(v) for v in self.obs_noise)
        F = len(OBS_FEATURES)
        ix, iy, ivx, ivy, ipr = (OBS_FEATURES.index(k) for k in ("x", "y", "vx", "vy", "presence"))
        out = []
        for o in obs:
            m = np.array(o, dtype=np.float32).reshape(-1, F)
            present = m[1:, ipr] > 0.5
            n = int(present.sum())
            if n:
                rows = np.where(present)[0] + 1
                m[rows, ix] += rng.normal(0.0, sp, n)
                m[rows, iy] += rng.normal(0.0, sp, n)
                m[rows, ivx] += rng.normal(0.0, sv, n)
                m[rows, ivy] += rng.normal(0.0, sv, n)
            out.append(m.reshape(-1))
        return out

    def _scripted_proposal(self, i: int, rng) -> Proposal:
        """Non-learned proposal sources for the learned-source contribution analysis (Reviewer 3 comment 2).
        cruise: accelerate to the top target speed and keep it; maintain: always MAINTAIN (do-nothing
        reference that exposes degenerate testbeds); uniform_random: uniform over the policy's
        action set (seeded per episode)."""
        allowed = list(self.cfg.scenario.policy_action_set or CANONICAL_ACTIONS)
        if self.proposal_source == "maintain":  # do-nothing reference: keep lane and target speed
            name = "MAINTAIN"
        elif self.proposal_source == "cruise":
            v = self.env.vehicles[i]
            top = len(getattr(v, "target_speeds", [0])) - 1
            name = "ACCELERATE" if int(getattr(v, "speed_index", top)) < top else "MAINTAIN"
        else:
            name = allowed[int(rng.integers(0, len(allowed)))]
        return Proposal(name, name_to_index(name), 0.0, 0.0, 0.0, None)

    # ------------------------------------------------------------------
    def _window_metrics(self) -> Dict[str, float]:
        W = list(self.window)
        if not W:
            return {k: 0.0 for k in ("success_rate", "collision_rate", "offroad_rate", "timeout_rate", "mean_team_return",
                                     "mean_state_safety_score", "shield_intervention_rate", "semantic_admission_rate",
                                     "coordination_intervention_rate", "mean_policy_entropy")}
        g = lambda k: float(np.mean([w["interventions"].get(k, 0.0) for w in W]))
        return {
            "success_rate": float(np.mean([w["success"] for w in W])),
            "collision_rate": float(np.mean([w["collision"] for w in W])),
            "offroad_rate": float(np.mean([w["offroad"] for w in W])),
            "timeout_rate": float(np.mean([w["timeout"] for w in W])),
            "mean_team_return": float(np.mean([w["team_return"] for w in W])),
            "mean_state_safety_score": g("mean_state_score"),
            "shield_intervention_rate": g("shield_intervention_rate"),
            "semantic_admission_rate": g("semantic_admission_rate"),
            "coordination_intervention_rate": g("coordination_intervention_rate"),
            "mean_policy_entropy": float(np.mean([w["entropy"] for w in W])),
        }

    def _meta_state(self, progress: float) -> np.ndarray:
        m = self._window_metrics()
        v = [m["success_rate"], m["collision_rate"], m["offroad_rate"], m["timeout_rate"], m["mean_team_return"] / 100.0,
             m["mean_state_safety_score"], m["shield_intervention_rate"], m["semantic_admission_rate"],
             m["coordination_intervention_rate"], m["mean_policy_entropy"], progress]
        assert len(v) == len(META_STATE_FEATURES)
        return np.asarray(v, dtype=np.float32)

    def train(self, total_meta_steps: Optional[int] = None, episodes_per_step: Optional[int] = None, log=print,
              restore_selected: bool = True) -> Dict:
        self.phase = "training"
        cfg = self.cfg
        T = total_meta_steps or cfg.meta.total_meta_steps
        E = episodes_per_step or cfg.meta.marl_episodes_per_meta_step
        adapt = cfg.flags.use_training_adaptation
        space = build_meta_action_space(cfg.flags)
        default_hp = {"lr_actor_factor": 1.0, "lr_critic_factor": 1.0, "entropy_coef": cfg.ppo.entropy_coef,
                      "beta_aux": cfg.ppo.beta_aux, "curriculum_ratio": cfg.ppo.curriculum_ratio}
        if adapt:
            if T * cfg.meta.transitions_per_meta_step < cfg.meta.batch_size:
                raise ValueError("meta replay would never reach the batch size")
            self.meta = SACMetaScheduler(len(META_STATE_FEATURES), space, cfg.meta, cfg.seeds.seed)
        ep = 0
        learning_curve = []
        select = cfg.run.model_selection == "best_validation"
        best_key = None
        crit_best: Dict[str, tuple] = {}
        self._selected_state, self.selected_meta_step, self.selection_applied, self.selection_log = None, None, False, []
        self._criterion_states, self._final_state, self.criterion_steps = {}, None, {}
        for k in range(T):
            progress = (k + 1) / T
            s = self._meta_state(progress)
            if adapt:
                hp, y = self.meta.select(s, deterministic=progress > cfg.meta.deterministic_after_fraction)
            else:
                hp, y = {n: default_hp[n] for n, _ in space}, None
            for a in self.agents:
                a.set_hyperparameters(hp)  # every active meta dimension is consumed here
            before = self._window_metrics()
            for _ in range(E):
                r = self.run_episode("train", ep, train=True, deterministic=False)
                self.window.append(r)
                learning_curve.append({k2: r[k2] for k2 in ("episode", "success", "collision", "offroad", "timeout", "team_return")})
                ep += 1
            after = self._window_metrics()
            rec = {"meta_step": k, "phase": training_phase(progress), "hp": hp, **{f"after_{a}": b for a, b in after.items()}}
            if adapt:
                r_meta = meta_reward(before, after, training_phase(progress), cfg.meta.reward_scale)
                s2 = self._meta_state(min(1.0, progress + 1.0 / T))
                for _ in range(cfg.meta.transitions_per_meta_step):
                    self.meta.store(s, y, r_meta, s2, float(k == T - 1))
                upd = None
                for _ in range(cfg.meta.updates_per_meta_step):
                    upd = self.meta.update() or upd
                rec.update({"meta_reward": r_meta, "meta_replay_size": len(self.meta.replay),
                            "meta_update_count": self.meta.meta_update_count, **({} if upd is None else upd)})
            if select and ((k + 1) % cfg.run.selection_interval_meta_steps == 0 or k == T - 1):
                v = self.validate(cfg.run.selection_episodes)
                key = (v["success"], -v["collision"])
                better = best_key is None or key >= best_key  # ties -> the later checkpoint
                snap = None
                if better:
                    snap = self._policy_snapshot()
                    best_key, self._selected_state = key, snap
                    self.selected_meta_step = k
                # alternative criteria (Reviewer 2 comment 10): bookkeeping only, never used for the main result
                for name, fn in SELECTION_CRITERIA.items():
                    ck = fn(v)
                    if name not in crit_best or ck >= crit_best[name]:
                        crit_best[name] = ck
                        snap = snap or self._policy_snapshot()
                        self._criterion_states[name] = {"meta_step": k, "state": snap}
                        self.criterion_steps[name] = k
                self.selection_log.append({"meta_step": k, "episodes": ep, "val_success": v["success"],
                                           "val_collision": v["collision"], "val_timeout": v["timeout"],
                                           "val_team_return": v["team_return"], "selected": better})
                rec["val_success"], rec["val_collision"] = v["success"], v["collision"]
            self.meta_log.append(rec)
            log(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in rec.items() if not isinstance(v, dict)}))
        if adapt:
            assert self.meta.meta_update_count > 0, "SAC meta learner never performed a gradient update"
        self._final_state = self._policy_snapshot() if select else None
        if select and restore_selected:
            self.apply_selected_checkpoint()
        return {"learning_curve": learning_curve, "meta_log": self.meta_log, "model_selection": self.selection_summary()}

    # ------------------------------------------------------------------
    # Model selection on the held-out validation split (never the test split)
    def validate(self, n_episodes: int) -> Dict[str, float]:
        # validation episodes 5000+ are reserved for model selection so that other validation-split studies
        # (e.g. the tau_c sensitivity on episodes 0..n-1) do not reuse the episodes that picked the checkpoint
        off = SELECTION_EPISODE_OFFSET
        eps = [self.run_episode("validation", off + e, train=False, deterministic=True) for e in range(n_episodes)]
        return {k: float(np.mean([e[k] for e in eps])) for k in ("success", "collision", "offroad", "timeout", "team_return")}

    def _policy_snapshot(self) -> Dict:
        return {"agents": [{k: v.detach().to("cpu", copy=True) for k, v in a.state_dict().items()} for a in self.agents],
                "normalizers": [a.normalizer.state_dict() for a in self.agents]}

    def apply_selected_checkpoint(self) -> None:
        if self._selected_state is None:
            return
        for a, sd, nd in zip(self.agents, self._selected_state["agents"], self._selected_state["normalizers"]):
            a.load_state_dict(sd)
            a.normalizer.load_state_dict(nd)
        self.selection_applied = True

    def selection_summary(self) -> Dict:
        return {"mode": self.cfg.run.model_selection, "selected_meta_step": self.selected_meta_step,
                "applied": self.selection_applied, "log": self.selection_log,
                "criteria": dict(self.criterion_steps),
                "final_meta_step": (self.cfg.meta.total_meta_steps - 1) if self.criterion_steps else None}

    def checkpoint_variants(self) -> Dict[str, Dict]:
        """In-memory policy snapshots: the final policy and the checkpoint of every selection criterion."""
        out = {n: dict(d) for n, d in self._criterion_states.items()}
        if self._final_state is not None:
            out["final"] = {"meta_step": self.cfg.meta.total_meta_steps - 1, "state": self._final_state}
        return out

    def load_policy_state(self, state: Dict) -> None:
        for a, sd, nd in zip(self.agents, state["agents"], state["normalizers"]):
            a.load_state_dict(sd)
            a.normalizer.load_state_dict(nd)

    def release_checkpoint_variants(self) -> None:
        self._criterion_states, self._final_state = {}, None

    # ------------------------------------------------------------------
    def evaluate(self, n_episodes: Optional[int] = None, split: str = "test") -> Dict:
        self.phase = "evaluation"
        assert self.artifacts.library.frozen
        if self.artifacts.calibrator is not None:
            assert self.artifacts.calibrator._frozen
        n = n_episodes or self.cfg.run.evaluation_episodes
        self.fusion.reset_timing()
        allrec = InterventionStatistics()
        eps = []
        for e in range(n):
            eps.append(self.run_episode(split, e, train=False, deterministic=True))
            allrec.add(self._last_episode_records)
        agg = {k: float(np.mean([e[k] for e in eps])) for k in ("success", "collision", "offroad", "timeout", "team_return", "mean_agent_return", "steps")}
        # outcome counts must partition the episodes
        assert all(sum(e[k] for k in ("success", "collision", "offroad", "timeout", "incomplete")) == 1 for e in eps)
        # intervention rates are pooled over ALL agent-steps of the split (not an average of
        # per-episode rates, which is wrong for conditional rates such as admission-given-match)
        agg["interventions"] = allrec.summary()
        # Reviewer 1: per-stage inference latency (ms per call; CPU wall clock of THIS machine, so it
        # is kept outside the deterministic 'aggregate' block) and model sizes.
        compute = {
            "latency_ms_per_call": self.fusion.latency_summary_ms(),
            "parameter_counts": {
                "learned_policy_per_agent": int(sum(p.numel() for p in self.agents[0].parameters())),
                "neuro_fuzzy": int(sum(p.numel() for p in self.shield.nf.parameters())),
                "semantic_templates": len(self.artifacts.library.templates),
            },
        }
        return {"aggregate": agg, "compute": compute, "episodes": eps}

    # ------------------------------------------------------------------
    def metadata(self) -> Dict:
        art = self.artifacts
        return {
            "seed": self.cfg.seeds.seed,
            "preset": self.cfg.preset_name,
            "flags": asdict(self.cfg.flags),
            "config": self.cfg.to_dict(),
            "environment_versions": environment_versions(),
            "environment_config": build_env_config(self.cfg.scenario),
            "action_map": self.action_map.metadata(),
            "semantic_library_sha256": art.library.sha256() if art else None,
            "semantic_library_frozen": art.library.frozen if art else None,
            "calibrator": art.calibrator.params() if (art and art.calibrator) else None,
            "calibration_labels": art.label_definitions if art else None,
            "calibration_counts": art.counts if art else None,
            "calibration_diagnostics": art.diagnostics if art else None,
            "library_refinement": ({k: v for k, v in art.refinement.items() if k != "rounds"} | {
                "rounds": [{k: v for k, v in r.items() if k not in ("effects", "llm_prompt", "llm_reply")} for r in art.refinement["rounds"]]}
                if (art and art.refinement) else None),
            "nf_features": list(NF_FEATURE_NAMES),
            "coordination_weights": {k: getattr(self.cfg.coordination, k) for k in ("w_r", "w_u", "w_g", "w_m")},
            "meta_action_space": build_meta_action_space(self.cfg.flags),
            "meta_update_count": self.meta.meta_update_count if self.meta else 0,
            "model_selection": self.selection_summary(),
            "compatibility_report": self.compatibility,
            "git_commit": git_commit_hash(os.path.dirname(os.path.abspath(globals().get("__file__", ".")))),
        }

    @classmethod
    def restore(cls, cfg: PnALConfig, out_dir: str, map_location: str = "cpu", load_policy: bool = True,
                artifacts_override=None) -> "PnALSystem":
        """Rebuild a trained, frozen system from ``save()`` output (no recalibration, no training).

        ``load_policy=False`` keeps the seeded INITIAL (untrained) policy of this run (Reviewer 3 comment 2).
        ``artifacts_override(art) -> art`` may replace frozen pre-deployment artefacts for an evaluation-time
        stress test (e.g. a corrupted semantic library); the returned artefacts must be frozen.

        The frozen artefacts (NF model, semantic library incl. q_bar, calibrators) and the policies are
        loaded from disk; the library SHA-256 and preset are checked against the saved metadata."""
        with open(os.path.join(out_dir, "metadata.json")) as f:
            meta = json.load(f)
        if meta["preset"] != cfg.preset_name or meta["seed"] != cfg.seeds.seed:
            raise ValueError("config does not match the saved run (preset/seed)")
        sysm = cls(cfg)
        nf = ADARSOFFENStateRisk(len(NF_FEATURE_NAMES), cfg.neuro_fuzzy.n_rules)
        nf.load_state_dict(torch.load(os.path.join(out_dir, "neuro_fuzzy.pt"), map_location="cpu", weights_only=True))
        nf.freeze()
        lib = SemanticLibrary.load(os.path.join(out_dir, "semantic_library.json"))
        lib.freeze()
        if lib.sha256() != meta["semantic_library_sha256"]:
            raise ValueError("semantic library on disk does not match the metadata hash")
        cal = iso = None
        if os.path.exists(os.path.join(out_dir, "calibrator_platt.json")):
            cal = SemanticConfidenceCalibrator.load(os.path.join(out_dir, "calibrator_platt.json"))
            iso = SemanticConfidenceCalibrator.load(os.path.join(out_dir, "calibrator_isotonic.json"))
        art = CalibrationArtifacts(nf, meta.get("calibration_diagnostics", {}).get("neuro_fuzzy_fit", {}), lib, cal, iso,
                                   meta.get("calibration_diagnostics") or {}, meta.get("calibration_labels") or {},
                                   meta.get("calibration_counts") or {}, meta.get("library_refinement"))
        if artifacts_override is not None:
            art = artifacts_override(art)
        sysm._install(art)
        if load_policy:
            ck = torch.load(os.path.join(out_dir, "pnal_policy.pt"), map_location=map_location, weights_only=True)
            sysm.load_policy_state(ck)
        sysm.phase = "restored" if load_policy else "restored_untrained_policy"
        return sysm

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"agents": [a.state_dict() for a in self.agents],
                    "normalizers": [a.normalizer.state_dict() for a in self.agents]}, os.path.join(out_dir, "pnal_policy.pt"))
        torch.save(self.shield.nf.state_dict(), os.path.join(out_dir, "neuro_fuzzy.pt"))
        self.artifacts.library.save(os.path.join(out_dir, "semantic_library.json"))
        if self.artifacts.refinement is not None:
            write_json(self.artifacts.refinement, os.path.join(out_dir, "library_refinement.json"))
        if self.artifacts.calibrator:
            self.artifacts.calibrator.save(os.path.join(out_dir, "calibrator_platt.json"))
            self.artifacts.isotonic_calibrator.save(os.path.join(out_dir, "calibrator_isotonic.json"))
        if self.meta:
            torch.save(self.meta.state_dict(), os.path.join(out_dir, "meta_scheduler.pt"))
        write_json(self.metadata(), os.path.join(out_dir, "metadata.json"))


# ---------------------------------------------------------------------------
# Pre-deployment artefact cache (efficiency only; results are unchanged)
# ---------------------------------------------------------------------------
#: configuration blocks that never enter refinement / calibration (the preset only decides which artefacts are USED)
_NOT_IN_PRE_DEPLOYMENT = ("flags", "preset_name", "ppo", "meta")


def calibration_cache_key(cfg: PnALConfig) -> str:
    """Hash of everything the refinement + calibration stage depends on.  Presets of one (testbed, N, seed) share it:
    the stage uses its own seeded RNGs and environment-seed range, never the preset flags (except for fitting the
    semantic calibrator, which the cache always contains) and never delta_t."""
    import hashlib

    d = cfg.to_dict()
    for k in _NOT_IN_PRE_DEPLOYMENT:
        d.pop(k, None)
    d["run"] = {"calibration_episodes": d["run"]["calibration_episodes"]}
    d["shield"].pop("risk_threshold_delta", None)
    for k in ("delta_temp_rule", "delta_temp_cap"):  # online fusion only, never used by the pre-deployment stage
        d["semantic"].pop(k, None)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _save_artifacts(art: CalibrationArtifacts, d: str) -> None:
    import shutil
    import uuid

    tmp = f"{d}.tmp_{uuid.uuid4().hex[:8]}"
    os.makedirs(tmp)
    torch.save(art.nf_model.state_dict(), os.path.join(tmp, "neuro_fuzzy.pt"))
    art.library.save(os.path.join(tmp, "semantic_library.json"))
    if art.calibrator is not None:
        art.calibrator.save(os.path.join(tmp, "calibrator_platt.json"))
        art.isotonic_calibrator.save(os.path.join(tmp, "calibrator_isotonic.json"))
    write_json({"nf_report": art.nf_report, "diagnostics": art.diagnostics, "labels": art.label_definitions,
                "counts": art.counts, "refinement": art.refinement, "library_sha256": art.library.sha256()},
               os.path.join(tmp, "artifacts.json"))  # written last: completeness marker
    try:
        os.replace(tmp, d)
    except OSError:  # another session stored the same artefacts meanwhile
        shutil.rmtree(tmp, ignore_errors=True)


def _load_artifacts(d: str, cfg: PnALConfig) -> CalibrationArtifacts:
    with open(os.path.join(d, "artifacts.json")) as f:
        meta = json.load(f)
    nf = ADARSOFFENStateRisk(len(NF_FEATURE_NAMES), cfg.neuro_fuzzy.n_rules)
    nf.load_state_dict(torch.load(os.path.join(d, "neuro_fuzzy.pt"), map_location="cpu", weights_only=True))
    nf.freeze()
    lib = SemanticLibrary.load(os.path.join(d, "semantic_library.json"))
    lib.freeze()
    if lib.sha256() != meta["library_sha256"]:
        raise ValueError(f"cached semantic library in {d} does not match its hash")
    cal = iso = None
    if os.path.exists(os.path.join(d, "calibrator_platt.json")):
        cal = SemanticConfidenceCalibrator.load(os.path.join(d, "calibrator_platt.json"))
        iso = SemanticConfidenceCalibrator.load(os.path.join(d, "calibrator_isotonic.json"))
    return CalibrationArtifacts(nf, meta["nf_report"], lib, cal, iso, meta["diagnostics"], meta["labels"], meta["counts"],
                                meta["refinement"])


def _for_preset(art: CalibrationArtifacts, flags) -> CalibrationArtifacts:
    """Artefacts exactly as ``calibrate()`` returns them for this preset: without the semantic source, no
    semantic calibrator is fitted and no semantic diagnostics are reported."""
    if flags.use_semantic:
        return art
    import dataclasses

    diag = {k: v for k, v in art.diagnostics.items() if not k.startswith("semantic_C_")}
    return dataclasses.replace(art, calibrator=None, isotonic_calibrator=None, diagnostics=diag)


def calibrate_with_cache(sysm: "PnALSystem", cache_root: Optional[str], llm_call=None, log=lambda *_: None):
    """``sysm.calibrate()`` with a cache shared by every preset of one (testbed, N, seed).  Returns (artefacts, hit).
    The stage consumes no global RNG (tests/test_calibration_cache.py), so a cache hit is exactly equivalent."""
    if cache_root is None or llm_call is not None:  # LLM refinement is not cached
        return sysm.calibrate(llm_call=llm_call, log=log), False
    d = os.path.join(cache_root, calibration_cache_key(sysm.cfg))
    flags = sysm.cfg.flags
    os.makedirs(cache_root, exist_ok=True)
    lock = d + ".lock"
    # parallel workers of one session: the first one computes, the others wait for its artefacts (a lock older
    # than 3 h is considered stale; worst case two workers compute the same, identical artefacts)
    owns_lock = False
    while not os.path.exists(os.path.join(d, "artifacts.json")):
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            owns_lock = True
            break
        except FileExistsError:  # another worker is computing: wait for its artefacts (stale lock after 3 h)
            try:
                if time.time() - os.path.getmtime(lock) > 3 * 3600:
                    break
            except FileNotFoundError:
                continue
            time.sleep(20)
    try:
        return _calibrate_with_cache_inner(sysm, d, flags, log)
    finally:
        if owns_lock and os.path.exists(lock):
            os.remove(lock)


def _calibrate_with_cache_inner(sysm, d, flags, log):
    import dataclasses

    if os.path.exists(os.path.join(d, "artifacts.json")):
        art = _for_preset(_load_artifacts(d, sysm.cfg), flags)
        sysm._install(art)
        return art, True
    if not flags.use_semantic:  # the cached copy must also serve the presets that use the semantic source
        sysm.cfg.flags = dataclasses.replace(flags, use_semantic=True, use_calibration=True, always_trust_semantic=False)
    try:
        art = sysm.calibrate(log=log)
    except CalibrationDataError:
        if flags.use_semantic:
            raise
        sysm.cfg.flags = flags
        return sysm.calibrate(log=log), False  # this testbed cannot calibrate the semantic source: no caching
    finally:
        sysm.cfg.flags = flags
    _save_artifacts(art, d)
    art = _for_preset(art, flags)
    sysm._install(art)
    return art, False


# ===========================
# Statistical Analysis (seed-level CI, significance, AULC, paired episode effects)
# ===========================
# Seed-level statistics requested by the reviewers (R1: significance, AULC, multiple thresholds).
#
# All functions consume per-seed numbers produced by scripts/run_experiment.py; they never
# generate or modify results.

def mean_std_ci(x: Sequence[float], conf: float = 0.95) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    m, s = float(x.mean()), float(x.std(ddof=1)) if n > 1 else 0.0
    h = float(stats.t.ppf(0.5 + conf / 2, n - 1) * s / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": m, "std": s, "ci_low": m - h, "ci_high": m + h}


def compare_methods(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Welch t-test, Mann-Whitney U and a bootstrap 95% CI of mean(a) - mean(b) over seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    return {
        "diff_mean": float(a.mean() - b.mean()),
        "bootstrap_ci_low": float(np.percentile(diffs, 2.5)),
        "bootstrap_ci_high": float(np.percentile(diffs, 97.5)),
        "welch_t_p": float(stats.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 1 and len(b) > 1 else float("nan"),
        "mann_whitney_p": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
    }


def area_under_learning_curve(success_per_episode: Sequence[float], window: int = 50) -> float:
    y = np.convolve(np.asarray(success_per_episode, float), np.ones(window) / window, mode="valid")
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 / <2
    return float(_trapz(y, dx=1.0) / max(1, len(y) - 1)) if len(y) > 1 else float(y.mean()) if len(y) else float("nan")


def episodes_to_thresholds(success_per_episode: Sequence[float], thresholds=(0.3, 0.5, 0.65, 0.8), window: int = 50) -> Dict[str, float]:
    y = np.convolve(np.asarray(success_per_episode, float), np.ones(window) / window, mode="valid")
    out = {}
    for t in thresholds:
        idx = np.nonzero(y >= t)[0]
        out[f"episodes_to_{t:.2f}"] = float(idx[0] + window) if len(idx) else float("nan")
    return out


def paired_episode_effect(ref: Sequence[Sequence[int]], cond: Sequence[Sequence[int]], n_boot: int = 5000,
                          seed: int = 0) -> Dict[str, float]:
    """Effect of an evaluation-time condition on the SAME trained policies and the SAME test episodes.

    ref[k][e], cond[k][e]: 0/1 outcome (e.g. success) of common test episode e for training seed k under the
    reference pathway and under the condition.  Effect = mean(cond - ref) over all (seed, episode) pairs.
    95% CI: bootstrap resampling episodes within every seed (the trained policies are held fixed, so the
    uncertainty is over the episode distribution).  p: exact McNemar test on the discordant pairs."""
    r = [np.asarray(x, dtype=float) for x in ref]
    c = [np.asarray(x, dtype=float) for x in cond]
    if len(r) != len(c) or any(len(a) != len(b) for a, b in zip(r, c)) or not r:
        raise ValueError("paired comparison needs the same seeds and the same episodes")
    d = [b - a for a, b in zip(r, c)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(np.concatenate([x[rng.integers(0, len(x), len(x))] for x in d]).mean())
    allr, allc = np.concatenate(r), np.concatenate(c)
    worse, better = int(np.sum((allr == 1) & (allc == 0))), int(np.sum((allr == 0) & (allc == 1)))
    p = float(stats.binomtest(better, better + worse, 0.5).pvalue) if better + worse > 0 else 1.0
    return {"diff_mean": float(np.concatenate(d).mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "mcnemar_p": p, "n_pairs": int(len(allr)),
            "discordant_better": better, "discordant_worse": worse}


# ===========================
# Records, evaluation-time analyses and tables (from the runner of the single file)
# ===========================
def _show(df):
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string())


def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _record(text, event=None):
    with open(os.path.join(RESULTS_DIR, RECORD_MD), "a", encoding="utf-8") as f:
        f.write(text + "\n")
    if event is not None:
        with open(os.path.join(RESULTS_DIR, RECORD_JSONL), "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": _now(), **event}, default=str) + "\n")


def _md_table(df):
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    cols = [" / ".join(str(p) for p in c if str(p)) if isinstance(c, tuple) else str(c) for c in df.columns]
    fmt = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols), *rows])


def _report(title, df, name):
    """Show a table, save it as CSV (+ LaTeX for the paper) and append it to the cumulative record."""
    print("\n" + title)
    _show(df)
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    base = os.path.join(RESULTS_DIR, "tables", f"{name}_{SCENARIO}_N{N_AGENTS}")
    df.to_csv(base + ".csv")
    try:
        with open(base + ".tex", "w", encoding="utf-8") as f:
            f.write(df.to_latex(escape=True, caption=title, label=f"tab:{name}"))
    except Exception as e:  # LaTeX export needs jinja2; the CSV is the primary record
        print(f"  (LaTeX export skipped: {type(e).__name__})")
    _record(f"\n#### {title}\n\n" + _md_table(df) + "\n", {"event": "table", "name": name, "title": title})


class _Tee:
    """Copies everything written to the console into the console log (append)."""
    _pnal_tee = True

    def __init__(self, stream, path):
        self.stream, self.path, self.f = stream, path, open(path, "a", encoding="utf-8", buffering=1)

    def retarget(self, path):
        if path != self.path:
            self.f.close()
            self.path, self.f = path, open(path, "a", encoding="utf-8", buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def __getattr__(self, k):
        return getattr(self.stream, k)


def start_console_log():
    path = os.path.join(RESULTS_DIR, CONSOLE_LOG)
    for name in ("stdout", "stderr"):
        cur = getattr(sys, name)
        if getattr(cur, "_pnal_tee", False):  # re-running the cell: do not wrap twice
            cur.retarget(path)
        else:
            setattr(sys, name, _Tee(cur, path))


def config_fingerprint(cfg_dict):
    """Short hash of the full configuration, ignoring the device (results must not depend on it)."""
    d = json.loads(json.dumps(cfg_dict, default=str))
    d.get("run", {}).pop("device", None)
    d.get("run", {}).pop("output_dir", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _config_diff(a, b, prefix=""):
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if isinstance(va, dict) and isinstance(vb, dict):
            out += _config_diff(va, vb, f"{prefix}{k}.")
        elif va != vb and f"{prefix}{k}" not in ("run.device", "run.output_dir"):
            out.append(f"{prefix}{k}: saved={va!r} current={vb!r}")
    return out


#: configuration fields added to the code after earlier runs were made, with the value that reproduces the earlier
#: behaviour: a saved configuration WITHOUT such a field is equivalent iff the current value equals this one
LATER_ADDED_FIELDS = {"scenario.background_vehicle_collisions": True, "scenario.background_collisions": True,
                      "semantic.delta_temp_rule": "paper_max", "semantic.delta_temp_cap": 1.0}


def _config_equivalent(saved, current):
    """(equivalent?, differences): equal up to fields added later whose current value keeps the earlier behaviour."""
    cur = json.loads(json.dumps(current, default=str))
    diffs = []
    for line in _config_diff(saved or {}, cur):
        key = line.split(":", 1)[0]
        if "saved='<missing>'" in line and key in LATER_ADDED_FIELDS:
            node = cur
            for part in key.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            if node == LATER_ADDED_FIELDS[key]:
                continue
        diffs.append(line)
    return (saved is not None and not diffs), diffs


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _run_row(status, scenario, n_agents, tag, preset, seed, profile, fp, out, minutes=None):
    ev, meta = _load_json(os.path.join(out, "evaluation.json")), _load_json(os.path.join(out, "metadata.json")) or {}
    cv = _load_json(os.path.join(out, "checkpoint_variants.json")) or {}
    fin = _variant_aggregate(cv, "final", ev) or {}
    a, iv = ev["aggregate"], ev["aggregate"].get("interventions", {})
    cal = meta.get("calibrator") or {}
    sel = meta.get("model_selection") or {}
    rf = meta.get("library_refinement") or {}
    changes = sum(int(r.get("changes", 0) or 0) for r in rf.get("rounds", []))
    nf = ((meta.get("calibration_diagnostics") or {}).get("neuro_fuzzy_fit") or {}).get("val_auroc")
    return {"recorded_at": _now(), "status": status, "scenario": scenario, "n_agents": n_agents, "tag": tag, "preset": preset,
            "seed": seed, "profile": profile, "config_fingerprint": fp,
            **{m: a.get(m) for m in ("success", "collision", "offroad", "timeout", "team_return")},
            "test_episodes": len(ev.get("episodes", [])),
            **{f"final_{m}": fin.get(m) for m in ("success", "collision", "timeout")},
            "semantic_admission_rate": iv.get("semantic_admission_rate"), "shield_intervention_rate": iv.get("shield_intervention_rate"),
            "coordination_intervention_rate": iv.get("coordination_intervention_rate"),
            "selected_meta_step": (sel.get("selected_meta_step") + 1) if sel.get("selected_meta_step") is not None else None,
            "nf_val_auroc": nf, "platt_A": cal.get("A"), "platt_unconstrained_A": cal.get("unconstrained_A"),
            "refinement_changes": changes if rf else None, "minutes": minutes, "run_dir": os.path.abspath(out)}


def _register_run(row):
    """Append to the registry unless this (run folder, configuration) is already registered."""
    import csv
    path = os.path.join(RESULTS_DIR, RUNS_CSV)
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            if any(r.get("run_dir") == row["run_dir"] and r.get("config_fingerprint") == row["config_fingerprint"] for r in csv.DictReader(f)):
                return False
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in RUNS_CSV_FIELDS})
    return True


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _num(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _record_run(row):
    _record(f"\n### {row['scenario']} N={row['n_agents']} | {row['tag']} | seed {row['seed']} — {row['status']} ({row['recorded_at']})\n"
            f"- profile `{row['profile']}`, config fingerprint `{row['config_fingerprint']}`, folder `{row['run_dir']}`\n"
            f"- TEST ({row['test_episodes']} episodes), selected checkpoint: success {_pct(row['success'])}, collision {_pct(row['collision'])}, "
            f"off-road {_pct(row['offroad'])}, timeout {_pct(row['timeout'])}, team return {row['team_return']:.3f}\n"
            f"- TEST, final checkpoint: success {_pct(row['final_success'])}, collision {_pct(row['final_collision'])}, "
            f"timeout {_pct(row['final_timeout'])}\n"
            f"- interventions: semantic admission {_pct(row['semantic_admission_rate'])}, shield {_pct(row['shield_intervention_rate'])}, "
            f"coordination {_pct(row['coordination_intervention_rate'])}\n"
            f"- selected checkpoint: meta-step {_num(row['selected_meta_step'])} | NF val AUROC {_num(row['nf_val_auroc'])} | "
            f"Platt A {_num(row['platt_A'], 2)} (unconstrained {_num(row['platt_unconstrained_A'], 2)}) | "
            f"refinement changes {_num(row['refinement_changes'])}"
            + (f" | {row['minutes']:.1f} min" if row["minutes"] is not None else ""),
            {"event": "run", **row})


SESSION_SETTINGS = ["SHARD", "SEEDS", "SETTINGS", "SOURCE_OVERRIDES", "SUPPLEMENT_PLAN", "TAU_GRID",
                    "ANALYSIS_EPISODES", "SWEEP_EPISODES", "REPRO_EPISODES", "PROFILE", "RISK_THRESHOLD_DELTA",
                    "ROUNDABOUT_BACKGROUND_VEHICLES", "ROUNDABOUT_DENSITY_LEVELS", "HIGHWAY_FAST_SIMULATION",
                    "INTERSECTION_DESTINATIONS", "INTERSECTION_CONTROL_MODE", "REWARD_MODE", "COMMON_EVAL_SET",
                    "MODEL_SELECTION", "SEMANTIC_REFINEMENT_ROUNDS", "RAW_CONFIDENCE_MODE", "DEVICE"]


def record_session_start():
    md = os.path.join(RESULTS_DIR, RECORD_MD)
    if not os.path.exists(md):
        _record("# PnAL experiment record" + (f" (shard {SHARD})" if SHARD else "") + "\n\nAppend-only. Every session (also every "
                "resume) adds its settings, one block per finished or skipped run, and the summary tables recomputed over all "
                f"finished runs in this folder. The registry of runs is `{RUNS_CSV}`; the full console output is `{CONSOLE_LOG}`.")
    g = globals()  # settings defined later (e.g. in a later notebook cell) are recorded as None
    settings = {n.lower(): g.get(n) for n in SESSION_SETTINGS}
    settings["budget"] = BUDGETS.get(PROFILE)
    settings["versions"] = environment_versions()
    _record(f"\n---\n\n## Session started {_now()}\n\n| setting | value |\n|---|---|\n"
            + "\n".join(f"| {k} | `{v}` |" for k, v in settings.items()), {"event": "session_start", "settings": settings})


def _traffic_suffix(scenario):
    """Roundabout with a non-default background count gets its own delta_t selection (the selection depends on
    the traffic); the default traffic keeps the original file name."""
    bg = globals().get("ROUNDABOUT_BACKGROUND_VEHICLES")
    if scenario == "highway" and globals().get("HIGHWAY_FAST", False):
        return "_fast"
    if scenario == "highway" and globals().get("HIGHWAY_FAST_SIMULATION", False):
        return "_fast"
    return f"_bg{int(bg)}" if scenario == "roundabout" and bg is not None else ""


def build_config(scenario, n_agents, preset, seed, profile, overrides=None, allow_auto_delta=False):
    kw = {}
    if scenario == "intersection":
        kw["legacy_target_speeds"] = INTERSECTION_LEGACY_SPEEDS
        if n_agents >= 5:
            kw["second_row_offset_m"] = INTERSECTION_SECOND_ROW_OFFSET_M
    cfg = make_config(scenario, n_agents, preset, seed=seed, **kw)
    if scenario == "intersection":
        # V0 = "fixed_o1" (first submission: every vehicle exits at o1), V1 = "random_turn" (report #56/#57)
        configure_intersection_traffic(cfg.scenario, destinations=INTERSECTION_DESTINATIONS)
    if scenario == "intersection" and not INTERSECTION_LEGACY_SPEEDS and INTERSECTION_TARGET_SPEEDS is not None:
        cfg.scenario.target_speeds = tuple(float(v) for v in INTERSECTION_TARGET_SPEEDS)
        cfg.coordination.reference_speed_mps = float(max(cfg.scenario.target_speeds))
    if scenario == "intersection" and INTERSECTION_CONTROL_MODE == "cautious_to_full":
        if INTERSECTION_ACTION_SPACE == "paper":
            cfg.scenario.policy_action_set = None  # all five actions of paper Sec. 5.1 / Table 2
        elif INTERSECTION_ACTION_SPACE == "longitudinal":
            cfg.scenario.policy_action_set = ("MAINTAIN", "ACCELERATE", "DECELERATE")
        else:
            raise ValueError("INTERSECTION_ACTION_SPACE must be 'paper' or 'longitudinal'")
        cfg.scenario.nominal_speed_floor_index = 1
        cfg.scenario.duration_s = 20.0
    elif scenario == "intersection" and INTERSECTION_CONTROL_MODE != "paper_default":
        raise ValueError("INTERSECTION_CONTROL_MODE must be 'cautious_to_full' or 'paper_default'")
    if scenario == "intersection" and INTERSECTION_DURATION_S is not None:
        cfg.scenario.duration_s = float(INTERSECTION_DURATION_S)
    if scenario == "roundabout" and ROUNDABOUT_SPEED_FLOOR_INDEX is not None:
        cfg.scenario.nominal_speed_floor_index = int(ROUNDABOUT_SPEED_FLOOR_INDEX)
    if scenario == "roundabout":
        cfg.scenario.extra_env_config["normalize_reward"] = bool(ROUNDABOUT_NORMALIZE_REWARD)
    if scenario == "roundabout" and ROUNDABOUT_BACKGROUND_VEHICLES is not None:
        # N controlled + K background vehicles on the ring (paper Table 2: 10 in total, i.e. K = 10 - N)
        cfg.scenario.total_vehicles = int(n_agents) + int(ROUNDABOUT_BACKGROUND_VEHICLES)
        cfg.scenario.enable_background = int(ROUNDABOUT_BACKGROUND_VEHICLES) > 0
    if scenario == "highway" and globals().get("HIGHWAY_FAST", False):
        # highway-env "highway-fast-v0" execution: 5 Hz simulation, no collision checks among background vehicles
        # (collisions of controlled vehicles are still detected); lanes / vehicles / duration stay as in Table 2
        cfg.scenario.simulation_frequency = 5
        cfg.scenario.background_collisions = False
    if scenario == "highway" and globals().get("HIGHWAY_FAST_SIMULATION", False):
        # highway-fast-v0 execution: 5 Hz simulation, no collision checks AMONG background vehicles (report #67)
        cfg.scenario.simulation_frequency = 5
        cfg.scenario.background_vehicle_collisions = False
    if cfg.flags.use_semantic:  # the rule only exists for presets with the semantic source (report #70)
        cfg.semantic.delta_temp_rule = globals().get("DELTA_TEMP_RULE", "paper_max")
        cfg.semantic.delta_temp_cap = float(globals().get("DELTA_TEMP_CAP", 1.0))
    delta = _risk_delta_for(scenario, n_agents, allow_auto=allow_auto_delta)
    if delta is not None:
        cfg.shield.risk_threshold_delta = delta
    cfg.scenario.reward_arrival_once = bool(REWARD_ARRIVAL_ONCE)
    cfg.ppo.reward_mode = REWARD_MODE
    b = BUDGETS[profile]
    cfg.meta.total_meta_steps = b["meta_steps"]
    cfg.meta.marl_episodes_per_meta_step = b["episodes_per_meta_step"]
    cfg.run.calibration_episodes = b["calibration_episodes"]
    cfg.run.evaluation_episodes = b["evaluation_episodes"]
    cfg.run.model_selection = MODEL_SELECTION
    cfg.run.selection_interval_meta_steps = max(1, b["meta_steps"] // 10)
    cfg.run.selection_episodes = b["selection_episodes"]
    cfg.seeds.common_eval_set = bool(COMMON_EVAL_SET)
    cfg.semantic.refinement_rounds = int(SEMANTIC_REFINEMENT_ROUNDS)
    cfg.semantic.refinement_episodes = b["refinement_episodes"]
    cfg.semantic.refinement_mode = SEMANTIC_REFINEMENT_MODE
    cfg.semantic.raw_confidence_mode = RAW_CONFIDENCE_MODE
    cfg.run.device = DEVICE
    if overrides is not None:
        overrides(cfg)
    cfg.validate()
    return cfg


def run_dir(scenario, n_agents, tag, seed):
    return os.path.join(RESULTS_DIR, f"{scenario}_N{n_agents}", tag, f"seed{seed}")


def _compact_outcomes(ev):
    """Per-episode 0/1 outcomes [success, collision, offroad, timeout] in common-test-set order."""
    return [[int(e["success"]), int(e["collision"]), int(e["offroad"]), int(e["timeout"])] for e in ev["episodes"]]


def _variant_aggregate(cv, name, ev=None):
    v = (cv or {}).get(name)
    if v is None:
        return None
    if "same_as" in v:
        if v["same_as"] == "success" and ev is not None:
            return ev["aggregate"]
        return _variant_aggregate(cv, v["same_as"], ev)
    return v.get("aggregate")


_MATCH_CACHE = {}


def _run_matches_current(tag, seed, scenario=None, n_agents=None):
    scenario, n_agents = scenario or SCENARIO, n_agents or N_AGENTS
    out = run_dir(scenario, n_agents, tag, seed)
    if not os.path.exists(os.path.join(out, "evaluation.json")):
        return False
    key = (out, os.path.getmtime(os.path.join(out, "evaluation.json")))
    if key not in _MATCH_CACHE:
        meta = _load_json(os.path.join(out, "metadata.json")) or {}
        try:
            cur = build_config(scenario, n_agents, meta.get("preset", tag), seed, PROFILE).to_dict()
            ok = _config_equivalent(meta.get("config"), cur)[0]
        except Exception:
            ok = False
        if not ok:
            print(f"  [summary] excluded {out}: produced with different settings than the current ones")
        _MATCH_CACHE[key] = ok
    return _MATCH_CACHE[key]


def finished_runs():
    """{preset: [seeds]} of every finished run in this folder that matches the current settings."""
    base = os.path.join(RESULTS_DIR, f"{SCENARIO}_N{N_AGENTS}")
    found = {}
    if not os.path.isdir(base):
        return found
    for tag in os.listdir(base):
        d = os.path.join(base, tag)
        if not os.path.isdir(d):
            continue
        seeds = sorted(int(s[4:]) for s in os.listdir(d) if s.startswith("seed") and s[4:].isdigit())
        seeds = [s for s in seeds if _run_matches_current(tag, s)]
        if seeds:
            found[tag] = seeds
    order = {p: i for i, p in enumerate(PRESET_ORDER)}
    return dict(sorted(found.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0])))


def load_eval(tag, seed):
    return _load_json(os.path.join(run_dir(SCENARIO, N_AGENTS, tag, seed), "evaluation.json"))


METRICS = ["success", "collision", "offroad", "timeout", "team_return"]


def _fmt_ms(vals, scale=100.0, nd=1):
    s = mean_std_ci(vals)
    return f"{scale * s['mean']:.{nd}f} ± {scale * s['std']:.{nd}f}"


def summarize_results():
    """Main controlled comparison (Reviewer 3 comment 1, Reviewer 2 comment 9): test set, seed level.
    Selected checkpoint = reported result; the final checkpoint is reported next to it (Reviewer 3 comment 2)."""
    runs = finished_runs()
    rows = []
    for tag, seeds in runs.items():
        for seed in seeds:
            out = run_dir(SCENARIO, N_AGENTS, tag, seed)
            ev = load_eval(tag, seed)
            fin = _variant_aggregate(_load_json(os.path.join(out, "checkpoint_variants.json")), "final", ev) or {}
            rows.append({"preset": tag, "seed": seed, **{m: ev["aggregate"][m] for m in METRICS},
                         **{f"final_{m}": fin.get(m, np.nan) for m in ("success", "collision", "timeout")}})
    per_seed = pd.DataFrame(rows, columns=["preset", "seed", *METRICS, "final_success", "final_collision", "final_timeout"])
    if per_seed.empty:
        print("no finished runs yet")
        return per_seed
    per_seed.to_csv(os.path.join(RESULTS_DIR, f"per_seed_results_{SCENARIO}_N{N_AGENTS}.csv"), index=False)  # Reviewer 3 comment 3
    summary = []
    for tag in runs:
        d = per_seed[per_seed.preset == tag]
        r = {"preset": tag, "n_seeds": len(d)}
        for m in ("success", "collision", "timeout"):
            s = mean_std_ci(d[m].values)
            r[m] = f"{100 * s['mean']:.1f} ± {100 * s['std']:.1f}"
            r[m + " 95% CI"] = f"[{100 * s['ci_low']:.1f}, {100 * s['ci_high']:.1f}]" if len(d) > 1 else "-"
        r["team return"] = f"{d.team_return.mean():.2f} ± {d.team_return.std(ddof=1) if len(d) > 1 else 0.0:.2f}"
        fin = d.final_success.dropna()
        r["final ckpt success"] = _fmt_ms(fin.values) if len(fin) else "n/a"
        finc = d.final_collision.dropna()
        r["final ckpt collision"] = _fmt_ms(finc.values) if len(finc) else "n/a"
        summary.append(r)
    _report(f"T1 {SCENARIO} N={N_AGENTS} — test set (%, mean ± std over training seeds; selected checkpoint unless noted)",
            pd.DataFrame(summary).set_index("preset"), "T1_main")
    ref = per_seed[per_seed.preset == REFERENCE_PRESET]
    comp = []
    for tag in runs:
        d = per_seed[per_seed.preset == tag]
        if tag == REFERENCE_PRESET or len(d) < 2 or len(ref) < 2:
            continue
        for m in ("success", "collision"):
            c = compare_methods(ref[m].values, d[m].values)
            comp.append({"comparison": f"{REFERENCE_PRESET} − {tag}", "metric": m,
                         "diff (pp)": round(100 * c["diff_mean"], 1),
                         "bootstrap 95% CI (pp)": f"[{100 * c['bootstrap_ci_low']:.1f}, {100 * c['bootstrap_ci_high']:.1f}]",
                         "Welch p": round(c["welch_t_p"], 4), "Mann-Whitney p": round(c["mann_whitney_p"], 4)})
    if comp:
        _report("T1b Pairwise differences over training seeds (3 vs 3: Mann-Whitney min p = 0.1 -> read the CI)",
                pd.DataFrame(comp), "T1b_pairwise")
    return per_seed


SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a8a86", "#6b4fbb", "#c0392b"]


def plot_learning_curves():
    """Reviewer 3 minor 1/3: training curves = mean ± std over ALL seeds (n shown in the legend), plus the
    validation curve used for checkpoint selection; efficiency = AULC and episodes to several thresholds per seed."""
    runs = finished_runs()
    color = {tag: SERIES_COLORS[i % len(SERIES_COLORS)] for i, tag in enumerate(runs)}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    eff, w = [], 1
    for tag, seeds in runs.items():
        curves, vals = [], []
        for s in seeds:
            out = run_dir(SCENARIO, N_AGENTS, tag, s)
            lc = _load_json(os.path.join(out, "learning_curve.json"))
            if lc is not None:
                curves.append((s, np.array([float(r["success"]) for r in lc])))
            sel = ((_load_json(os.path.join(out, "metadata.json")) or {}).get("model_selection") or {}).get("log") or []
            if sel:
                vals.append(([r["episodes"] for r in sel], [r["val_success"] for r in sel]))
        if curves:
            L = min(len(c) for _, c in curves)
            w = max(1, min(50, L // 5))
            sm = np.array([np.convolve(c[:L], np.ones(w) / w, mode="valid") for _, c in curves])
            xs = np.arange(sm.shape[1]) + w
            mu, sd = sm.mean(0), sm.std(0)
            ax.plot(xs, mu, lw=2, color=color[tag], label=f"{tag} (n={len(curves)})")
            ax.fill_between(xs, mu - sd, mu + sd, color=color[tag], alpha=0.15, lw=0)
            for s, c in curves:
                eff.append({"preset": tag, "seed": s, "AULC": area_under_learning_curve(c, w), **episodes_to_thresholds(c, window=w)})
        if vals:
            K = min(len(v[1]) for v in vals)
            v = np.array([x[1][:K] for x in vals])
            ax2.errorbar(vals[0][0][:K], v.mean(0), yerr=v.std(0), color=color[tag], lw=2, marker="o", ms=4, capsize=3,
                         label=f"{tag} (n={len(vals)})")
    for a_, yl, tl in ((ax, f"training success (moving average, window {w})", "training episodes"),
                       (ax2, "validation success (selection episodes)", "validation curve used for model selection")):
        a_.set_xlabel("training episode")
        a_.set_ylabel(yl)
        a_.set_ylim(-0.02, 1.02)
        a_.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.set_title(f"{SCENARIO} N={N_AGENTS}: {tl} (mean ± std over seeds)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"learning_curves_{SCENARIO}_N{N_AGENTS}.png"), dpi=150)
    plt.show()
    if eff:
        df = pd.DataFrame(eff)
        os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
        df.to_csv(os.path.join(RESULTS_DIR, "tables", f"training_efficiency_per_seed_{SCENARIO}_N{N_AGENTS}.csv"), index=False)
        _report("T10 Training efficiency (AULC; episodes to several success thresholds; mean, std over seeds)",
                df.drop(columns="seed").groupby("preset").agg(["mean", "std"]).round(3), "T10_training_efficiency")


INTERVENTION_KEYS = ["semantic_match_rate", "semantic_admission_rate", "semantic_action_change_rate", "shield_trigger_rate",
                     "shield_intervention_rate", "shield_topology_remap_rate", "shield_safety_intervention_rate",
                     "shield_emergency_rate", "speed_floor_rate", "coordination_intervention_rate", "proposal_executed_agreement",
                     "proposal_executed_physical_agreement", "double_intervention_rate", "triple_intervention_rate"]


def report_interventions():
    """Reviewer 2 comment 5 / Reviewer 3 comment 2 / Reviewer 1: how often each stage modifies the executed action."""
    rows = []
    for tag, seeds in finished_runs().items():
        for seed in seeds:
            iv = load_eval(tag, seed)["aggregate"]["interventions"]
            rows.append({"preset": tag, **{k: iv.get(k) for k in INTERVENTION_KEYS}})
    if rows:
        df = pd.DataFrame(rows).groupby("preset", sort=False)
        _report("T2 Per-stage modification of executed actions (% of test agent-steps, mean over seeds)",
                (df.mean(numeric_only=True) * 100).round(1).T, "T2_interventions")


def _calibration_row(label, seed, cal, meta=None):
    dg = cal["diagnostics"]
    cp = (meta or {}).get("calibrator") or {}
    g = lambda k, m: (dg.get(k) or {}).get(m)
    return {"testbed": label, "seed": seed,
            "ECE raw κ̂": g("semantic_C_uncalibrated_raw", "ece"), "ECE Platt": g("semantic_C_platt", "ece"),
            "ECE isotonic": g("semantic_C_isotonic", "ece"),
            "Brier raw": g("semantic_C_uncalibrated_raw", "brier"), "Brier Platt": g("semantic_C_platt", "brier"),
            "NLL raw": g("semantic_C_uncalibrated_raw", "nll"), "NLL Platt": g("semantic_C_platt", "nll"),
            "Platt A": cp.get("A"), "unconstrained A": cp.get("unconstrained_A"),
            "label rate (B)": cal["counts"].get("semantic_label_rate_B"),
            "records A/B/C": "/".join(str(v) for v in (cal["counts"].get("semantic_records") or {}).values()),
            "NF val AUROC": g("neuro_fuzzy_fit", "val_auroc"), "NF ECE (C)": g("neuro_fuzzy_holdout_C", "ece")}


def report_calibration():
    """Reviewer 2 comment 1: calibration evaluated AS calibration (held-out split C), per seed and per testbed."""
    rows, first = [], None
    for tag in ("full_pnal",):
        for seed in finished_runs().get(tag, []):
            d = run_dir(SCENARIO, N_AGENTS, tag, seed)
            cal, meta = _load_json(os.path.join(d, "calibration.json")), _load_json(os.path.join(d, "metadata.json"))
            if cal is None:
                continue
            rows.append(_calibration_row(f"{SCENARIO} N={N_AGENTS} (main runs)", seed, cal, meta))
            if first is None and "semantic_C_platt" in cal["diagnostics"]:
                first = (tag, seed, cal["diagnostics"])
    for sc, n in globals().get("CALIBRATION_TESTBEDS", []):  # cross-testbed calibration: not part of this file
        if (sc, n) == (SCENARIO, N_AGENTS) and rows:
            continue
        for seed in SEEDS:
            d = os.path.join(RESULTS_DIR, "calibration_only", f"{sc}_N{n}", f"seed{seed}")
            cal = _load_json(os.path.join(d, "calibration.json"))
            if cal is not None:
                rows.append(_calibration_row(f"{sc} N={n} (calibration stage only)", seed, cal, _load_json(os.path.join(d, "calibrator.json"))))
            elif os.path.exists(os.path.join(d, "failed.json")):
                rows.append({"testbed": f"{sc} N={n}", "seed": seed, "ECE raw κ̂": None,
                             "records A/B/C": "FAILED: " + _load_json(os.path.join(d, "failed.json"))["error"][:120]})
    if rows:
        _report("T3 Semantic-confidence calibration on held-out calibration split C (ECE / Brier / NLL; "
                "unconstrained Platt A < 0 would mean raw κ̂ is uninformative)", pd.DataFrame(rows).round(3), "T3_calibration")
    if first:
        tag, seed, dg = first
        fig, ax = plt.subplots(figsize=(4.8, 4.6))
        ax.plot([0, 1], [0, 1], color="#8a8a86", lw=1, ls="--", label="perfect calibration")
        for key, name, col in (("semantic_C_uncalibrated_raw", "semantic raw κ̂", SERIES_COLORS[1]),
                               ("semantic_C_platt", "semantic Platt", SERIES_COLORS[0]),
                               ("semantic_C_isotonic", "semantic isotonic", SERIES_COLORS[3]),
                               ("neuro_fuzzy_holdout_C", "NF state score", SERIES_COLORS[2])):
            b = dg.get(key, {}).get("reliability_bins", [])
            if b:
                ax.plot([r["mean_conf"] for r in b], [r["accuracy"] for r in b], marker="o", ms=5, lw=2, color=col,
                        label=f"{name} (ECE {dg[key]['ece']:.3f})")
        ax.set_xlabel("predicted confidence / score")
        ax.set_ylabel("observed frequency of label 1")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_title(f"Reliability on calibration split C\n{tag}, seed {seed}", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"reliability_{SCENARIO}_N{N_AGENTS}.png"), dpi=150)
        plt.show()


def _set_flags(**kw):
    def f(s):
        nf = dataclasses.replace(s.cfg.flags, **kw)
        s.cfg.flags = nf
        s.fusion.flags = nf
    return f


def _chain(*fs):
    def f(s):
        for g in fs:
            g(s)
    return f


def _set_attr(path, value):
    """path like 'fusion.admission_rule' or 'cfg.shield.risk_mode'."""
    def f(s):
        obj = s
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)
    return f


def _isotonic_gate(s):
    s.fusion.semantic.calibrator = s.artifacts.isotonic_calibrator


def _coord_weights(w):
    def f(s):
        for k, v in zip(("w_r", "w_u", "w_g", "w_m"), w):
            setattr(s.cfg.coordination, k, float(v))
    return f


def _density(bg, p):
    def f(cfg):
        configure_intersection_traffic(cfg.scenario, destinations=INTERSECTION_DESTINATIONS, background_vehicles=bg,
                                       spawn_probability=p, observation_vehicles_count=cfg.scenario.observation_vehicles_count)
    return f


def _corrupted_library(art):
    return dataclasses.replace(art, library=reversed_ranking_copy(art.library, keep_reliability=True))


def _reversed_recalibrated(out):
    """Reversed library passed through the normal pre-deployment calibration (refinement OFF so that the gate
    itself is tested).  Cached in <run>/analysis/reversed_library/.  Returns an artefact override or a string
    explaining why the calibration stage rejected the library."""
    d = os.path.join(out, "analysis", "reversed_library")
    if os.path.exists(os.path.join(d, "rejected.json")):
        return _load_json(os.path.join(d, "rejected.json"))["reason"]
    if not os.path.exists(os.path.join(d, "calibrator_platt.json")):
        meta = _load_json(os.path.join(out, "metadata.json"))
        seed = meta["seed"]
        cfg = build_config(SCENARIO, N_AGENTS, meta["preset"], seed, PROFILE)
        cfg.semantic.refinement_rounds = 0
        s0 = PnALSystem(cfg)
        lib = reversed_ranking_copy(SemanticLibrary.load(os.path.join(out, "semantic_library.json")), keep_reliability=False)
        os.makedirs(d, exist_ok=True)
        try:
            art2 = run_calibration_stage(cfg, s0.env, s0.extractor, library=lib)
        except CalibrationDataError as e:
            reason = f"calibration stage rejected the reversed library: {e}"
            write_json({"reason": reason}, os.path.join(d, "rejected.json"))
            return reason
        if art2.calibrator is None:
            reason = "calibration stage produced no calibrator: " + str(art2.diagnostics.get("semantic_source_inactive"))
            write_json({"reason": reason}, os.path.join(d, "rejected.json"))
            return reason
        art2.library.save(os.path.join(d, "semantic_library.json"))
        art2.calibrator.save(os.path.join(d, "calibrator_platt.json"))
        art2.isotonic_calibrator.save(os.path.join(d, "calibrator_isotonic.json"))
        write_json({"counts": art2.counts, "diagnostics": art2.diagnostics}, os.path.join(d, "calibration.json"))
    lib = SemanticLibrary.load(os.path.join(d, "semantic_library.json"))
    lib.freeze()
    cal = SemanticConfidenceCalibrator.load(os.path.join(d, "calibrator_platt.json"))
    iso = SemanticConfidenceCalibrator.load(os.path.join(d, "calibrator_isotonic.json"))
    return lambda art: dataclasses.replace(art, library=lib, calibrator=cal, isotonic_calibrator=iso)


SEMANTIC_PRESETS = ["full_pnal", "ippo_full_fusion"]


STRESS_PRESETS = ["full_pnal", "ippo_full_fusion", "ippo_plus_shield", "ippo"]


def revision_conditions():
    """Registry of evaluation-time conditions: name -> spec.  Every condition changes ONE factor of a trained,
    frozen run and is evaluated on the common test set (sweeps: validation set, because they concern settings
    that are chosen on validation data)."""
    C = {}

    def add(name, group, reviewer, presets, desc, setup=None, cfg_mut=None, restore=None, split="test"):
        C[name] = dict(name=name, group=group, reviewer=reviewer, presets=presets, desc=desc, setup=setup,
                       cfg_mut=cfg_mut, restore=restore, split=split)

    F = ["full_pnal"]
    FF = ["full_pnal", "ippo_full_fusion"]  # presets with the complete online stack (other testbeds: ippo_full_fusion)
    # online component ablation with the learned source held fixed (R3-5, R2-1, R2-5; factorial: R1)
    add("gate_uncalibrated", "online_ablation", "R3-5", FF, "semantic gate on raw κ̂ (no Platt calibration)", _set_flags(use_calibration=False))
    add("gate_always_trust", "online_ablation", "R3-5", FF, "matched semantic prior always admitted (no gate)", _set_flags(always_trust_semantic=True))
    add("gate_isotonic", "online_ablation", "R2-1", F, "gate on isotonic instead of Platt calibration", _isotonic_gate)
    add("no_semantic", "online_ablation", "R3-5", FF, "semantic source removed", _set_flags(use_semantic=False, use_calibration=False))
    add("no_coordination", "online_ablation", "R2-5", FF, "coordination layer removed", _set_flags(use_coordination=False))
    add("no_shield", "online_ablation", "R3-1", FF + ["ippo_plus_shield"], "SafetyShield removed", _set_flags(use_safety_shield=False))
    # neuro-fuzzy risk gate switched off (delta_t = 0: A_hard feasibility screening + fallback only) -- isolates the
    # contribution of the neuro-fuzzy state-risk source inside the SafetyShield (Reviewer 2 comment 2)
    add("no_risk_gate", "online_ablation", "R2-2", FF + ["ippo_plus_shield"],
        "neuro-fuzzy risk gate off (δ_t = 0; A_hard feasibility + fallback only)", _set_attr("fusion.delta_t", 0.0))
    # 2x2 with the trained ippo / ippo_plus_shield runs: Shield at execution time vs Shield during training (R3-1)
    add("shield_added", "online_ablation", "R3-1", ["ippo"], "SafetyShield added at evaluation (policy trained without it)",
        _set_flags(use_safety_shield=True))
    add("no_semantic_no_shield", "online_ablation", "R1", F, "semantic source and SafetyShield removed",
        _set_flags(use_semantic=False, use_calibration=False, use_safety_shield=False))
    add("no_fusion", "online_ablation", "R2-9", FF, "every online source removed (learned policy only)",
        _set_flags(use_semantic=False, use_calibration=False, use_safety_shield=False, use_coordination=False))
    # alternative fusion rules on the same learned source (R2-3, R2-4)
    for rule, why in (("soft_mixture", "R2-3"), ("product_of_experts", "R2-3"), ("relative_reliability", "R2-4")):
        add(f"rule_{rule}", "fusion_rule", why, F, f"semantic admission rule = {rule}", _set_attr("fusion.admission_rule", rule))
    # learned-source contribution (R3-2)
    LS = ["full_pnal", "ippo_full_fusion", "ippo_plus_shield"]
    add("source_untrained", "learned_source", "R3-2", LS, "proposal = the run's untrained initial policy (same online stack)",
        restore=lambda out: dict(load_policy=False))
    add("source_uniform_random", "learned_source", "R3-2", LS, "proposal = uniform random action (same online stack)",
        _set_attr("proposal_source", "uniform_random"))
    add("source_cruise", "learned_source", "R3-2", LS, "proposal = scripted 'accelerate to top speed' (same online stack)",
        _set_attr("proposal_source", "cruise"))
    add("source_cruise_no_fusion", "learned_source", "R3-2", FF, "scripted 'accelerate to top speed', no online source",
        _chain(_set_attr("proposal_source", "cruise"),
               _set_flags(use_semantic=False, use_calibration=False, use_safety_shield=False, use_coordination=False)))
    # do-nothing reference (always MAINTAIN, no online source): exposes testbeds a trivial policy already solves (R2-7)
    add("source_maintain_no_fusion", "learned_source", "R2-7", FF, "scripted 'always maintain', no online source",
        _chain(_set_attr("proposal_source", "maintain"),
               _set_flags(use_semantic=False, use_calibration=False, use_safety_shield=False, use_coordination=False)))
    # state-level vs action-conditioned risk gate (R2-2) -- comparison only, the main method stays s = f_NF(x)
    add("risk_action_conditioned", "risk_gate", "R2-2", F, "risk gate on s_a = f_NF(x'_a) (one-step prediction)",
        _set_attr("cfg.shield.risk_mode", "action_conditioned"))
    # stress tests (R2-7): perception noise, unseen traffic density, misleading semantic knowledge
    for nm, (sp, sv) in OBS_NOISE_LEVELS.items():
        add(nm, "stress", "R2-7", STRESS_PRESETS, f"Gaussian noise on observed neighbours (σ_pos {sp} m, σ_vel {sv} m/s)",
            _set_attr("obs_noise", (sp, sv)))
    if SCENARIO == "intersection":
        for nm, (bg, p) in DENSITY_LEVELS.items():
            add(nm, "stress", "R2-7", STRESS_PRESETS, f"unseen background traffic ({bg} IDM vehicles at reset, spawn prob. {p})",
                cfg_mut=_density(bg, p))
    elif SCENARIO == "roundabout":
        for nm, k in ROUNDABOUT_DENSITY_LEVELS.items():
            add(nm, "stress", "R2-7", STRESS_PRESETS,
                f"unseen traffic density ({k} background vehicles instead of {ROUNDABOUT_BACKGROUND_VEHICLES})", cfg_mut=_rb_density(k))
    add("semantic_corrupted", "stress", "R2-7", SEMANTIC_PRESETS,
        "library rankings reversed AFTER calibration (q and Platt of the correct library kept), gated",
        restore=lambda out: dict(artifacts_override=_corrupted_library))
    add("semantic_corrupted_always_trust", "stress", "R2-7", SEMANTIC_PRESETS,
        "library rankings reversed after calibration, admitted whenever matched",
        _set_flags(always_trust_semantic=True), restore=lambda out: dict(artifacts_override=_corrupted_library))
    add("semantic_reversed_recalibrated", "stress", "R2-7", SEMANTIC_PRESETS,
        "reversed library passed through the normal calibration stage (refinement off), gated",
        restore=lambda out: _reversed_recalibrated(out))
    add("semantic_unrefined_recalibrated", "stress", "R3-4", SEMANTIC_PRESETS,
        "original (unrefined) library passed through the normal calibration stage (refinement off), gated",
        restore=lambda out: _unrefined_recalibrated(out))
    # sensitivity sweeps on the VALIDATION split (R2-2 δ_t, R2-6 coordination weights, R1 τc / fallback ranking)
    for t in TAU_GRID:
        add(f"sweep_tau_{t}", "sweep_tau", "R1/R3", F, f"τc = {t}", _set_attr("fusion.tau_c", float(t)), split="validation")
    for dlt in DELTA_GRID:
        add(f"sweep_delta_{dlt}", "sweep_delta", "R2-2", ["full_pnal", "ippo_plus_shield"], f"δ_t = {dlt}",
            _set_attr("fusion.delta_t", float(dlt)), split="validation")
    for nm, w in COORDINATION_WEIGHT_SETS.items():
        add(f"sweep_coord_{nm}", "sweep_coordination", "R2-6", F, f"(w_r, w_u, w_g, w_m) = {tuple(w)}", _coord_weights(w), split="validation")
    for nm, rk in FALLBACK_RANKINGS.items():
        add(f"sweep_fallback_{nm}", "sweep_fallback", "R1", F, f"fallback ranking {' > '.join(rk)}",
            _set_attr("cfg.shield.fallback_ranking", tuple(rk)), split="validation")
    return C


def run_condition(preset, seed, spec):
    """Evaluate one condition on one finished run; cached in <run>/analysis/<name>.json (resumable)."""
    out = run_dir(SCENARIO, N_AGENTS, preset, seed)
    if not _run_matches_current(preset, seed):
        return None
    n = SWEEP_EPISODES if spec["split"] == "validation" else ANALYSIS_EPISODES
    meta = _load_json(os.path.join(out, "metadata.json"))
    key = {"condition": spec["name"], "desc": spec["desc"], "split": spec["split"], "episodes": n,
           "run_fingerprint": config_fingerprint(meta["config"])}
    path = os.path.join(out, "analysis", f"{spec['name']}.json")
    cached = _load_json(path)
    if cached is not None and cached.get("key") == key:
        return cached
    cfg = build_config(SCENARIO, N_AGENTS, preset, seed, PROFILE)
    if spec["cfg_mut"] is not None:
        spec["cfg_mut"](cfg)
        cfg.validate()
    rkw = spec["restore"](out) if spec["restore"] is not None else {}
    t0 = time.time()
    if isinstance(rkw, str):  # the pre-deployment stage refused the stressed library: recorded, not evaluated
        res = {"key": key, "status": "rejected_at_calibration", "reason": rkw}
    else:
        if callable(rkw):
            rkw = dict(artifacts_override=rkw)
        sysm = PnALSystem.restore(cfg, _checkpoint_dir(preset, seed, out), map_location=DEVICE, **rkw)
        if spec["setup"] is not None:
            spec["setup"](sysm)
        ev = sysm.evaluate(n, split=spec["split"])
        res = {"key": key, "status": "evaluated", "aggregate": {k: v for k, v in ev["aggregate"].items() if k != "interventions"},
               "interventions": ev["aggregate"]["interventions"], "outcomes": _compact_outcomes(ev),
               "minutes": (time.time() - t0) / 60}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_json(res, path)
    a = res.get("aggregate") or {}
    print(f"  [analysis] {SCENARIO} N={N_AGENTS} {preset} s{seed} {spec['name']} ({spec['split']}, {n} ep): "
          + (f"success={a['success']:.2f} collision={a['collision']:.2f} timeout={a['timeout']:.2f}" if a else res.get("reason", ""))
          + f" | {(time.time() - t0) / 60:.1f} min", flush=True)
    return res


def _reference_outcomes(preset, seed, split, n):
    """Reference = the unchanged pathway on the same episodes."""
    if split == "test":
        ev = load_eval(preset, seed)
        if ev is not None and len(ev["episodes"]) >= n:
            return _compact_outcomes(ev)[:n], {k: v for k, v in ev["aggregate"].items() if k != "interventions"}, ev["aggregate"]["interventions"]
    ref = run_condition(preset, seed, dict(name=f"reference_{split}", group="reference", reviewer="-", presets=[preset],
                                           desc="unchanged pathway", setup=None, cfg_mut=None, restore=None, split=split))
    return (ref["outcomes"], ref["aggregate"], ref["interventions"]) if ref else (None, None, None)


def report_conditions():
    """T4-T9: one table per analysis group; Δ = condition − unchanged pathway on the same runs and episodes
    (paired; CI by bootstrap over episodes within seeds; exact McNemar p on discordant episodes)."""
    runs = finished_runs()
    groups = {}
    for c in revision_conditions().values():
        groups.setdefault(c["group"], []).append(c)
    titles = {"online_ablation": ("T4 Online component ablation, learned source held fixed (test set)", "T4_online_ablation"),
              "fusion_rule": ("T5 Alternative semantic-admission rules on the same learned source (test set)", "T5_fusion_rules"),
              "learned_source": ("T6 Contribution of the learned source: proposal replaced, online stack unchanged (test set)", "T6_learned_source"),
              "risk_gate": ("T7 State-level vs action-conditioned risk gate (test set; main method = state level)", "T7_risk_gate"),
              "stress": ("T8 Stress tests (test set)", "T8_stress"),
              "sweep_tau": ("S1 τc sensitivity (validation set)", "S1_tau"),
              "sweep_delta": ("S2 δ_t sensitivity (validation set)", "S2_delta"),
              "sweep_coordination": ("S3 Coordination-weight sensitivity (validation set)", "S3_coordination_weights"),
              "sweep_fallback": ("S4 Fallback-ranking sensitivity (validation set)", "S4_fallback_ranking")}
    for g, conds in groups.items():  # every group / condition with cached results (independent of this session's filters)
        rows = []
        presets = [p for p in runs if any(p in c["presets"] for c in conds)]
        for p in presets:
            seeds = runs[p]
            split = conds[0]["split"]
            n = SWEEP_EPISODES if split == "validation" else ANALYSIS_EPISODES
            refs = {s: _reference_outcomes(p, s, split, n) for s in seeds} if split == "test" else {}
            if split == "test":
                ro = [refs[s][0] for s in seeds if refs[s][0] is not None]
                rows.append({"preset": p, "condition": "unchanged pathway (reference)", "reviewer": "-", "n_seeds": len(ro),
                             **_rates(ro), **_iv_cols([refs[s][2] for s in seeds if refs[s][2]])})
            for c in conds:
                if p not in c["presets"]:
                    continue
                res = {s: _load_json(os.path.join(run_dir(SCENARIO, N_AGENTS, p, s), "analysis", f"{c['name']}.json")) for s in seeds}
                res = {s: r for s, r in res.items() if r is not None and r.get("key", {}).get("episodes") == n}
                if not res:
                    continue
                rej = [r for r in res.values() if r.get("status") == "rejected_at_calibration"]
                if rej:
                    rows.append({"preset": p, "condition": c["desc"], "reviewer": c["reviewer"], "n_seeds": len(rej),
                                 "note": "library rejected by the calibration stage (semantic source inactive)"})
                ok = {s: r for s, r in res.items() if r.get("status") == "evaluated"}
                if not ok:
                    continue
                co = [ok[s]["outcomes"] for s in ok]
                row = {"preset": p, "condition": c["desc"], "reviewer": c["reviewer"], "n_seeds": len(co), **_rates(co),
                       **_iv_cols([ok[s]["interventions"] for s in ok])}
                if split == "test" and all(refs.get(s) and refs[s][0] is not None for s in ok):
                    for m, j in (("success", 0), ("collision", 1)):
                        e = paired_episode_effect([[o[j] for o in refs[s][0]] for s in ok], [[o[j] for o in ok[s]["outcomes"]] for s in ok])
                        row[f"Δ{m} pp [95% CI]"] = f"{100 * e['diff_mean']:+.1f} [{100 * e['ci_low']:+.1f}, {100 * e['ci_high']:+.1f}]"
                        row[f"McNemar p ({m})"] = f"{e['mcnemar_p']:.3g}"
                rows.append(row)
        if any(r["condition"] != "unchanged pathway (reference)" for r in rows):  # skip tables with reference rows only
            title, name = titles.get(g, (g, g))
            _report(title, pd.DataFrame(rows).set_index(["preset", "condition"]), name)


def _rates(outcomes_by_seed):
    """success / collision / timeout in %, mean ± std over seeds (each seed = mean over its episodes)."""
    out = {}
    for m, j in (("success", 0), ("collision", 1), ("timeout", 3)):
        v = [float(np.mean([o[j] for o in x])) for x in outcomes_by_seed]
        out[m] = _fmt_ms(v) if v else "n/a"
    return out


def _iv_cols(ivs):
    if not ivs:
        return {}
    g = lambda k: float(np.mean([iv.get(k) or 0.0 for iv in ivs]))
    return {"sem. admit %": round(100 * g("semantic_admission_rate"), 1),
            "shield (safety) %": round(100 * g("shield_safety_intervention_rate"), 1),
            "coord %": round(100 * g("coordination_intervention_rate"), 1),
            "executed = proposal (physical) %": round(100 * g("proposal_executed_physical_agreement"), 1)}


def report_selection():
    """T9 (Reviewer 2 comment 10): which checkpoint each validation criterion picks, and its test result."""
    rows = []
    names = {"success": "max val. success, ties → lower collision (pre-declared, reported)",
             "team_return": "max val. team return", "success_minus_collision": "max val. (success − collision)",
             "final": "last checkpoint (no selection)"}
    for tag, seeds in finished_runs().items():
        per = {k: [] for k in names}
        for s in seeds:
            out = run_dir(SCENARIO, N_AGENTS, tag, s)
            cv = _load_json(os.path.join(out, "checkpoint_variants.json"))
            if cv is None:
                continue
            ev = load_eval(tag, s)
            for k in names:
                a = _variant_aggregate(cv, k, ev)
                if a is not None:
                    per[k].append((a["success"], a["collision"], a["timeout"], cv[k]["meta_step"] + 1))
        for k, v in per.items():
            if v:
                arr = np.array(v, dtype=float)
                rows.append({"preset": tag, "criterion": names[k], "n_seeds": len(v), "selected meta-step (per seed)": ", ".join(str(int(x)) for x in arr[:, 3]),
                             "test success": _fmt_ms(arr[:, 0]), "test collision": _fmt_ms(arr[:, 1]), "test timeout": _fmt_ms(arr[:, 2])})
    if rows:
        _report("T9 Checkpoint-selection criteria (chosen on validation; test reported for each)",
                pd.DataFrame(rows).set_index(["preset", "criterion"]), "T9_selection_criteria")


def report_compute():
    lat, par = [], []
    for tag, seeds in finished_runs().items():
        for seed in seeds:
            ev = load_eval(tag, seed)
            if "compute" in ev:
                lat.append({"preset": tag, **ev["compute"]["latency_ms_per_call"]})
                par.append({"preset": tag, **ev["compute"]["parameter_counts"]})
    if lat:
        _report("Latency, ms per call (mean over seeds; hardware dependent)", pd.DataFrame(lat).groupby("preset", sort=False).mean(numeric_only=True).T.round(3), "latency")
        _report("Parameter counts", pd.DataFrame(par).groupby("preset", sort=False).first().T.astype(int), "parameters")


def summarize_all():
    _record(f"\n### Summary at {_now()} — recomputed over all finished runs of this folder for the current settings\n",
            {"event": "summary_start"})
    summarize_results()
    plot_learning_curves()
    report_interventions()
    report_calibration()
    report_selection()
    report_conditions()
    report_compute()
    for png in (f"learning_curves_{SCENARIO}_N{N_AGENTS}.png", f"reliability_{SCENARIO}_N{N_AGENTS}.png"):
        if os.path.exists(os.path.join(RESULTS_DIR, png)):
            _record(f"\n![{png}]({png})")


def _risk_delta_for(scenario, n_agents=None, allow_auto=False):
    return float(RISK_THRESHOLD_DELTA)  # delta_t = 0.3 for every testbed (as in all runs of the paper)


RECORD_MD, RECORD_JSONL = "experiment_record.md", "experiment_record.jsonl"   # set per task by _use_records
RUNS_CSV, CONSOLE_LOG = "all_runs.csv", "console_log.txt"
RUNS_CSV_FIELDS = ["recorded_at", "status", "scenario", "n_agents", "tag", "preset", "seed", "profile", "config_fingerprint",
                   "success", "collision", "offroad", "timeout", "team_return", "test_episodes",
                   "final_success", "final_collision", "final_timeout",
                   "semantic_admission_rate", "shield_intervention_rate", "coordination_intervention_rate",
                   "selected_meta_step", "nf_val_auroc", "platt_A", "platt_unconstrained_A", "refinement_changes",
                   "minutes", "run_dir"]


# ===========================
# Supplementary evaluation-time experiments (three Colab windows: SHARD A / B / C = seed 0 / 1 / 2)
# ===========================
# Nothing is trained.  (1) τc sensitivity on the validation split for the PnAL runs of the 8 settings of the paper;
# (2) the Roundabout component attribution: its conditions are run for the seeds still missing (finished ones are read
# from the cache).  Before any test-set condition of a run, the unchanged pathway is re-evaluated by THIS file on
# REPRO_EPISODES test episodes and must reproduce the saved results episode by episode (otherwise the run's test-set
# conditions are not evaluated and the failure is reported).
import glob
import shutil

_TASK_FAILURES = []


def _rb_density(k):
    """Unseen traffic density for Roundabout: k background vehicles instead of ROUNDABOUT_BACKGROUND_VEHICLES."""
    def f(cfg):
        cfg.scenario.total_vehicles = int(cfg.scenario.n_agents) + int(k)
        cfg.scenario.enable_background = True
    return f


def _unrefined_recalibrated(out):
    """The ORIGINAL (unrefined) template library passed through the normal pre-deployment calibration with
    refinement OFF; learned policy, NF model, shield and coordination are the run's own.  Same procedure as
    _reversed_recalibrated without reversing the rankings.  Cached in <run>/analysis/unrefined_library/."""
    d = os.path.join(out, "analysis", "unrefined_library")
    if os.path.exists(os.path.join(d, "rejected.json")):
        return _load_json(os.path.join(d, "rejected.json"))["reason"]
    if not os.path.exists(os.path.join(d, "calibrator_platt.json")):
        meta = _load_json(os.path.join(out, "metadata.json"))
        cfg = build_config(SCENARIO, N_AGENTS, meta["preset"], meta["seed"], PROFILE)
        cfg.semantic.refinement_rounds = 0
        s0 = PnALSystem(cfg)
        lib = SemanticLibrary.from_config(cfg.semantic)
        os.makedirs(d, exist_ok=True)
        try:
            art2 = run_calibration_stage(cfg, s0.env, s0.extractor, library=lib)
        except CalibrationDataError as e:
            reason = f"calibration stage rejected the unrefined library: {e}"
            write_json({"reason": reason}, os.path.join(d, "rejected.json"))
            return reason
        if art2.calibrator is None:
            reason = "calibration stage produced no calibrator: " + str(art2.diagnostics.get("semantic_source_inactive"))
            write_json({"reason": reason}, os.path.join(d, "rejected.json"))
            return reason
        art2.library.save(os.path.join(d, "semantic_library.json"))
        art2.calibrator.save(os.path.join(d, "calibrator_platt.json"))
        art2.isotonic_calibrator.save(os.path.join(d, "calibrator_isotonic.json"))
        write_json({"counts": art2.counts, "diagnostics": art2.diagnostics}, os.path.join(d, "calibration.json"))
    lib = SemanticLibrary.load(os.path.join(d, "semantic_library.json"))
    lib.freeze()
    cal = SemanticConfidenceCalibrator.load(os.path.join(d, "calibrator_platt.json"))
    iso = SemanticConfidenceCalibrator.load(os.path.join(d, "calibrator_isotonic.json"))
    return lambda art: dataclasses.replace(art, library=lib, calibrator=cal, isotonic_calibrator=iso)


def _use_records(suffix):
    global RECORD_MD, RECORD_JSONL, RUNS_CSV, CONSOLE_LOG
    os.makedirs(os.path.join(RESULTS_DIR, "records"), exist_ok=True)
    RECORD_MD, RECORD_JSONL = (os.path.join("records", f"experiment_record_{suffix}.md"),
                               os.path.join("records", f"experiment_record_{suffix}.jsonl"))
    RUNS_CSV, CONSOLE_LOG = os.path.join("records", f"all_runs_{suffix}.csv"), os.path.join("records", f"console_log_{suffix}.txt")


def _activate(key):
    """Switch every protocol global to one setting of SETTINGS (scenario, N, δ_temp rule, results folder)."""
    global SCENARIO, N_AGENTS, DELTA_TEMP_RULE, RESULTS_DIR, PRESETS
    st = SETTINGS[key]
    SCENARIO, N_AGENTS, DELTA_TEMP_RULE = st["scenario"], int(st["n"]), st["delta_temp_rule"]
    RESULTS_DIR = st["results"]
    PRESETS = list(st["sources"])
    _MATCH_CACHE.clear()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return st


def _key():
    return f"{SCENARIO}_N{N_AGENTS}"


CHECKPOINT_FILES = ("pnal_policy.pt", "neuro_fuzzy.pt", "meta_scheduler.pt")   # large binary files of a run
_NEEDED_FILES = ("evaluation.json", "metadata.json", "semantic_library.json", "pnal_policy.pt", "neuro_fuzzy.pt")


def _source_equivalent(d, preset, seed):
    """(usable?, reasons): finished, complete, and saved with the protocol of the active setting (incl. δ_temp rule)."""
    missing = [f for f in _NEEDED_FILES if not os.path.exists(os.path.join(d, f))]
    if missing:
        return False, [f"missing {missing}"]
    saved = (_load_json(os.path.join(d, "metadata.json")) or {}).get("config")
    return _config_equivalent(saved, build_config(SCENARIO, N_AGENTS, preset, seed, PROFILE).to_dict())


_SOURCE_CACHE = {}


def _resolve_source(preset, seed):
    """The finished run of the paper for (active setting, preset, seed); raises with the reasons if there is none
    or if the choice is ambiguous (never falls back silently to another run)."""
    ck = (SCENARIO, N_AGENTS, preset, seed, DELTA_TEMP_RULE)
    if ck in _SOURCE_CACHE:
        return _SOURCE_CACHE[ck]
    root, mode = SETTINGS[_key()]["sources"][preset]
    base = os.path.join(root, f"{SCENARIO}_N{N_AGENTS}", preset)
    over = SOURCE_OVERRIDES.get((SCENARIO, N_AGENTS, preset, seed))
    if over:
        cands = [over]
    elif mode == "current":
        cands = [os.path.join(base, f"seed{seed}")]
    elif mode == "archived":
        cands = sorted(glob.glob(os.path.join(base, f"seed{seed}_superseded_*")))
    else:
        raise ValueError(f"unknown source mode {mode}")
    ok, bad = [], []
    for c in cands:
        good, why = _source_equivalent(c, preset, seed)
        (ok if good else bad).append(c if good else f"{c} ({'; '.join(why[:3])})")
    if mode == "archived" and not ok and not over:  # never archived: the run may still sit in its original folder
        c = os.path.join(base, f"seed{seed}")
        good, why = _source_equivalent(c, preset, seed)
        (ok if good else bad).append(c if good else f"{c} ({'; '.join(why[:3])})")
        if good:
            print(f"[source] {_key()} {preset} seed {seed}: no archived run fits; using the run in its original folder {c}")
    if not ok:
        raise RuntimeError(f"[source] no usable run for {_key()} {preset} seed {seed} (mode '{mode}', δ_temp rule "
                           f"{DELTA_TEMP_RULE}): " + (" | ".join(bad) if bad else f"nothing found under {base}"))
    if len(ok) > 1:
        res = {json.dumps({k: (_load_json(os.path.join(c, "evaluation.json")) or {}).get("aggregate", {}).get(k)
                           for k in ("success", "collision", "offroad", "timeout")}, sort_keys=True) for c in ok}
        if len(res) > 1:
            raise RuntimeError(f"[source] several archived runs fit {_key()} {preset} seed {seed} with DIFFERENT results: {ok}. "
                               f"Set SOURCE_OVERRIDES[({SCENARIO!r}, {N_AGENTS}, {preset!r}, {seed})] to the run used in the paper.")
    _SOURCE_CACHE[ck] = ok[-1]
    return ok[-1]


def _copy_atomic(a, b):
    tmp = f"{b}.part{os.getpid()}"
    shutil.copyfile(a, tmp)
    os.replace(tmp, b)


def register_runs():
    """For every preset / seed of the active setting: resolve the source run and make sure the run folder in
    RESULTS_DIR holds its small result files (checkpoints stay in the read-only source and are staged locally) and
    the evaluation-time analyses already computed on that run (never overwritten).  A different run already sitting
    in the destination raises."""
    rows = []
    for preset in PRESETS:
        for seed in SEEDS:
            src = _resolve_source(preset, seed)
            dst = run_dir(SCENARIO, N_AGENTS, preset, seed)
            fp = config_fingerprint((_load_json(os.path.join(src, "metadata.json")) or {})["config"])
            if os.path.abspath(src) != os.path.abspath(dst):
                if os.path.exists(os.path.join(dst, "metadata.json")):
                    fp_dst = config_fingerprint((_load_json(os.path.join(dst, "metadata.json")) or {}).get("config") or {})
                    ev_s = (_load_json(os.path.join(src, "evaluation.json")) or {}).get("aggregate")
                    ev_d = (_load_json(os.path.join(dst, "evaluation.json")) or {}).get("aggregate")
                    if fp_dst != fp or ev_s != ev_d:
                        raise RuntimeError(f"[source] {dst} already holds a DIFFERENT run than {src}; move it away and start again")
                os.makedirs(dst, exist_ok=True)
                copied = []
                for f in sorted(os.listdir(src)):
                    a, b = os.path.join(src, f), os.path.join(dst, f)
                    if os.path.isfile(a) and f not in CHECKPOINT_FILES and (not os.path.exists(b) or os.path.getsize(b) != os.path.getsize(a)):
                        _copy_atomic(a, b)
                        copied.append(f)
                sa, da = os.path.join(src, "analysis"), os.path.join(dst, "analysis")
                if os.path.isdir(sa):  # analyses already computed on this very run (cache keys are checked on use)
                    for root_, _, files in os.walk(sa):
                        for f in files:
                            a = os.path.join(root_, f)
                            b = os.path.join(da, os.path.relpath(a, sa))
                            if not os.path.exists(b) and ".part" not in f:
                                os.makedirs(os.path.dirname(b), exist_ok=True)
                                _copy_atomic(a, b)
                                copied.append("analysis/" + os.path.relpath(a, sa))
                if copied:
                    print(f"[source] {_key()} {preset} seed {seed}: copied {len(copied)} small file(s) from {src}", flush=True)
            info = {"setting": _key(), "preset": preset, "seed": seed, "source": src, "config_fingerprint": fp,
                    "delta_temp_rule": DELTA_TEMP_RULE if preset in ("full_pnal", "ippo_full_fusion") else "n/a (no semantic source)"}
            prev = _load_json(os.path.join(dst, "supplement_source.json"))
            if prev is None or prev.get("source") != src:
                write_json({**info, "registered_at": _now()}, os.path.join(dst, "supplement_source.json"))
                _record(f"\n- {_now()}: {_key()} | {preset} | seed {seed}: source `{src}` (fingerprint `{fp}`; not retrained)",
                        {"event": "supplement_source", **info})
            rows.append(info)
    _MATCH_CACHE.clear()
    return rows


def _stage_local(src):
    """Copy a source run (without analysis/) to the local disk of the runtime and verify it: every file has the
    source size and every .pt file loads.  Up to 3 attempts; raises with the reason if the source cannot be read.
    Safe when several processes stage the same run (per-process temporary names, atomic rename)."""
    local = os.path.join(LOCAL_CHECKPOINT_DIR, os.path.relpath(src, "/"))
    for attempt in range(1, 4):
        try:
            os.makedirs(local, exist_ok=True)
            for f in sorted(os.listdir(src)):
                a, b = os.path.join(src, f), os.path.join(local, f)
                if os.path.isfile(a) and ".part" not in f and (not os.path.exists(b) or os.path.getsize(b) != os.path.getsize(a)):
                    _copy_atomic(a, b)
                if os.path.isfile(a) and ".part" not in f and os.path.getsize(b) != os.path.getsize(a):
                    raise OSError(f"size mismatch after copy: {f}")
            for f in CHECKPOINT_FILES:
                if os.path.exists(os.path.join(local, f)):
                    torch.load(os.path.join(local, f), map_location="cpu", weights_only=True)
            return local
        except Exception as e:
            print(f"[stage] {src} -> {local}: attempt {attempt} failed ({type(e).__name__}: {e})", flush=True)
            time.sleep(20 * attempt)
    raise RuntimeError(f"could not stage the checkpoints of {src} on the local disk (3 attempts)")


_STAGED = {}


def _checkpoint_dir(preset, seed, out):
    """Folder a run is RESTORED from: verified local copy of the source run.  Analyses are written to `out`."""
    src = _resolve_source(preset, seed)
    if src not in _STAGED:
        _STAGED[src] = _stage_local(src)
    return _STAGED[src]


class ReproductionMismatch(RuntimeError):
    pass


def reproduction_check(preset, seed):
    """Re-evaluate the UNCHANGED pathway of the run with this file on the first REPRO_EPISODES common test episodes;
    outcomes and episode lengths must equal the saved test results episode by episode."""
    out = run_dir(SCENARIO, N_AGENTS, preset, seed)
    meta, stored = _load_json(os.path.join(out, "metadata.json")), load_eval(preset, seed)
    n = min(REPRO_EPISODES, len(stored["episodes"]))
    fields = ("success", "collision", "offroad", "timeout", "steps")
    saved = [[int(e[f]) for f in fields] + [float(e["team_return"])] for e in stored["episodes"][:n]]
    key = {"check": "reproduction", "episodes": n, "run_fingerprint": config_fingerprint(meta["config"]),
           "delta_temp_rule": DELTA_TEMP_RULE, "saved_results_sha": hashlib.sha256(json.dumps(saved).encode()).hexdigest()[:16]}
    path = os.path.join(out, "analysis", "reproduction_check.json")
    res = _load_json(path)
    if res is None or res.get("key") != key:
        t0 = time.time()
        sysm = PnALSystem.restore(build_config(SCENARIO, N_AGENTS, preset, seed, PROFILE), _checkpoint_dir(preset, seed, out),
                                  map_location=DEVICE)
        ev = sysm.evaluate(n, split="test")
        pairs = list(zip(ev["episodes"], stored["episodes"][:n]))
        same = sum(all(int(a[f]) == int(b[f]) for f in fields) for a, b in pairs)
        dret = max(abs(float(a["team_return"]) - float(b["team_return"])) for a, b in pairs)
        res = {"key": key, "episodes": n, "identical_outcomes_and_lengths": same, "max_abs_team_return_diff": dret,
               "status": "identical" if same == n else "DIFFERENT", "minutes": (time.time() - t0) / 60}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_json(res, path)
        print(f"  [reproduce] {_key()} {preset} s{seed}: {same}/{n} test episodes identical to the saved results "
              f"(max |Δ team return| {dret:.2e}) -> {res['status']}", flush=True)
    if res["status"] != "identical":
        raise ReproductionMismatch(f"{_key()} {preset} seed {seed}: only {res['identical_outcomes_and_lengths']}/{res['episodes']} "
                                   "test episodes reproduce the saved results; its test-set conditions are NOT evaluated")
    return res


def _cost(task):
    key, preset, block = task
    per = 2.5 if block == "validation" else 4.0
    scale = {"intersection": 0.6 + 0.15 * SETTINGS[key]["n"], "roundabout": 0.6 + 0.2 * SETTINGS[key]["n"], "highway": 0.6}
    return len(SUPPLEMENT_PLAN[key][preset][block]) * per * scale[SETTINGS[key]["scenario"]] * (1.0 if preset == "full_pnal" else 0.5)


def _supp_tasks():
    tasks = [(key, preset, block) for key in SETTINGS for preset, blocks in SUPPLEMENT_PLAN[key].items() for block in blocks]
    return sorted(tasks, key=_cost, reverse=True)   # longest first: balances the parallel processes


def _supp_task(task):
    """Returns the list of failures (never silent: printed, recorded and listed again at the end of the window)."""
    key, preset, block = task
    _activate(key)
    _use_records(f"supp_{key}_{preset}_{block}_{SHARD}")
    try:
        start_console_log()
    except Exception:
        pass
    torch.set_num_threads(1)
    specs = revision_conditions()
    names = SUPPLEMENT_PLAN[key][preset][block]
    print(f"\n[supplement] start {key} | {preset} | {block} ({len(names)} conditions) | seeds {SEEDS}", flush=True)
    failures = []
    for seed in SEEDS:
        try:
            if block == "test":
                reproduction_check(preset, seed)
            for name in names:
                spec = specs[name]
                if preset not in spec["presets"] or spec["split"] != block:
                    raise ValueError(f"condition {name} is not defined for {preset} on the {block} split")
                if run_condition(preset, seed, spec) is None:
                    raise RuntimeError(f"run {key} {preset} seed {seed} is not registered with the current protocol")
        except Exception as e:
            import traceback
            print(f"[supplement] FAILED {key} | {preset} | {block} | seed {seed}: {type(e).__name__}: {e}\n"
                  + ("" if isinstance(e, ReproductionMismatch) else traceback.format_exc()), flush=True)
            _record(f"\n- {_now()}: **FAILED** {key} | {preset} | {block} | seed {seed}: `{type(e).__name__}: {e}`",
                    {"event": "task_failed", "setting": key, "preset": preset, "block": block, "seed": seed, "error": repr(e)})
            failures.append((key, preset, block, seed, repr(e)))
    print(f"[supplement] done {key} | {preset} | {block}" + (f" with {len(failures)} failure(s)" if failures else ""), flush=True)
    return failures


def _running_from_file():
    """Parallel processes are fresh interpreters ('spawn') that re-import this file; pasted into a notebook cell
    there is no file to re-import, and the tasks run one after another."""
    main = sys.modules.get("__main__")
    return bool(getattr(main, "__file__", None)) and str(main.__file__).endswith(".py")


# ---------------------------------------------------------------------------
# Cross-setting tables (written to SUPP_DIR/tables)
# ---------------------------------------------------------------------------
LABELS = {"intersection_N2": "Intersection N=2", "intersection_N3": "Intersection N=3", "intersection_N4": "Intersection N=4",
          "intersection_N5": "Intersection N=5", "roundabout_N2": "Roundabout N=2", "roundabout_N4": "Roundabout N=4",
          "highway_N2": "Highway N=2", "highway_N4": "Highway N=4"}


def _rd(key, preset, seed):
    st = SETTINGS[key]
    return os.path.join(st["results"], f"{st['scenario']}_N{st['n']}", preset, f"seed{seed}")


def _cond(key, preset, seed, name):
    """Cached result of one condition, or None (only results evaluated with this file's episode counts)."""
    r = _load_json(os.path.join(_rd(key, preset, seed), "analysis", f"{name}.json"))
    if r is None or r.get("status") != "evaluated":
        return None
    n = SWEEP_EPISODES if r["key"]["split"] == "validation" else ANALYSIS_EPISODES
    return r if r["key"].get("episodes") == n else None


def _report_x(title, df, name):
    """Cross-setting table: shown, saved to SUPP_DIR/tables/<name>.csv (+ .tex) and appended to the record."""
    print("\n" + title)
    _show(df)
    os.makedirs(os.path.join(SUPP_DIR, "tables"), exist_ok=True)
    base = os.path.join(SUPP_DIR, "tables", name)
    df.to_csv(base + ".csv")
    try:
        with open(base + ".tex", "w", encoding="utf-8") as f:
            f.write(df.to_latex(escape=True, caption=title, label=f"tab:{name}"))
    except Exception as e:
        print(f"  (LaTeX export skipped: {type(e).__name__})")
    _record(f"\n#### {title}\n\n" + _md_table(df) + "\n", {"event": "table", "name": name, "title": title})


def _effect_cols(ref, cond):
    """Δ (pp, condition − reference) with episode-bootstrap CI and exact McNemar p, success and collision."""
    out = {}
    for m, j in (("success", 0), ("collision", 1)):
        e = paired_episode_effect([[o[j] for o in r] for r in ref], [[o[j] for o in c] for c in cond])
        out[f"Δ{m} pp [95% CI]"] = f"{100 * e['diff_mean']:+.1f} [{100 * e['ci_low']:+.1f}, {100 * e['ci_high']:+.1f}]"
        out[f"McNemar p ({m})"] = f"{e['mcnemar_p']:.3g}"
    return out


def _admit(rs):
    return round(100 * float(np.mean([r["interventions"].get("semantic_admission_rate") or 0.0 for r in rs])), 1)


def _sweep_table(names, label_of, ref_name, title, name, pivot_title, pivot_name):
    """Sweep on the validation split: every point vs the paper setting (ref_name) on the same episodes."""
    rows, piv = [], {}
    for key in SETTINGS:
        if not set(names) & set(SUPPLEMENT_PLAN[key].get("full_pnal", {}).get("validation", [])):
            continue
        for nm in names:
            res = {s: _cond(key, "full_pnal", s, nm) for s in (0, 1, 2)}
            res = {s: r for s, r in res.items() if r is not None}
            if not res:
                continue
            row = {"setting": LABELS[key], "value": label_of(nm), "n_seeds": len(res), **_rates([r["outcomes"] for r in res.values()]),
                   "sem. admit %": _admit(res.values())}
            ref = {s: _cond(key, "full_pnal", s, ref_name) for s in res}
            both = [s for s in res if ref.get(s) is not None]
            if nm != ref_name and both:
                row.update(_effect_cols([ref[s]["outcomes"] for s in both], [res[s]["outcomes"] for s in both]))
                row["paired seeds"] = len(both)
            rows.append(row)
            piv.setdefault(label_of(nm), {})[LABELS[key]] = round(100 * float(np.mean([np.mean([o[0] for o in r["outcomes"]]) for r in res.values()])), 1)
    if not rows:
        return
    _report_x(title, pd.DataFrame(rows).set_index(["setting", "value"]), name)
    p = pd.DataFrame(piv).T
    p = p.reindex([label_of(nm) for nm in names if label_of(nm) in p.index])
    best = {c: ", ".join(str(i) for i in p.index[p[c] == p[c].max()]) for c in p.columns}
    p = p.astype(object)
    p.loc["best (max val. success)"] = pd.Series(best)
    _report_x(pivot_title, p, pivot_name)


def report_sources():
    rows, agg = [], {}
    for key in SETTINGS:
        for preset in SETTINGS[key]["sources"]:
            for s in (0, 1, 2):
                info = _load_json(os.path.join(_rd(key, preset, s), "supplement_source.json"))
                ev = _load_json(os.path.join(_rd(key, preset, s), "evaluation.json"))
                if info is None or ev is None:
                    continue
                a = ev["aggregate"]
                rows.append({"setting": LABELS[key], "preset": preset, "seed": s, "test success %": round(100 * a["success"], 1),
                             "test collision %": round(100 * a["collision"], 1), "δ_temp rule": info["delta_temp_rule"],
                             "config fingerprint": info["config_fingerprint"], "source run": info["source"]})
                agg.setdefault((key, preset), []).append(100 * a["success"])
    if not rows:
        return
    _report_x("R0 Runs used (finished runs of the paper; nothing retrained)", pd.DataFrame(rows).set_index(["setting", "preset", "seed"]),
              "R0_sources")
    chk = []
    for (key, preset), v in agg.items():
        paper = PAPER_TEST_SUCCESS.get((key, preset))
        row = {"setting": LABELS[key], "preset": preset, "n_seeds": len(v), "test success % (these runs)": round(float(np.mean(v)), 1),
               "paper table": paper}
        if paper is not None and len(v) == 3:
            row["check"] = "matches" if abs(round(float(np.mean(v)), 1) - paper) < 0.05 else "DIFFERS"
        chk.append(row)
    _report_x("R0b Check against the paper tables (3-seed mean of the saved test results)",
              pd.DataFrame(chk).set_index(["setting", "preset"]), "R0b_sources_check")


def report_reproduction():
    rows = []
    for key in SETTINGS:
        for preset in SETTINGS[key]["sources"]:
            for s in (0, 1, 2):
                r = _load_json(os.path.join(_rd(key, preset, s), "analysis", "reproduction_check.json"))
                if r is not None:
                    rows.append({"setting": LABELS[key], "preset": preset, "seed": s, "test episodes compared": r["episodes"],
                                 "identical outcome and length": r["identical_outcomes_and_lengths"],
                                 "max |Δ team return|": f"{r['max_abs_team_return_diff']:.2e}", "status": r["status"]})
    if rows:
        _report_x("R1 Reproduction of the saved test results by this file (unchanged pathway, episode by episode)",
                  pd.DataFrame(rows).set_index(["setting", "preset", "seed"]), "R1_reproduction_check")


def summarize_supplement():
    """Per-setting tables in each setting's results folder (Roundabout: the complete attribution tables, recomputed
    over all seeds), then the cross-setting tables in SUPP_DIR/tables."""
    global RESULTS_DIR, SCENARIO, N_AGENTS
    for key in SETTINGS:
        _activate(key)
        _use_records(f"supp_summary_{SHARD}")
        print(f"\n{'=' * 80}\n{LABELS[key]} — tables in {os.path.join(RESULTS_DIR, 'tables')}\n{'=' * 80}")
        if SETTINGS[key]["scenario"] == "roundabout":
            summarize_all()          # Roundabout attribution tables, now with the missing seeds
        else:
            report_conditions()
    RESULTS_DIR, SCENARIO, N_AGENTS = SUPP_DIR, "all", 0
    _use_records(f"supp_{SHARD}")
    print(f"\n{'=' * 80}\nCross-setting tables — {os.path.join(SUPP_DIR, 'tables')}\n{'=' * 80}")
    report_sources()
    report_reproduction()
    tau_lbl = lambda nm: float(nm.rsplit("_", 1)[1])
    _sweep_table(TAU_SWEEP, tau_lbl, "sweep_tau_0.6",
                 "S1x τc sensitivity across testbeds and agent numbers (validation split, frozen runs; Δ vs τc = 0.6 on the same episodes)",
                 "S1x_tau_cross_testbed", "S1x-pivot validation success (%) by τc (rows) and setting (columns)", "S1x_tau_pivot")


if __name__ == "__main__":
    if USE_DRIVE and not os.path.isdir(DRIVE):
        raise RuntimeError("Google Drive is not mounted. Run `from google.colab import drive; drive.mount('/content/drive')` "
                           "in a notebook cell first, then start this script.")
    pd.set_option("display.max_columns", None)  # print every column of the summary tables
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    RESULTS_DIR = SUPP_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _use_records(f"supp_{SHARD}")
    start_console_log()
    record_session_start()
    print("\n" + "=" * 80)
    print(f"PnAL supplementary evaluation-time experiments | shard {SHARD} | seeds {SEEDS} | nothing is trained")
    print("=" * 80)
    print("versions:", environment_versions())
    print(f"device: {DEVICE} | results: {SUPP_DIR} (Roundabout: {RB_ATTR_DIR})")
    failed = []
    if SHARD != "summary":
        ready = []
        for key in SETTINGS:  # 1. sources: the paper's finished runs, verified (sequential; small files only)
            _activate(key)
            _use_records(f"supp_{SHARD}")
            try:
                for r in register_runs():
                    print(f"[source] {key} {r['preset']} seed {r['seed']}: {r['source']} (fingerprint {r['config_fingerprint']}, "
                          f"δ_temp rule {r['delta_temp_rule']})")
                ready.append(key)
            except Exception as e:
                print(f"[source] FAILED {key}: {type(e).__name__}: {e}", flush=True)
                _record(f"\n- {_now()}: **FAILED** sources of {key}: `{e}`", {"event": "source_failed", "setting": key, "error": repr(e)})
                failed.append((key, "-", "sources", SEEDS, repr(e)))
        RESULTS_DIR = SUPP_DIR
        _use_records(f"supp_{SHARD}")
        tasks = [t for t in _supp_tasks() if t[0] in ready]   # 2. evaluation-time conditions, longest tasks first
        W = max(1, min(int(SUPP_WORKERS), len(tasks), os.cpu_count() or 1)) if _running_from_file() else 1
        print(f"\n{len(tasks)} tasks | {W} parallel process(es) on {os.cpu_count()} CPU cores")
        if W == 1:
            for t in tasks:
                failed += _supp_task(t)
        else:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(W, maxtasksperchild=1) as pool:
                for r in pool.imap_unordered(_supp_task, tasks, chunksize=1):
                    failed += r
    summarize_supplement()  # 3. tables over everything finished so far (all windows)
    _record(f"\n_Session finished {_now()}_", {"event": "session_end"})
    print("\n" + "=" * 80)
    print("Results:", SUPP_DIR, "| tables:", os.path.join(SUPP_DIR, "tables"), "| Roundabout tables:", os.path.join(RB_ATTR_DIR, "tables"))
    print("=" * 80)
    if failed:
        for f in failed:
            print("  FAILED:", f)
        raise RuntimeError(f"{len(failed)} supplement task(s) of this window failed (listed above; finished parts are kept, "
                           "start the window again to retry only what is missing)")