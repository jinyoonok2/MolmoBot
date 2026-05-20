from molmobot_spoc.eval.config.spoc_policy_configs import (
    SPOCPolicyConfig,
    SPOCRBY1ArticulatedManipPolicyConfig,
    SPOCRBY1RigidManipPolicyConfig,
)
from pathlib import Path
import datetime
import logging
import os
import socket
from molmo_spaces.configs.robot_configs import RBY1MConfig
from molmo_spaces.configs.camera_configs import RBY1GoProD455CameraSystem
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from huggingface_hub import snapshot_download

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log = logging.getLogger(__name__)


def _snapshot_download_ipv4(repo_id: str) -> str:
    """Download a Hugging Face snapshot using IPv4 only.

    This machine has working IPv4 access to Hugging Face but broken IPv6 access, which
    can make `snapshot_download` hang before it notices the local cache.
    """

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    original_getaddrinfo = socket.getaddrinfo

    def ipv4_getaddrinfo(*args, **kwargs):
        return [
            info
            for info in original_getaddrinfo(*args, **kwargs)
            if info[0] == socket.AF_INET
        ]

    socket.getaddrinfo = ipv4_getaddrinfo
    try:
        return snapshot_download(repo_id)
    finally:
        socket.getaddrinfo = original_getaddrinfo


def _resolve_hf_snapshot(repo_id: str) -> str:
    """Resolve a checkpoint snapshot, preferring the local cache."""

    try:
        path = snapshot_download(repo_id, local_files_only=True)
        log.info("Using cached Hugging Face snapshot for %s: %s", repo_id, path)
        return path
    except Exception as exc:
        log.info(
            "No cached Hugging Face snapshot for %s, downloading with IPv4 only: %s",
            repo_id,
            exc,
        )
        return _snapshot_download_ipv4(repo_id)


class RBY1EvalBaseConfig(JsonBenchmarkEvalConfig):
    wandb_project: str = "mujoco-thor-opening-eval"
    use_wandb: bool = True
    use_passive_viewer: bool = False
    viewer_cam_dict: dict = {"camera": "robot_0/camera_follower"}
    filter_for_successful_trajectories: bool = False
    task_type: str = ""
    task_horizon: int = 200
    policy_dt_ms: float = 100.0  # Default policy time step
    ctrl_dt_ms: float = 20.0  # Default control time step
    sim_dt_ms: float = 4.0  # Default simulation time step

    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()
    robot_config: RBY1MConfig = RBY1MConfig(
        command_mode={
            "arm": "joint_rel_position",
            "gripper": "joint_position",
            "base": "holo_joint_rel_planar_position",
            "head": None,  # Must be None - RBY1 head actuation is disabled
            "torso": "height",
        }
    )
    hf_model_name: str | None = None

    def _init_policy_config(self) -> SPOCPolicyConfig:
        """Override parent's policy config initialization to use SPOC policy"""
        from molmobot_spoc.eval.spoc_policy import SPOCModelPolicy

        # Set the policy class on the already-configured SPOCDoorOpeningPolicyConfig
        self.policy_config.policy_cls = SPOCModelPolicy

        return self.policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
        assert self.task_type != "", "Set the task_type in the eval config."
        if self.hf_model_name is not None:
            self.policy_config.checkpoint_dir = _resolve_hf_snapshot(self.hf_model_name)


class RBY1ArticulatedManipEvalConfig(RBY1EvalBaseConfig):
    task_type: str = "open"
    hf_model_name: str | None = "allenai/MolmoBot-SPOC-RBY1Articulated"
    policy_config: SPOCRBY1ArticulatedManipPolicyConfig = (
        SPOCRBY1ArticulatedManipPolicyConfig()
    )


class RBY1RigidManipEvalConfig(RBY1EvalBaseConfig):
    task_type: str = "pick"
    hf_model_name: str | None = "allenai/MolmoBot-SPOC-RBY1Rigid"
    policy_config: SPOCRBY1RigidManipPolicyConfig = SPOCRBY1RigidManipPolicyConfig()
