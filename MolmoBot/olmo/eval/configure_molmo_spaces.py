import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.camera_configs import RBY1GoProD455CameraSystem
from molmo_spaces.configs.robot_configs import FrankaRobotConfig, RBY1MConfig
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import BasePolicy, InferencePolicy, StatefulPolicy
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig

logger = logging.getLogger(__name__)


@dataclass
class SynthVLAPolicyState:
    action_buffer: list[dict[str, np.ndarray]] | None = None
    buffer_index: int = 0
    step_count: int = 0
    obs_history: list[dict] | None = None

class SynthVLAPolicy(InferencePolicy, StatefulPolicy):
    """Minimal InferencePolicy wrapper for SynthManipMolmoInferenceWrapper.

    Extracts observations from the simulator and calls the wrapped agent.
    Uses action buffering: predicts action_horizon actions, executes
    execute_horizon before refreshing the buffer.

    Loads the SynthManipMolmoInferenceWrapper from config.policy_config.checkpoint_path.
    """

    def __init__(
        self,
        config: MlSpacesExpConfig,
        task_type: str,
    ):
        super().__init__(config)
        self.task = task_type
        self.camera_names = config.policy_config.camera_names
        self.action_move_group_names = config.policy_config.action_move_group_names
        self.action_spec = config.policy_config.action_spec
        self.action_horizon = config.policy_config.action_horizon
        self.execute_horizon = config.policy_config.execute_horizon
        self.action_type = config.policy_config.action_type
        self.relative_max_joint_delta = getattr(config.policy_config, "relative_max_joint_delta", None)
        if self.relative_max_joint_delta is not None:
            self.relative_max_joint_delta = np.array(self.relative_max_joint_delta)
        self.action_generator_seed = self._resolve_action_generator_seed()
        self._action_generator: torch.Generator | None = None

        self.action_buffer: list[dict[str, np.ndarray]] = []
        self.buffer_index = 0
        self.step_count = 0
        self.obs_history: list[dict] = []
        self._prepared = False

        self.prepare_model()

        # Default obs is 1 and delta is 8
        self.input_window_size = getattr(self.agent.model_config , "n_obs_steps", 1)
        self.obs_step_delta = getattr(self.agent.model_config , "obs_step_delta", 8)
        self._reset_action_generator()

    def _resolve_action_generator_seed(self) -> int | None:
        raw_seed = os.environ.get(
            "RBY1_MODEL_ACTION_SEED",
            getattr(self.config.policy_config, "action_generator_seed", None),
        )
        if raw_seed in (None, ""):
            return None
        if isinstance(raw_seed, str) and raw_seed.strip().lower() in {"none", "null", "off"}:
            return None
        return int(raw_seed)

    def _reset_action_generator(self) -> None:
        if self.action_generator_seed is None:
            self._action_generator = None
            return

        device = getattr(self.agent, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self._action_generator = torch.Generator(device=device)
        self._action_generator.manual_seed(self.action_generator_seed)
        logger.info("Using seeded MolmoBot action generator: seed=%d device=%s", self.action_generator_seed, device)

    def get_state(self):
        return SynthVLAPolicyState(
            action_buffer=self.action_buffer,
            buffer_index=self.buffer_index,
            step_count=self.step_count,
            obs_history=self.obs_history,
        )

    def set_state(self, state: SynthVLAPolicyState):
        self.action_buffer = state.action_buffer
        self.buffer_index = state.buffer_index
        self.step_count = state.step_count
        self.obs_history = state.obs_history if state.obs_history is not None else []

    def prepare_model(self):
        """Load SynthManipMolmoInferenceWrapper from checkpoint specified in policy_config."""
        if self._prepared:
            return
        self._prepared = True
        from olmo.models.molmobot.inference_wrapper import SynthManipMolmoInferenceWrapper

        checkpoint_path = self.config.policy_config.checkpoint_path
        # logger.info(f"Loading SynthManipMolmoInferenceWrapper from: {checkpoint_path}")
        states_mode = getattr(self.config.policy_config, "states_mode", "cross_attn")
        self.agent = SynthManipMolmoInferenceWrapper(checkpoint_path=checkpoint_path, states_mode=states_mode)
        # logger.info("SynthManipMolmoInferenceWrapper loaded successfully")

    def reset(self):
        self.action_buffer = []
        self.buffer_index = 0
        self.step_count = 0
        self.obs_history = []
        self._reset_action_generator()

    def _populate_action_buffer(self, observation) -> None:
        """Call agent to get new action chunk and populate the buffer."""
        obs = observation[0] if isinstance(observation, list) else observation

        # Extract images from observations
        images = []
        for cam_name in self.camera_names:
            if cam_name == "exo_camera_1":
                cam_name =  "droid_shoulder_light_randomization" if "droid_shoulder_light_randomization" in obs else "exo_camera_1"
            elif cam_name == "wrist_camera":
                cam_name = "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"

            if cam_name not in obs:
                raise KeyError(f"Camera '{cam_name}' not in observation. Available: {list(obs.keys())}")

            cam_images = []
            # Simple case: single frame
            if self.input_window_size == 1:
                cam_images.append(obs[cam_name])
            else:
                # Multiple frames: calculate history indices using reference logic
                current_history_len = len(self.obs_history)

                # Only proceed if we have history images
                if current_history_len > 0:
                    # Calculate frame indices relative to current step (like reference implementation)
                    current_step = current_history_len - 1  # Current step is the last index in history

                    for i in range(self.input_window_size):
                        # Use the same logic as _get_camera_frames in synthmanip_dataset
                        frame_idx = current_step - (self.input_window_size - 1 - i) * self.obs_step_delta

                        # Only add valid indices (no padding)
                        if 0 <= frame_idx < current_history_len:
                            cam_images.append(self.obs_history[frame_idx][cam_name])

                assert cam_images, "No frames found when generating observations"

            # Always send a list of images
            images.extend(cam_images)

        # Extract qpos state
        robot_state = obs["robot_state"]
        qpos_parts = []
        for group_name in self.action_move_group_names:
            if "gripper" not in group_name:
                qpos_parts.append(robot_state["qpos"][group_name])
            else:
                qpos_parts.append(robot_state["qpos"][group_name][:self.config.policy_config.gripper_representation_count])

        state = np.concatenate(qpos_parts).astype(np.float32)

        if "task" in obs:
            goal = obs["task"]
        else:
            goal = self.task.get_task_description()

        # Call agent
        pred_actions = self.agent.get_action_chunk(
            images=images,
            task_description=goal,
            state=state,
            generator=self._action_generator,
        )

        # logger.info(f"Predicted action chunk: shape={pred_actions.shape}")

        # Convert to list of action dicts and store in buffer
        self.action_buffer = []
        for t in range(pred_actions.shape[0]):
            action = {}
            start_idx = 0
            for group_name in self.action_move_group_names:
                dim = self.action_spec[group_name]
                selected_action = pred_actions[t, start_idx: start_idx + dim]
                if "gripper" in group_name and self.config.policy_config.clamp_gripper:
                    action[group_name] = np.where(selected_action > 128, 255, 0).astype(selected_action.dtype)
                else:
                    action[group_name] = pred_actions[t, start_idx : start_idx + dim]
                start_idx += dim
            self.action_buffer.append(action)

        self.buffer_index = 0
        # logger.info(f"Populated action buffer with {len(self.action_buffer)} actions")

    def obs_to_model_input(self, obs) -> dict[str, np.ndarray]:
        return obs

    def model_output_to_action(self, model_output) -> dict[str, np.ndarray]:
        return model_output

    def inference_model(self, model_input) -> dict[str, np.ndarray]:
        """Return single action from buffer, refreshing when needed."""
        # Add current observation to history
        obs = model_input[0] if isinstance(model_input, list) else model_input
        self.obs_history.append(obs)

        # Refresh buffer if empty or executed enough actions
        if self.buffer_index >= self.execute_horizon or not self.action_buffer:
            self._populate_action_buffer(model_input)

        action = self.action_buffer[self.buffer_index]
        # logger.info(f"Executing action {self.buffer_index}/{len(self.action_buffer)} (refresh at {self.execute_horizon})")

        self.buffer_index += 1
        self.step_count += 1

        if self.relative_max_joint_delta is not None:
            obs = model_input[0] if isinstance(model_input, list) else model_input
            qpos = obs.get("qpos", obs.get("robot_state", {}).get("qpos", {}))

            for group_name in action:
                if group_name != "arm" and not group_name.endswith("_arm"):
                    continue

                if self.action_type == "joint_pos_rel":
                    predicted_deltas = action[group_name][:7]
                    current_qpos = None
                else:
                    current_qpos = qpos[group_name][:7]
                    predicted_deltas = action[group_name][:7] - current_qpos

                relative_scale = np.abs(predicted_deltas) / self.relative_max_joint_delta
                if np.max(relative_scale) > 1:
                    scaled_predicted_deltas = predicted_deltas / np.max(relative_scale)
                    if current_qpos is None:
                        action[group_name][:7] = scaled_predicted_deltas
                    else:
                        action[group_name][:7] = current_qpos + scaled_predicted_deltas

        return action


class SynthVLAPolicyConfig(BasePolicyConfig):
    """Policy config for SynthVLA models using external SynthManipMolmoInferenceWrapper.

    This config is for models loaded via olmo.models.molmobot.agent.SynthManipMolmoInferenceWrapper,
    NOT the internal SPOC model loading mechanism.
    """

    policy_type: str = "learned"
    action_type: str = "joint_pos_rel"
    policy_cls: type = None  # Set in model_post_init to avoid circular imports
    device: str | None = "cuda" if torch.cuda.is_available() else "cpu"

    # Set to your checkpoint path (local dir or use --hf_repo with serve scripts)
    checkpoint_path: str = ""
    camera_names: list[str] = ["exo_camera_1", "wrist_camera"]
    action_move_group_names: list[str] = ["arm", "gripper"]
    action_spec: dict[str, int] = {
        "arm": 7,  # 7-DOF arm joint positions
        "gripper": 1,  # gripper position
    }
    action_keys: dict[str, str] = {
        "arm": "joint_pos_rel",
        "gripper": "joint_pos",
    }
    action_horizon: int = 16  # Number of action steps predicted per chunk
    execute_horizon: int = 8  # Number of actions to execute before re-querying
    action_generator_seed: int | None = None

    clamp_gripper: bool = True
    gripper_representation_count: int = 1  # Number of gripper state values to input

    states_mode:str = "cross_attn"
    relative_max_joint_delta: list[float] | None = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        if self.policy_cls is None:
            from olmo.eval.configure_molmo_spaces import SynthVLAPolicy

            object.__setattr__(self, "policy_cls", SynthVLAPolicy)


class FrankaState8ClampConfig(JsonBenchmarkEvalConfig):
    policy_config: SynthVLAPolicyConfig = SynthVLAPolicyConfig()
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_dt_ms: float = 66.0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Disable action noise for evaluation
        self.robot_config.action_noise_config.enabled = False
        # Set command modes based on policy action keys
        # RMH note: if you train with non-delta-joint actions, this will need adjustment
        for mg, action_key in self.policy_config.action_keys.items():
            if action_key == "joint_pos_rel" or action_key == "delta_actions":
                if mg == "base":
                    self.robot_config.command_mode[mg] = "holo_joint_rel_planar_position"
                elif "arm" in mg:
                    self.robot_config.command_mode["arm"] = "joint_rel_position"
                elif "gripper" in mg:
                    self.robot_config.command_mode["gripper"] = "joint_rel_position"


class FrankaState8ClampAbsPosConfig(JsonBenchmarkEvalConfig):
    policy_config: SynthVLAPolicyConfig = SynthVLAPolicyConfig(action_type="joint_pos")
    policy_config.action_keys['arm'] = "joint_pos"

    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_dt_ms: float = 66.0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Disable action noise for evaluation
        self.robot_config.action_noise_config.enabled = False
        # Set command modes based on policy action keys
        # RMH note: if you train with non-delta-joint actions, this will need adjustment
        for mg, action_key in self.policy_config.action_keys.items():
            if action_key == "joint_pos_rel" or action_key == "delta_actions":
                if mg == "base":
                    self.robot_config.command_mode[mg] = "holo_joint_rel_planar_position"
                elif "arm" in mg:
                    self.robot_config.command_mode["arm"] = "joint_rel_position"
                elif "gripper" in mg:
                    self.robot_config.command_mode["gripper"] = "joint_rel_position"


class FrankaAbsPosRandomCamConfig(JsonBenchmarkEvalConfig):
    policy_config: SynthVLAPolicyConfig = SynthVLAPolicyConfig(action_type="joint_pos")
    policy_config.action_keys['arm'] = "joint_pos"

    policy_config.camera_names = ["randomized_zed2_analogue_1", "wrist_camera"]

    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_dt_ms: float = 200.0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Disable action noise for evaluation
        self.robot_config.action_noise_config.enabled = False
        # Set command modes based on policy action keys
        # RMH note: if you train with non-delta-joint actions, this will need adjustment
        for mg, action_key in self.policy_config.action_keys.items():
            if action_key == "joint_pos_rel" or action_key == "delta_actions":
                if mg == "base":
                    self.robot_config.command_mode[mg] = "holo_joint_rel_planar_position"
                elif "arm" in mg:
                    self.robot_config.command_mode["arm"] = "joint_rel_position"
                elif "gripper" in mg:
                    self.robot_config.command_mode["gripper"] = "joint_rel_position"


# RMH note: this is the one you actually invoke for the eval launch
class SynthVLAFrankaBenchmarkOriginalEvalConfig(JsonBenchmarkEvalConfig):
    """
    Minimal benchmark-only eval config for SynthVLA Franka pick-and-place.

    Use this config ONLY with JSON benchmarks. All episode-specific configuration
    (cameras, poses, task parameters) comes from the benchmark, not this config.

    This is the recommended pattern for external repos: subclass JsonBenchmarkEvalConfig,
    provide your robot_config and policy_config, and run with run_evaluation().
    """

    policy_config: SynthVLAPolicyConfig = SynthVLAPolicyConfig(gripper_representation_count=1, clamp_gripper=False)
    robot_config: FrankaRobotConfig = FrankaRobotConfig()

    # Set policy_dt to match the trained model's expected control rate
    policy_dt_ms: float = 200.0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Disable action noise for evaluation
        self.robot_config.action_noise_config.enabled = False
        # Set command modes based on policy action keys
        # RMH note: if you train with non-delta-joint actions, this will need adjustment
        for mg, action_key in self.policy_config.action_keys.items():
            if action_key == "joint_pos_rel" or action_key == "delta_actions":
                if mg == "base":
                    self.robot_config.command_mode[mg] = "holo_joint_rel_planar_position"
                elif "arm" in mg:
                    self.robot_config.command_mode["arm"] = "joint_rel_position"
                elif "gripper" in mg:
                    self.robot_config.command_mode["gripper"] = "joint_rel_position"


# ── RBY1 ─────────────────────────────────────────────────────────────────


class SynthVLARBY1PolicyConfig(BasePolicyConfig):
    """Policy config for SynthVLA on RBY1.

    Uses delta actions for base/arms, absolute for grippers.
    Grippers are 1D (squeezed from 2D qpos via gripper_representation_count=1).
    """

    policy_type: str = "learned"
    action_type: str = "joint_pos_rel"
    policy_cls: type = None
    device: str | None = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path: str = ""  # Set to trained checkpoint
    camera_names: list[str] = ["wrist_camera_r", "head_camera", "wrist_camera_l"]
    action_move_group_names: list[str] = [
        "base", "left_arm", "left_gripper", "right_arm", "right_gripper",
    ]
    action_spec: dict[str, int] = {
        "base": 3,
        "left_arm": 7,
        "left_gripper": 1,
        "right_arm": 7,
        "right_gripper": 1,
    }
    action_keys: dict[str, str] = {
        "base": "joint_pos_rel",
        "left_arm": "joint_pos_rel",
        "left_gripper": "joint_pos",  # absolute
        "right_arm": "joint_pos_rel",
        "right_gripper": "joint_pos",  # absolute
    }
    action_horizon: int = 16
    execute_horizon: int = 8
    action_generator_seed: int | None = None

    clamp_gripper: bool = True
    gripper_representation_count: int = 1

    def model_post_init(self, __context) -> None:
        if self.policy_cls is None:
            from olmo.eval.configure_molmo_spaces import SynthVLAPolicy

            object.__setattr__(self, "policy_cls", SynthVLAPolicy)


class SynthVLARBY1EvalConfig(JsonBenchmarkEvalConfig):
    """Base eval config for SynthVLA RBY1.

    Command modes: delta base/arms, absolute grippers.
    Timing matches RBY1 data generation configs.
    """

    policy_config: SynthVLARBY1PolicyConfig = SynthVLARBY1PolicyConfig()
    robot_config: RBY1MConfig = RBY1MConfig()  # matches data gen (rby1_v1.2_site_control.xml)

    # Updated to match config from before Feb20.
    # policy_dt_ms: float = 66.0   # ~15 Hz (matches older RBY1 datagen)
    # ctrl_dt_ms: float = 2.0      # control time step
    # sim_dt_ms: float = 2.0       # simulation time step
    # Config after Feb20 is:
    policy_dt_ms: float = 100.0  # 10 Hz
    ctrl_dt_ms: float = 20.0    # control time step
    sim_dt_ms: float = 4.0      # simulation time step
    task_horizon: int = 400

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
        # Derive command_mode from action_keys (same logic as Maya's DoorOpeningEvalConfig)
        for mg, action_key in self.policy_config.action_keys.items():
            if action_key in ("joint_pos_rel", "delta_actions"):
                if mg == "base":
                    self.robot_config.command_mode["base"] = "holo_joint_rel_planar_position"
                elif "arm" in mg:
                    self.robot_config.command_mode["arm"] = "joint_rel_position"
                elif "gripper" in mg:
                    self.robot_config.command_mode["gripper"] = "joint_rel_position"
                elif mg == "head":
                    self.robot_config.command_mode["head"] = "joint_rel_position"


# ── MolmoBot RBY1 ───────────────────────────────────────────────────────


@dataclass
class MolmoBotRBY1PolicyState(SynthVLAPolicyState):
    conditioning_points: dict | None = None


class MolmoBotRBY1DoorOpeningPolicy(SynthVLAPolicy):
    """MolmoBot policy for RBY1 door opening with point prompts and fisheye warping.

    Extends SynthVLAPolicy with:
    - Fisheye warping on configurable cameras (matching training augmentation)
    - Conditioning-frame point capture (first frame points reused for all inferences)
    - Point prompts formatted as Molmo html-v2 <points> tags
    - RBY1 observation format (obs["qpos"] with fallback to obs["robot_state"]["qpos"])
    """

    def __init__(self, config: MlSpacesExpConfig, task_type: str):
        pc = config.policy_config
        self.cameras_to_warp: list[str] = getattr(pc, "cameras_to_warp", [])
        self.use_point_prompts: bool = getattr(pc, "use_point_prompts", False)
        self.point_prompt_camera: str = getattr(pc, "point_prompt_camera", "head_camera")
        self.max_conditioning_points: int = getattr(pc, "max_conditioning_points", 1)
        self.clamp_gripper: bool = getattr(pc, "clamp_gripper", True)
        self.gripper_threshold: float = getattr(pc, "gripper_threshold", 5.0)
        self._conditioning_points: dict | None = None
        self._logged_obs_keys: bool = False

        super().__init__(config, task_type)

    def reset(self):
        super().reset()
        self._conditioning_points = None

    def get_state(self):
        state = super().get_state()
        return MolmoBotRBY1PolicyState(
            action_buffer=state.action_buffer,
            buffer_index=state.buffer_index,
            step_count=state.step_count,
            conditioning_points=self._conditioning_points,
        )

    def set_state(self, state: MolmoBotRBY1PolicyState):
        super().set_state(state)
        self._conditioning_points = state.conditioning_points

    def _populate_action_buffer(self, observation) -> None:
        """Override to handle RBY1 obs format, fisheye warping, and point prompts."""
        obs = observation[0] if isinstance(observation, list) else observation

        if not self._logged_obs_keys:
            logger.info(f"Observation keys: {list(obs.keys())}")
            if "qpos" in obs:
                logger.info(f"  qpos keys: {list(obs['qpos'].keys())}")
            self._logged_obs_keys = True

        # --- Extract images in camera order ---
        images = []
        for cam_name in self.camera_names:
            if cam_name not in obs:
                raise KeyError(
                    f"Camera '{cam_name}' not in observation. "
                    f"Available: {list(obs.keys())}"
                )
            images.append(obs[cam_name])

        # --- Fisheye warping (match training augmentation) ---
        if self.cameras_to_warp:
            from olmo.data.image_warping_utils import apply_fisheye_warping

            for i, cam_name in enumerate(self.camera_names):
                if cam_name in self.cameras_to_warp:
                    images[i] = apply_fisheye_warping(images[i])

        # --- Capture conditioning-frame points from first observation ---
        if self.use_point_prompts and self._conditioning_points is None:
            if "object_image_points" in obs:
                self._conditioning_points = obs["object_image_points"]
                logger.info("Captured conditioning points from first observation")

        # --- Extract qpos state (RBY1 format: obs["qpos"], fallback to obs["robot_state"]["qpos"]) ---
        qpos = obs.get("qpos")
        if qpos is None:
            robot_state = obs.get("robot_state", {})
            qpos = robot_state.get("qpos", {})

        qpos_parts = []
        for group_name in self.action_move_group_names:
            part = np.asarray(qpos[group_name], dtype=np.float32)
            expected_dim = self.action_spec[group_name]
            if part.shape[0] > expected_dim:
                part = part[:expected_dim]
            qpos_parts.append(part)
        state = np.concatenate(qpos_parts).astype(np.float32)

        # --- Task description with optional point prompts ---
        if "task" in obs:
            goal = obs["task"]
        else:
            goal = self.task.get_task_description()

        if self.use_point_prompts and self._conditioning_points is not None:
            goal = self._format_point_prompt(goal, self._conditioning_points)

        # --- Call inference agent ---
        pred_actions = self.agent.get_action_chunk(
            images=images,
            task_description=goal,
            state=state,
            generator=self._action_generator,
        )

        # --- Convert to list of action dicts ---
        self.action_buffer = []
        for t in range(pred_actions.shape[0]):
            action = {}
            start_idx = 0
            for group_name in self.action_move_group_names:
                dim = self.action_spec[group_name]
                selected_action = pred_actions[t, start_idx : start_idx + dim]
                if "gripper" in group_name and self.clamp_gripper:
                    action[group_name] = np.where(
                        selected_action >= self.gripper_threshold, 100.0, -100.0
                    ).astype(selected_action.dtype)
                else:
                    action[group_name] = selected_action
                start_idx += dim
            self.action_buffer.append(action)

        self.buffer_index = 0

    def _format_point_prompt(self, goal: str, object_image_points: dict) -> str:
        """Append Molmo html-v2 <points> tag to goal string.

        Coordinates are 0-1000 integers (Molmo html-v2 scale).
        If point_prompt_camera is in cameras_to_warp, coordinates are warped
        through the same fisheye pipeline.
        """
        cam = self.point_prompt_camera

        for obj_name, cam_dict in object_image_points.items():
            if "gripper" in obj_name:
                continue
            pts_data = cam_dict.get(cam)
            if pts_data is None:
                continue

            pts = pts_data["points"]
            num_pts = int(pts_data["num_points"][0])
            if num_pts == 0:
                continue

            valid_pts = pts[:min(num_pts, self.max_conditioning_points)].copy()  # (N, 2) normalized 0-1

            # Warp point coordinates if this camera is fisheye-warped
            if cam in self.cameras_to_warp:
                try:
                    from olmo.data.image_warping_utils import warp_point_coordinates

                    img_w, img_h = self.config.camera_config.img_resolution
                    valid_pts = warp_point_coordinates(
                        valid_pts, orig_h=img_h, orig_w=img_w
                    )
                except Exception as e:
                    logger.warning(f"Point coordinate warping failed: {e}")

            if len(valid_pts) == 0:
                continue

            # Format as Molmo html-v2: <points coords="FRAME_ID PID X Y PID X Y ...">
            coords_parts = []
            for i, pt in enumerate(valid_pts):
                x = int(np.clip(pt[0], 0.0, 1.0) * 1000)
                y = int(np.clip(pt[1], 0.0, 1.0) * 1000)
                coords_parts.append(f"{i + 1} {x:03d} {y:03d}")

            coords_str = "1 " + " ".join(coords_parts)
            return f'{goal} <points coords="{coords_str}">{obj_name}</points>'

        return goal


class MolmoBotRBY1PolicyConfig(SynthVLARBY1PolicyConfig):
    """Base MolmoBot policy config for RBY1 with fisheye warping + point prompts.

    Extends SynthVLARBY1PolicyConfig with:
    - Camera order matching training preset RBY1_full_with_head_gopro
    - Fisheye warping on head_camera
    - Point prompt support
    """

    # Camera order must match training preset RBY1_full_with_head_gopro
    camera_names: list[str] = ["wrist_camera_r", "head_camera", "wrist_camera_l"]

    # Gripper clamping: >= threshold → +100 (close), < threshold → -100 (open)
    clamp_gripper: bool = True
    gripper_threshold: float = 5.0

    # MolmoBot-specific features
    cameras_to_warp: list[str] = ["head_camera"]
    use_point_prompts: bool = True
    point_prompt_camera: str = "head_camera"
    max_conditioning_points: int = 10  # max points per object in prompt

    def model_post_init(self, __context) -> None:
        if self.policy_cls is None:
            from olmo.eval.configure_molmo_spaces import MolmoBotRBY1DoorOpeningPolicy

            object.__setattr__(self, "policy_cls", MolmoBotRBY1DoorOpeningPolicy)


class MolmoBotRBY1EvalConfig(SynthVLARBY1EvalConfig):
    """Base MolmoBot eval config for RBY1.

    Extends SynthVLARBY1EvalConfig with:
    - MolmoBot policy config (with point prompts + fisheye warping)
    - RBY1GoProD455CameraSystem (1024x576, matching training data)
    """

    policy_config: MolmoBotRBY1PolicyConfig = MolmoBotRBY1PolicyConfig()
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()


# Backwards compat aliases
MolmoBotRBY1DoorEvalConfig = MolmoBotRBY1EvalConfig
MolmoBotRBY1DoorPolicyConfig = MolmoBotRBY1PolicyConfig


# ── MolmoBot RBY1 Multitask (Door+Open, Pick+PnP) ──────────────────────


@dataclass
class MolmoBotRBY1MultitaskPolicyState(MolmoBotRBY1PolicyState):
    conditioning_image: np.ndarray | None = None


class MolmoBotRBY1MultitaskPolicy(MolmoBotRBY1DoorOpeningPolicy):
    """MolmoBot policy for RBY1 multitask with torso + optional conditioning image.

    Extends MolmoBotRBY1DoorOpeningPolicy with:
    - state_spec / state_indices for torso qpos extraction (3D state from 6D qpos)
    - Optional conditioning image (first frame head_camera appended as 4th image)
    """

    def __init__(self, config: MlSpacesExpConfig, task_type: str):
        pc = config.policy_config
        self.state_spec: dict[str, int] = getattr(pc, "state_spec", {})
        self.state_indices: dict[str, list[int]] = getattr(pc, "state_indices", {})
        self.use_conditioning_image: bool = getattr(pc, "use_conditioning_image", False)
        self.gripper_output_mode: str = os.environ.get(
            "RBY1_GRIPPER_OUTPUT_MODE",
            getattr(pc, "gripper_output_mode", "raw"),
        )
        self.gripper_binary_threshold: float = float(
            os.environ.get(
                "RBY1_GRIPPER_BINARY_THRESHOLD",
                getattr(pc, "gripper_binary_threshold", 0.0),
            )
        )
        self.gripper_open_position: float = float(
            os.environ.get(
                "RBY1_GRIPPER_OPEN_POSITION",
                getattr(pc, "gripper_open_position", -0.05),
            )
        )
        self.gripper_closed_position: float = float(
            os.environ.get(
                "RBY1_GRIPPER_CLOSED_POSITION",
                getattr(pc, "gripper_closed_position", 0.0),
            )
        )
        self._conditioning_image: np.ndarray | None = None
        self._debug_chunks_logged: int = 0

        super().__init__(config, task_type)

    def reset(self):
        super().reset()
        self._conditioning_image = None
        self._debug_chunks_logged = 0

    def _convert_gripper_action(self, selected_action: np.ndarray) -> np.ndarray:
        if self.gripper_output_mode == "binary_rby1":
            return np.where(
                selected_action >= self.gripper_binary_threshold,
                self.gripper_closed_position,
                self.gripper_open_position,
            ).astype(selected_action.dtype)
        if self.gripper_output_mode == "binary_rby1_inverted":
            return np.where(
                selected_action >= self.gripper_binary_threshold,
                self.gripper_open_position,
                self.gripper_closed_position,
            ).astype(selected_action.dtype)
        return selected_action

    def get_state(self):
        state = super().get_state()
        return MolmoBotRBY1MultitaskPolicyState(
            action_buffer=state.action_buffer,
            buffer_index=state.buffer_index,
            step_count=state.step_count,
            conditioning_points=state.conditioning_points,
            conditioning_image=self._conditioning_image,
        )

    def set_state(self, state: MolmoBotRBY1MultitaskPolicyState):
        super().set_state(state)
        self._conditioning_image = state.conditioning_image

    def _populate_action_buffer(self, observation) -> None:
        """Override to handle torso state extraction + conditioning image."""
        obs = observation[0] if isinstance(observation, list) else observation

        if not self._logged_obs_keys:
            logger.info(f"Observation keys: {list(obs.keys())}")
            if "qpos" in obs:
                logger.info(f"  qpos keys: {list(obs['qpos'].keys())}")
            self._logged_obs_keys = True

        # --- Extract images in camera order ---
        images = []
        for cam_name in self.camera_names:
            if cam_name not in obs:
                raise KeyError(
                    f"Camera '{cam_name}' not in observation. "
                    f"Available: {list(obs.keys())}"
                )
            images.append(obs[cam_name])

        # --- Fisheye warping (match training augmentation) ---
        if self.cameras_to_warp:
            from olmo.data.image_warping_utils import apply_fisheye_warping

            for i, cam_name in enumerate(self.camera_names):
                if cam_name in self.cameras_to_warp:
                    images[i] = apply_fisheye_warping(images[i])

        # --- Capture conditioning image from first observation (post-warp) ---
        if self.use_conditioning_image and self._conditioning_image is None:
            cam_idx = self.camera_names.index(self.point_prompt_camera)
            self._conditioning_image = images[cam_idx].copy()
            logger.info("Captured conditioning image from first observation")

        # --- Append conditioning image as 4th image ---
        if self.use_conditioning_image and self._conditioning_image is not None:
            images.append(self._conditioning_image)

        # --- Capture conditioning-frame points from first observation ---
        if self.use_point_prompts and self._conditioning_points is None:
            if "object_image_points" in obs:
                self._conditioning_points = obs["object_image_points"]
                logger.info("Captured conditioning points from first observation")

        # --- Extract qpos state (with state_spec + state_indices for torso) ---
        qpos = obs.get("qpos")
        if qpos is None:
            robot_state = obs.get("robot_state", {})
            qpos = robot_state.get("qpos", {})

        qpos_parts = []
        for group_name in self.action_move_group_names:
            part = np.asarray(qpos[group_name], dtype=np.float32)

            if group_name in self.state_indices:
                # Select specific indices (e.g., torso [1,2,3] from 6D)
                part = part[self.state_indices[group_name]]
            else:
                # Take first N dims based on state_spec (falls back to action_spec)
                expected_dim = self.state_spec.get(group_name, self.action_spec[group_name])
                if part.shape[0] > expected_dim:
                    part = part[:expected_dim]

            qpos_parts.append(part)
        state = np.concatenate(qpos_parts).astype(np.float32)

        # --- Task description with optional point prompts ---
        if "task" in obs:
            goal = obs["task"]
        else:
            goal = self.task.get_task_description()

        if self.use_point_prompts and self._conditioning_points is not None:
            goal = self._format_point_prompt(goal, self._conditioning_points)

        # --- Call inference agent ---
        pred_actions = self.agent.get_action_chunk(
            images=images,
            task_description=goal,
            state=state,
            generator=self._action_generator,
        )

        if self._debug_chunks_logged < 3:
            image_shapes = [getattr(image, "shape", None) for image in images]
            logger.info(
                "RBY1 multitask debug chunk %d: goal=%r state_shape=%s "
                "state_min=%.4f state_max=%.4f image_shapes=%s pred_shape=%s",
                self._debug_chunks_logged,
                goal,
                state.shape,
                float(np.nanmin(state)),
                float(np.nanmax(state)),
                image_shapes,
                pred_actions.shape,
            )

            start_idx = 0
            for group_name in self.action_move_group_names:
                dim = self.action_spec[group_name]
                chunk = pred_actions[:, start_idx : start_idx + dim]
                if "gripper" in group_name:
                    converted = self._convert_gripper_action(chunk)
                    logger.info(
                        "RBY1 multitask debug chunk %d: %s raw min=%.4f max=%.4f "
                        "mean=%.4f first_values=%s converted_min=%.4f converted_max=%.4f "
                        "mode=%s binary_threshold=%.4f clamp_gripper=%s threshold=%.4f",
                        self._debug_chunks_logged,
                        group_name,
                        float(np.nanmin(chunk)),
                        float(np.nanmax(chunk)),
                        float(np.nanmean(chunk)),
                        np.array2string(chunk[: min(8, len(chunk))].reshape(-1), precision=3),
                        float(np.nanmin(converted)),
                        float(np.nanmax(converted)),
                        self.gripper_output_mode,
                        self.gripper_binary_threshold,
                        self.clamp_gripper,
                        self.gripper_threshold,
                    )
                start_idx += dim

            self._debug_chunks_logged += 1

        # --- Convert to list of action dicts ---
        self.action_buffer = []
        for t in range(pred_actions.shape[0]):
            action = {}
            start_idx = 0
            for group_name in self.action_move_group_names:
                dim = self.action_spec[group_name]
                selected_action = pred_actions[t, start_idx : start_idx + dim]
                if "gripper" in group_name and self.clamp_gripper:
                    action[group_name] = np.where(
                        selected_action >= self.gripper_threshold, 100.0, -100.0
                    ).astype(selected_action.dtype)
                elif "gripper" in group_name:
                    action[group_name] = self._convert_gripper_action(selected_action)
                else:
                    action[group_name] = selected_action
                start_idx += dim
            self.action_buffer.append(action)

        self.buffer_index = 0


# ── Multitask Policy Configs ─────────────────────────────────────────────


class MolmoBotRBY1DoorPlusOpenPolicyConfig(MolmoBotRBY1PolicyConfig):
    """Policy config for MolmoBot RBY1 door+open with torso, point prompts, conditioning image."""

    action_move_group_names: list[str] = [
        "base", "left_arm", "left_gripper", "right_arm", "right_gripper", "torso",
    ]
    action_spec: dict[str, int] = {
        "base": 3, "left_arm": 7, "left_gripper": 1,
        "right_arm": 7, "right_gripper": 1, "torso": 1,
    }
    action_keys: dict[str, str] = {
        "base": "joint_pos_rel", "left_arm": "joint_pos_rel",
        "left_gripper": "joint_pos", "right_arm": "joint_pos_rel",
        "right_gripper": "joint_pos", "torso": "joint_pos",
    }
    state_spec: dict[str, int] = {
        "base": 3, "left_arm": 7, "left_gripper": 1,
        "right_arm": 7, "right_gripper": 1, "torso": 3,
    }
    state_indices: dict[str, list[int]] = {"torso": [1, 2, 3]}

    use_point_prompts: bool = True
    use_conditioning_image: bool = True
    max_conditioning_points: int = 1  # training uses max_points_in_conditioning_frame=1

    def model_post_init(self, __context) -> None:
        if self.policy_cls is None:
            from olmo.eval.configure_molmo_spaces import MolmoBotRBY1MultitaskPolicy

            object.__setattr__(self, "policy_cls", MolmoBotRBY1MultitaskPolicy)


class MolmoBotRBY1PickPnPPolicyConfig(MolmoBotRBY1PolicyConfig):
    """Policy config for MolmoBot RBY1 pick+pnp with torso, no points, no conditioning."""

    clamp_gripper: bool = False  # Disable gripper clamping for pick/pnp
    gripper_output_mode: str = "raw"
    gripper_binary_threshold: float = 0.0
    gripper_open_position: float = -0.05
    gripper_closed_position: float = 0.0

    action_move_group_names: list[str] = [
        "base", "left_arm", "left_gripper", "right_arm", "right_gripper", "torso",
    ]
    action_spec: dict[str, int] = {
        "base": 3, "left_arm": 7, "left_gripper": 1,
        "right_arm": 7, "right_gripper": 1, "torso": 1,
    }
    action_keys: dict[str, str] = {
        "base": "joint_pos_rel", "left_arm": "joint_pos_rel",
        "left_gripper": "joint_pos", "right_arm": "joint_pos_rel",
        "right_gripper": "joint_pos", "torso": "joint_pos",
    }
    state_spec: dict[str, int] = {
        "base": 3, "left_arm": 7, "left_gripper": 1,
        "right_arm": 7, "right_gripper": 1, "torso": 3,
    }
    state_indices: dict[str, list[int]] = {"torso": [1, 2, 3]}

    use_point_prompts: bool = False
    use_conditioning_image: bool = False

    def model_post_init(self, __context) -> None:
        if self.policy_cls is None:
            from olmo.eval.configure_molmo_spaces import MolmoBotRBY1MultitaskPolicy

            object.__setattr__(self, "policy_cls", MolmoBotRBY1MultitaskPolicy)


# ── Multitask Eval Configs ───────────────────────────────────────────────


class MolmoBotRBY1DoorPlusOpenEvalConfig(MolmoBotRBY1EvalConfig):
    """Eval config for MolmoBot RBY1 door+open tasks."""

    policy_config: MolmoBotRBY1DoorPlusOpenPolicyConfig = MolmoBotRBY1DoorPlusOpenPolicyConfig()
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Model outputs 1D torso action → use "height" mode (scalar → 6D joint mapping)
        self.robot_config.command_mode["torso"] = "height"


class MolmoBotRBY1PickPnPEvalConfig(MolmoBotRBY1EvalConfig):
    """Eval config for MolmoBot RBY1 pick+pnp tasks."""

    policy_config: MolmoBotRBY1PickPnPPolicyConfig = MolmoBotRBY1PickPnPPolicyConfig()
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()
    task_horizon: int = 400

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Model outputs 1D torso action → use "height" mode (scalar → 6D joint mapping)
        self.robot_config.command_mode["torso"] = "height"


class RBY1RecordedTrajectoryPolicyConfig(BasePolicyConfig):
    """Policy config for replaying recorded MolmoBot-data RBY1 action traces."""

    policy_type: str = "recorded_trajectory"
    policy_cls: type = None
    trajectory_path: str | None = None
    trajectory_key: str = "traj_0"
    action_dataset: str = "commanded_action"
    action_start_index: int = 1  # frame 0 is padding in saved MolmoSpaces trajectories
    action_move_group_names: list[str] = [
        "base",
        "left_arm",
        "left_gripper",
        "right_arm",
        "right_gripper",
        "torso",
    ]
    action_keys: dict[str, str] = {
        "base": "joint_pos",
        "left_arm": "joint_pos",
        "left_gripper": "joint_pos",
        "right_arm": "joint_pos",
        "right_gripper": "joint_pos",
        "torso": "joint_pos",
    }
    camera_names: list[str] = ["wrist_camera_r", "head_camera", "wrist_camera_l"]

    def model_post_init(self, __context) -> None:
        if self.policy_cls is None:
            from olmo.eval.configure_molmo_spaces import RBY1RecordedTrajectoryPolicy

            object.__setattr__(self, "policy_cls", RBY1RecordedTrajectoryPolicy)


class RBY1RecordedTrajectoryPolicy(BasePolicy):
    """Replay recorded per-step actions from a MolmoSpaces trajectory H5 file."""

    def __init__(self, config: MlSpacesExpConfig, task) -> None:
        super().__init__(config, task)
        pc = config.policy_config
        self.trajectory_path = os.environ.get(
            "RBY1_REPLAY_TRAJECTORY_PATH",
            getattr(pc, "trajectory_path", None) or "",
        )
        self.trajectory_key = os.environ.get(
            "RBY1_REPLAY_TRAJECTORY_KEY",
            getattr(pc, "trajectory_key", "traj_0"),
        )
        self.action_dataset = os.environ.get(
            "RBY1_REPLAY_ACTION_DATASET",
            getattr(pc, "action_dataset", "commanded_action"),
        )
        self.action_start_index = int(
            os.environ.get(
                "RBY1_REPLAY_ACTION_START_INDEX",
                getattr(pc, "action_start_index", 1),
            )
        )
        self.action_move_group_names = list(getattr(pc, "action_move_group_names", []))
        self.actions = self._load_actions()
        self.step_index = 0

    @staticmethod
    def _decode_json_row(row: np.ndarray) -> dict:
        import json

        raw = row.tobytes().decode("utf-8").rstrip("\x00")
        return json.loads(raw) if raw else {}

    def _load_actions(self) -> list[dict[str, np.ndarray]]:
        if not self.trajectory_path:
            raise ValueError(
                "Recorded replay requires RBY1_REPLAY_TRAJECTORY_PATH or "
                "policy_config.trajectory_path"
            )

        import h5py

        actions: list[dict[str, np.ndarray]] = []
        with h5py.File(self.trajectory_path, "r") as f:
            action_group = f[self.trajectory_key]["actions"]
            if self.action_dataset not in action_group:
                raise KeyError(
                    f"Missing actions/{self.action_dataset} in {self.trajectory_path}; "
                    f"available={list(action_group.keys())}"
                )
            dataset = action_group[self.action_dataset]
            for frame_idx in range(self.action_start_index, dataset.shape[0]):
                frame = self._decode_json_row(dataset[frame_idx])
                if not frame:
                    continue
                action: dict[str, np.ndarray] = {}
                for move_group in self.action_move_group_names:
                    if move_group not in frame:
                        continue
                    action[move_group] = np.asarray(frame[move_group], dtype=np.float32)
                actions.append(action)

        if not actions:
            raise ValueError(
                f"No replayable actions decoded from {self.trajectory_path}:{self.trajectory_key}"
            )
        logger.info(
            "Loaded %d recorded actions from %s:%s",
            len(actions),
            self.trajectory_path,
            self.trajectory_key,
        )
        return actions

    def reset(self):
        self.step_index = 0

    def get_action(self, observation):
        if self.step_index >= len(self.actions):
            return {"done": True}
        action = self.actions[self.step_index]
        self.step_index += 1
        return action

    def get_info(self) -> dict:
        return {
            "policy_type": "recorded_trajectory",
            "trajectory_path": self.trajectory_path,
            "trajectory_key": self.trajectory_key,
            "action_dataset": self.action_dataset,
            "step_index": self.step_index,
            "num_actions": len(self.actions),
        }

    def get_phase(self) -> str:
        return "replay"

    def get_all_phases(self) -> dict[str, int]:
        return {"replay": 0}


class RBY1RecordedTrajectoryEvalConfig(SynthVLARBY1EvalConfig):
    """Eval config that replays absolute actions recorded in MolmoBot-data."""

    policy_config: RBY1RecordedTrajectoryPolicyConfig = RBY1RecordedTrajectoryPolicyConfig()
    robot_config: RBY1MConfig = RBY1MConfig()
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()
    policy_dt_ms: float = 100.0
    ctrl_dt_ms: float = 20.0
    sim_dt_ms: float = 4.0
    task_horizon: int = 400

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Replay uses the absolute commands saved under actions/commanded_action.
        self.robot_config.action_noise_config.enabled = False
        self.robot_config.command_mode["base"] = "holo_joint_planar_position"
        self.robot_config.command_mode["arm"] = "joint_position"
        self.robot_config.command_mode["gripper"] = "joint_position"
        self.robot_config.command_mode["head"] = None
        self.robot_config.command_mode["torso"] = "joint_position"
