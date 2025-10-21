import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .walk_env_cfg import G1WalkEnvCfg

class G1WalkEnv(DirectRLEnv):
    cfg:G1WalkEnvCfg

    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.commands_scale = torch.tensor([1.0, 0.5, 1.0], device=self.device)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.actions = torch.zeros(self.num_envs, 23, device=self.device)
        self.previous_actions = torch.zeros(self.num_envs, 23, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        self.processed_actions = self.cfg.action_scale * self.actions + self.robot.data.joint_pos

    def _apply_action(self):
        self.robot.set_joint_position_target(self.processed_actions)

    def _get_observations(self):
        self.previous_actions = self.actions.clone()

        root_ang_vel = self.robot.data.root_ang_vel_b        # (num_envs, 3)
        base_orientation = self.robot.data.projected_gravity_b        # (num_envs, 3)
        joint_pos = self.robot.data.joint_pos - self.robot.data.default_joint_pos # (num_envs, 23)
        joint_vel = self.robot.data.joint_vel               # (num_envs, 23)
        previous_actions = self.previous_actions           # (num_envs, 23)

        root_ang_vel_noise = torch.rand_like(root_ang_vel) * 0.1
        base_orientation_noise = torch.rand_like(base_orientation)
        joint_pos_noise = torch.rand_like(joint_pos)
        joint_vel_noise = torch.rand_like(joint_vel)

        if self.cfg.training:
            root_ang_vel += root_ang_vel_noise
            base_orientation += base_orientation_noise
            joint_pos += joint_pos_noise
            joint_vel += joint_vel_noise

        obs = torch.cat([
            root_ang_vel,
            base_orientation,
            joint_pos,
            joint_vel,
            previous_actions,
            self.commands
        ], dim=-1)

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        return (
            1.0 * self._height_reward() +
            1.0 * self._xy_velocity_reward() +
            1.0 * self._yaw_angel_velocity_reward() +
            -1.0 * self._difference_to_default_reward()
        )

    def _height_reward(self) -> torch.Tensor:
        base_height = self.robot.data.root_state_w[:, 2]
        height_target = 0.6

        # Reward is 1.0 if height is at or above the target
        reward = torch.where(
            base_height >= height_target,
            torch.ones_like(base_height),
            torch.zeros_like(base_height)
        )
        return reward

    def _xy_velocity_reward(self) -> torch.Tensor:
        xy_velocity = self.robot.data.root_lin_vel_b[:, :2]
        target_xy_velocity = self.commands[:, :2]

        error = torch.sum(torch.square(target_xy_velocity - xy_velocity), dim=1)
        return torch.exp(-error / 0.25)
    
    def _yaw_angel_velocity_reward(self) -> torch.Tensor:
        yaw_angle_velocity = self.robot.data.root_ang_vel_b[:, 2]
        target_yaw_angle_velocity = self.commands[:, 2]

        error = torch.square(target_yaw_angle_velocity - yaw_angle_velocity)
        return torch.exp(-error / 0.25)
    
    def _z_velocity_reward(self) -> torch.Tensor:
        z_velocity = self.robot.data.root_lin_vel_b[:, 2]

        return torch.square(z_velocity)

    def _get_action_rate_reward(self) -> torch.Tensor:
        return torch.sum((self.actions - self.previous_actions) ** 2, dim=1)
    
    def _joint_velocity_penalty(self) -> torch.Tensor:
        return torch.norm(self.robot.data.joint_vel, dim=1)

    def _difference_to_default_reward(self) -> torch.Tensor:
        return torch.sum((self.robot.data.joint_pos - self.robot.data.default_joint_pos) ** 2, dim=1)
     
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_height = self.robot.data.root_state_w[:, 2]
        terminate = base_height < 0.5
        #terminate = base_height < 0
        return terminate, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        #self.commands[env_ids] = torch.zeros_like(self.commands[env_ids]).uniform_(-1.0, 1.0)
        #self.commands[env_ids] *= self.commands_scale

        self.commands[env_ids] = torch.zeros_like(self.commands[env_ids])
        
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        #default_root_state = self.robot.data.default_root_state[env_ids]
        #default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        #self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        #self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
