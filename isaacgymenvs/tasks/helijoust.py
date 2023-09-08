import numpy as np
import os
import torch

from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask

from isaacgym import gymutil, gymtorch, gymapi

class HeliJoust(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # Observations:
        # 0:13 - root state
        # 14:21 - ee pose
        # 22:26 - ee vel
        # reserved - opponent
        self.cfg["env"]["numObservations"] = 26

        # Actions:
        # 0:3 - xyz force vector for the rotor
        # 4:10 - ee pose set point
        self.cfg["env"]["numActions"] = 9

        self.num_agents = self.cfg["env"]["numAgents"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dofs_per_agent = 9
        bodies_per_agent = 7 #?

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, self.num_agents, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, self.num_agents, dofs_per_agent, 2)

        self.root_states = vec_root_tensor
        self.root_positions = self.root_states[:, :, 0:3]
        self.target_root_positions = torch.zeros((self.num_envs, self.num_agents, 3), device=self.device, dtype=torch.float32)
        self.target_root_positions[:, :, 2] = 1
        self.root_quats = self.root_states[:, :, 3:7]
        self.root_linvels = self.root_states[:, :, 7:10]
        self.root_angvels = self.root_states[:, :, 10:13]

        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_velocities = vec_dof_tensor[..., 1]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()

        self.thrust_lower_limit = 0
        self.thrust_upper_limit = 2000
        self.thrust_lateral_component = 0.2

        # control tensors
        self.thrusts = torch.zeros((self.num_envs, self.num_agents, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, self.num_agents * bodies_per_agent, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_control = torch.zeros((self.num_envs, self.num_agents, dofs_per_agent - 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.all_actor_indices = torch.arange(self.num_envs * self.num_agents, dtype=torch.int32, device=self.device).reshape((self.num_envs, self.num_agents))

        if self.viewer:
            cam_pos = gymapi.Vec3(2.25, 2.25, 3.0)
            cam_target = gymapi.Vec3(3.5, 4.0, 1.9)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # need rigid body states for visualizing thrusts
            #self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            #self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor).view(self.num_envs, bodies_per_env, 13)
            #self.rb_positions = self.rb_states[..., 0:3]
            #self.rb_quats = self.rb_states[..., 3:7]

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z

        # gravity
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -10

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        #self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "./"
        asset_file = "ingenuity.xml"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 4 * math.pi
        asset_options.slices_per_cylinder = 40
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        asset_options.fix_base_link = True
        marker_asset = self.gym.create_sphere(self.sim, 0.1, asset_options)

        default_pose = gymapi.Transform()
        default_pose.p.z = 1.0

        self.envs = []
        self.actor_handles = []
        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor_handle = self.gym.create_actor(env, asset, default_pose, "ingenuity", i, 1, 1)

            dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            dof_props['stiffness'].fill(0)
            dof_props['damping'].fill(0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)

            marker_handle = self.gym.create_actor(env, marker_asset, default_pose, "marker", i, 1, 1)
            self.gym.set_rigid_body_color(env, marker_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))

            self.actor_handles.append(actor_handle)
            self.envs.append(env)

        if self.debug_viz:
            # need env offsets for the rotors
            self.rotor_env_offsets = torch.zeros((self.num_envs, 2, 3), device=self.device)
            for i in range(self.num_envs):
                env_origin = self.gym.get_env_origin(self.envs[i])
                self.rotor_env_offsets[i, ..., 0] = env_origin.x
                self.rotor_env_offsets[i, ..., 1] = env_origin.y
                self.rotor_env_offsets[i, ..., 2] = env_origin.z