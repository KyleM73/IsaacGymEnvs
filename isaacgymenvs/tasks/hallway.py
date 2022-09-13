import numpy as np
import os
import torch

import EnvCreator

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

def apply_tensor_value(tensor,idx,value,num=None):
    """
    applies 'value' to 'tensor' at position 'idx' with step size 'num' 
    """
    if num is None and len(tensor.size()) == 2:
        num = tensor.size()[1]
    if num == 0:
        tensor[idx, ...] = value
    else:
        tensor[idx::num, ...] = value
    return tensor

class Hallway(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.pose_cost_scale = self.cfg["env"]["poseCost"]
        self.ang_cost_scale = self.cfg["env"]["angCost"]
        self.velocity_cost_scale = self.cfg["env"]["velocityCost"]
        self.ang_velocity_cost_scale = self.cfg["env"]["angularVelocityCost"]
        self.contact_cost = self.cfg["env"]["contactCost"]
        self.contact_lim = self.cfg["env"]["contactLimit"]
        self.death_cost = self.cfg["env"]["deathCost"]
        
        self.force_scale = self.cfg["env"]["forceScale"]
        self.torque_scale = self.cfg["env"]["torqueScale"]
        self.max_range = self.cfg["env"]["maxRange"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 10
        self.cfg["env"]["numActions"] = 3

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        if self.viewer != None:
            cam_pos = gymapi.Vec3(*self.cfg["viewer"]["pos"])
            cam_target = gymapi.Vec3(*self.cfg["viewer"]["target"])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) #[num_bodies,13]
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)

        self._fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
        self.fsdata = gymtorch.wrap_tensor(self._fsdata)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        #print(self.root_tensor.size()) #[4,13] with 2 envs
        #print(self.net_cf.size()) #[num_envs*num_bodies,3]

        self.initial_root_state = self.root_tensor.clone()
        self.initial_root_state[:, 7:13] = 0  #set velocities to zero

        target = torch.tensor(self.cfg["env"]["target"]).to(torch.float)
        self.targets = to_torch(self.cfg["env"]["target"], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch((target/torch.norm(target)).tolist(), device=self.device).repeat((self.num_envs, 1))
        ang = np.arccos(np.dot((target/torch.norm(target))[:2],torch.tensor([1,0]))).tolist()
        self.target_angs = to_torch(ang, device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        self.goal_vel = self.cfg["env"]["velocityGoal"] * torch.ones(self.num_envs,1,device=self.device)
        self.goal_ang_vel = self.cfg["env"]["angVelocityGoal"] * torch.ones(self.num_envs,1,device=self.device)
        self.goal_radius = self.cfg["env"]["goalRadius"]
        self.goal_bonus = self.cfg["env"]["goalBonus"]

        #self.gym.refresh_dof_state_tensor(self.sim)
        #self.gym.refresh_dof_force_tensor(self.sim)

        #print(self.root_tensor)

    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        #    - call super().create_sim with device args (see docstring)
        #    - create ground plane
        #    - set up environments
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -1, 0)
        upper = gymapi.Vec3(spacing, 20, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/bumpybot.urdf"

        occ_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        occ_file = "occupancy/hall.png"

        walls_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)
            occ_file = self.cfg["env"]["asset"].get("occupancyFileName", occ_file)
        
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        occ_path = os.path.join(occ_root, occ_file)
        occ_root = os.path.dirname(occ_path)

        env_c = EnvCreator.envCreator(occ_path)
        walls_path = env_c.get_urdf(occ_root)
        walls_root = os.path.dirname(walls_path)
        walls_file = os.path.basename(walls_path)

        start = self.cfg["env"]["path"]["start"]
        target = self.cfg["env"]["path"]["target"]
        filter_dist = self.cfg["env"]["path"]["filterDist"]
        self.path = env_c.get_path(start,target,filter_dist)

        asset_options = gymapi.AssetOptions()
        asset_options.collapse_fixed_joints = True
        #asset_options.fix_base_link = True
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        asset_bodies = self.gym.get_asset_rigid_body_count(asset)
        
        base_idx = self.gym.find_asset_rigid_body_index(asset, "base")
        sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.enable_forward_dynamics_forces = False
        sensor_props.enable_constraint_solver_forces = True
        sensor_props.use_world_frame = False
        sensor_idx = self.gym.create_asset_force_sensor(asset, base_idx, sensor_pose, sensor_props)

        walls_options = gymapi.AssetOptions()
        walls_options.collapse_fixed_joints = True
        walls_options.fix_base_link = True
        walls = self.gym.load_asset(self.sim, walls_root, walls_file, walls_options)
        walls_bodies = self.gym.get_asset_rigid_body_count(walls)

        self.num_bodies = asset_bodies + walls_bodies

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.635/2, self.up_axis_idx))
        start_pose.r = gymapi.Quat.from_euler_zyx(0, 0, np.pi/2) #face y axis

        walls_pose = gymapi.Transform()
        walls_pose.p = gymapi.Vec3(0,0,0)
        walls_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        if self.cfg["sim"]["test"]:
            path_path = env_c.path2urdf(self.path,occ_root)
            path_root = os.path.dirname(path_path)
            path_file = os.path.basename(path_path)

            path_options = gymapi.AssetOptions()
            path_options.collapse_fixed_joints = True
            path_options.fix_base_link = True
            path = self.gym.load_asset(self.sim, path_root, path_file, path_options)
            path_bodies = self.gym.get_asset_rigid_body_count(path)

            self.num_bodies += path_bodies

            path_pose = gymapi.Transform()
            path_pose.p = gymapi.Vec3(0,0,0)
            path_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        self.handles = []
        self.wall_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            handle = self.gym.create_actor(env, asset, start_pose, "bumpybot", i)
            walls_handle = self.gym.create_actor(env, walls, walls_pose, "walls", i) # take this out of loop and use  -1 as last arg so all actors can collide with same walls
    

            self.envs.append(env)
            self.handles.append(handle)
            self.wall_handles.append(walls_handle)

            props_shape = self.gym.get_actor_rigid_shape_properties(env, handle)
            props_shape[0].rolling_friction = 0.0
            props_shape[0].torsion_friction = 0.0
            props_shape[0].friction = 0.0
            props_shape[0].restitution = 0.0
            self.gym.set_actor_rigid_shape_properties(env, handle, props_shape)

            if self.cfg["sim"]["test"]:
                self.gym.create_actor(env, path, path_pose, "path", i)

    def _compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf = compute_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.actions_cost_scale,
            self.pose_cost_scale,
            self.ang_cost_scale,
            self.velocity_cost_scale,
            self.ang_velocity_cost_scale,
            self.max_episode_length,
            self.goal_radius,
            self.goal_bonus,
            self.fsdata,
            self.contact_cost,
            self.contact_lim,
            self.death_cost,
            self.max_range
            )

    def _compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.obs_buf[:] = compute_observations(
            self.root_tensor,
            self.targets,
            self.target_angs,
            self.goal_vel,
            self.goal_ang_vel,
            self.dt,
            self.actions,
            self.max_range,
            self.num_bodies
            )

    def reset_idx(self, env_ids):

        positions = torch.cat((
            torch_rand_float(-0.2, 0.2, (len(env_ids), 1), device=self.device),
            torch.zeros(len(env_ids),12,device=self.device)
            ),dim=-1)
        pose = torch.zeros_like(self.initial_root_state)
        pose[self.num_bodies*env_ids] = positions

        random_root = self.initial_root_state + pose

        env_ids_int32 = (self.num_bodies*env_ids).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(random_root),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        to_target = self.targets[env_ids] - random_root[env_ids, :3]
        to_target[:, self.up_axis_idx] = 0

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        # apply  forces
        # for testing:
        #actions = torch.zeros_like(actions)
        #actions[:,:] = torch.tensor([-0.05,1,0]).repeat((self.num_envs,1))
        
        self.actions = actions.to(self.device).clone()

        forces = torch.zeros((self.num_envs,self.num_bodies, 3), device="cuda:0", dtype=torch.float)
        forces[:, 0, :2] = self.force_scale * self.actions[:, :2]

        torques = torch.zeros((self.num_envs,self.num_bodies, 3), device="cuda:0", dtype=torch.float)
        torques[:, 0, 2] = self.torque_scale * self.actions[:, 2]

        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1

        #print(get_euler_xyz(self.obs_buf[:, 3:7]))
        #<cylinder length="0.635" radius="0.2794" />
        # <mass value="4.53592"/>

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self._compute_observations()
        self._compute_reward(self.actions)


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_reward(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    actions_cost_scale,
    pose_cost_scale,
    ang_cost_scale,
    velocity_cost_scale,
    ang_velocity_cost_scale,
    max_episode_length,
    goal_radius,
    goal_bonus,
    fsdata,
    contact_cost_scale,
    contact_lim,
    death_cost,
    max_range
    ):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, Tensor, float, float, float, float) -> Tuple[Tensor, Tensor]

    actions_cost = actions_cost_scale * torch.sum(actions ** 2, dim=-1)

    pose_cost = pose_cost_scale * torch.linalg.norm(max_range * (obs_buf[:, :2] - obs_buf[:, 2:4]),dim=1)

    vel_cost = velocity_cost_scale * torch.abs(obs_buf[:, 4])

    ang_cost = ang_cost_scale * torch.abs(obs_buf[:, 5])

    ang_vel_cost = ang_velocity_cost_scale * torch.abs(obs_buf[:, 6])

    contact_cost = contact_cost_scale * torch.linalg.norm(fsdata[:, :2],dim=-1)

    total_reward = - actions_cost - pose_cost - ang_cost - vel_cost - ang_vel_cost - contact_cost

    # terminal costs
    total_reward = torch.where(pose_cost < pose_cost_scale * goal_radius, total_reward + goal_bonus, total_reward)

    # adjust reward for dead agents
    total_reward = torch.where(contact_cost > contact_cost_scale * contact_lim, torch.ones_like(total_reward) * death_cost, total_reward)

    # reset agents
    conds = (contact_cost > contact_cost_scale * contact_lim) | (progress_buf >= max_episode_length - 1)
    reset = torch.where(conds, torch.ones_like(reset_buf), reset_buf) #last arg "reset"

    ## TODO
    # - add power usage terms
    # - timeout bootstrapping (see ETH parallel walking paper)

    return total_reward, reset

@torch.jit.script
def compute_observations(
    root_states,
    targets,
    target_angs,
    goal_vel,
    goal_ang_vel,
    dt,
    actions,
    max_range,
    num_bodies
    ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, int) -> Tensor

    # x y th vx vy vth gx gy gth ang2target gv fx fy fth

    position = root_states[::num_bodies, 0:3]
    rotation = root_states[::num_bodies, 3:7]
    velocity = root_states[::num_bodies, 7:10]
    ang_velocity = root_states[::num_bodies, 10:13]

    _,_,yaw = get_euler_xyz(rotation)
    heading = normalize_angle(yaw).unsqueeze(-1)
    heading_err = heading - target_angs

    ang_vel_error = torch.linalg.norm(ang_velocity[:,2],dim=-1) - goal_ang_vel #1

    # Normalize Observations
    position_norm = position[:,:2] / max_range
    targets_norm = targets[:,:2] / max_range

    velocity_norm = torch.linalg.norm(velocity[:,:2],dim=-1).view(-1,1) / (goal_vel + 1e-9) - 1
    velocity_norm = torch.clamp(velocity_norm,-1,2) #somewhat arbitrary bounds

    # obs_buf shapes: 2, 2, 1, 1, 1, num_acts(3)
    obs = torch.cat((position_norm,targets_norm,velocity_norm,
        heading_err,ang_vel_error,actions),dim=-1) #10

    return obs