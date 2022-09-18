from typing import Dict, Any, Tuple

import gym
from gym import spaces

import numpy as np
import os
import torch
import imageio

import EnvCreator

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.utils.waypoint_checker import *
from isaacgymenvs.utils.np_formatting import *
from isaacgymenvs.tasks.base.vec_task import VecTask

class HallwayCamera(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        set_np_formatting()
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

        self.img_h = self.cfg["image"]["height"]
        self.img_w = self.cfg["image"]["width"]
        self.img_d = self.cfg["image"]["depth"]
        if self.img_d == 1:
            self.img_type = gymapi.IMAGE_DEPTH
        else:
            self.img_type = gymapi.IMAGE_COLOR
            self.img_d = 3
        self.cam_max_range = self.cfg["image"]["range"]
        self.update_freq = self.cfg["image"]["updateFreq"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 8
        self.cfg["env"]["numActions"] = 3

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        self.img_obs_space = spaces.Box(low=0,high=1,shape=(self.img_w,self.img_h,self.img_d))
        self.update_obs_space = spaces.Box(low=0,high=1,shape=(self.num_envs,))
        self.vec_obs_space = self.obs_space
        self.obs_space = spaces.Dict(
            spaces={
            "vec" : self.vec_obs_space,
            "img" : self.img_obs_space,
            "update_img" : self.update_obs_space,
            })

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

        self.initial_root_state = self.root_tensor.clone()
        self.initial_root_state[:, 7:13] = 0  #set velocities to zero

        self.path.insert(0,tuple(self.cfg["env"]["path"]["start"])) #append start location to beginning of path
        self.path.insert(-1,tuple(self.cfg["env"]["path"]["target"])) #append target location to beginning of path
        self.path = torch.tensor([list(p) for p in self.path]).to(self.device) #[n_wayptsx2]
        
        self.targets = torch.ones(self.num_envs,dtype=torch.long).to(self.device) #target waypoint index in sef.path. size: [num_envs]

        self.dt = self.cfg["sim"]["dt"]
        self.goal_vel = self.cfg["env"]["velocityGoal"] * torch.ones(self.num_envs,1,device=self.device)
        self.goal_ang_vel = self.cfg["env"]["angVelocityGoal"] * torch.ones(self.num_envs,1,device=self.device)
        self.goal_radius = self.cfg["env"]["goalRadius"]
        self.goal_bonus = self.cfg["env"]["goalBonus"]

        self._get_images()
        self.init_img_tensor = self.img_tensor.clone()

        # self.epoch = 0
        # self.steps = 0
        # self.phase = 0

    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        #    - call super().create_sim with device args (see docstring)
        #    - create ground plane
        #    - set up environments
        self.graphics_device_id = self.device_id
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], 2*int(np.sqrt(self.num_envs)))

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
        walls_path = env_c.get_urdf_fast(occ_root)
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
        sensor_props.use_world_frame = True #false??
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

        camera_props = gymapi.CameraProperties()
        camera_props.width = self.img_w
        camera_props.height = self.img_h
        camera_props.use_collision_geometry = True
        camera_props.enable_tensors = True

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(0,0.25,0.3) # get real values from robot
        local_transform.r = gymapi.Quat.from_euler_zyx(0, 0, np.pi/2)

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
            path_pose.p = gymapi.Vec3(0,0,0) # waypoints above env so camera cant see them
            path_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        self.handles = []
        self.wall_handles = []
        self.camera_handles = []
        self.cam_tensors = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            handle = self.gym.create_actor(env, asset, start_pose, "bumpybot", i)
            walls_handle = self.gym.create_actor(env, walls, walls_pose, "walls", i) # take this out of loop and use  -1 as last arg so all actors can collide with same walls
            camera_handle = self.gym.create_camera_sensor(env, camera_props)

            self.gym.attach_camera_to_body(camera_handle, env, handle, local_transform, gymapi.FOLLOW_TRANSFORM) #gymapi.FOLLOW_TRANSFORM,gymapi.FOLLOW_POSITION doesnt rotate with robot

            self.envs.append(env)
            self.handles.append(handle)
            self.wall_handles.append(walls_handle)
            self.camera_handles.append(camera_handle)

            cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, self.img_type)
            torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
            self.cam_tensors.append(torch_cam_tensor)

            props_shape = self.gym.get_actor_rigid_shape_properties(env, handle)
            props_shape[0].rolling_friction = 0.0
            props_shape[0].torsion_friction = 0.0
            props_shape[0].friction = 0.0
            props_shape[0].restitution = 0.0
            self.gym.set_actor_rigid_shape_properties(env, handle, props_shape)

            if self.cfg["sim"]["test"]:
                self.gym.create_actor(env, path, path_pose, "path", i)

    def allocate_buffers(self):
        super().allocate_buffers()
        self.img_tensor = torch.zeros(self.num_envs,self.img_w,self.img_h,self.img_d,device=self.device)
        self.capture = torch.ones(self.num_envs,1,device=self.device)

    def _get_images(self):
        #img_dir = "tmp_imgs"
        #if not os.path.exists(img_dir):
        #    os.mkdir(img_dir)
        #self.capture = torch.where(self.progress_buf % self.update_freq == 0,1,0) #set camera update rate
        #if torch.any(self.capture):
        self.gym.fetch_results(self.sim,True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        #print(torch.Tensor(self.cam_tensors).size())
        #imgs = self.cam_tensors[self.capture] #.view(-1,self.img_w,self.img_h,self.img_d)
        #print(imgs.size())
        #assert False
        for i in range(self.num_envs):
            #if self.capture[i] > 0:
            img = self.cam_tensors[i].view(self.img_w,self.img_h,self.img_d)
            #self.prev_img_tensor[i,...] = self.img_tensor[i,...]
            self.img_tensor[i,...] = self._normalize_image(img)
            #fname = os.path.join(img_dir, "cam-%04d.png" % i)
            #imageio.imwrite(fname, 255*self._normalize_image(img).cpu().numpy())
        self.gym.end_access_image_tensors(self.sim)

    def _normalize_image(self,img):
        return normalize_image(img,self.cam_max_range)

    def _compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf, self.targets = compute_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
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
            self.path,
            self.targets
            )

    def _compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.obs_buf[:] = compute_observations(
            self.root_tensor,
            self.path,
            self.targets,
            self.goal_vel,
            self.goal_ang_vel,
            self.dt,
            self.actions,
            self.num_bodies
            )

    def reset_idx(self, env_ids):

        vels = torch.cat((
            torch.zeros(len(env_ids),7,device=self.device), #pose,quat
            torch_rand_float(-0.1, 0.1, (len(env_ids),2), device=self.device), #vx,vy
            torch.zeros(len(env_ids),4,device=self.device) #vy,v_ang
            ),dim=-1)
        pose = torch.zeros_like(self.initial_root_state)
        pose[self.num_bodies*env_ids] = vels

        random_root = self.initial_root_state + pose

        env_ids_int32 = (self.num_bodies*env_ids).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(random_root),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.img_tensor[env_ids] = self.init_img_tensor[env_ids]

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

        # if self.phase == 0:
        #     self.steps += 1
        #     if self.steps % 32 == 0:
        #         self.epoch += 1
        #     if self.epoch >= 2000:
        #         self.phase = 1

        #if self.phase or self.cfg["sim"]["test"]: self._get_images()
        self._get_images()
        self._compute_observations()
        self._compute_reward(self.actions)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        out = super().step(actions)
        vec_obs = out[0]["obs"] #self.obs_dict
        self.obs_dict["obs"] = {"vec":vec_obs,"img":self.img_tensor,"update_img":self.capture}
        return self.obs_dict,*out[1:]

    def reset(self):
        self.obs_dict = super().reset()
        self.obs_dict["obs"] = {"vec":torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device),"img":self.img_tensor,"update_img":self.capture}

        return self.obs_dict

    def reset_done(self):
        self.obs_dict,done_env_ids = super().reset_done()
        self.obs_dict["obs"] = {"vec":torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device),"img":self.img_tensor,"update_img":self.capture}

        return self.obs_dict, done_env_ids


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def normalize_image(img,cam_max_range):
    # [W,H,C]
    img[img < -cam_max_range] = -cam_max_range
    img = -torch.abs(img/(img.min() + 1e-4)) + 1
    return img

@torch.jit.script
def compute_reward(
    obs_buf,
    reset_buf,
    progress_buf,
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
    path,
    targets
    ):
    # type: (Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, Tensor, float, float, float, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

    actions = torch.linalg.norm(obs_buf[:, 5:],dim=-1)
    actions_reward = actions_cost_scale * 1.0 / (1.0 + actions * actions)

    pose = torch.linalg.norm(obs_buf[:, :2],dim=-1)
    pose_reward = pose_cost_scale * 1.0 / (1.0 + pose * pose)

    vel = torch.abs(obs_buf[:, 2])
    vel_reward = velocity_cost_scale * 1.0 / (1.0 + vel * vel)

    ang = torch.abs(obs_buf[:, 3])
    ang_reward = ang_cost_scale * 1.0 / (1.0 + ang * ang)

    ang_vel = torch.abs(obs_buf[:, 4])
    ang_vel_reward = ang_velocity_cost_scale * 1.0 / (1.0 + ang_vel * ang_vel)

    contact = torch.linalg.norm(fsdata[:, :2],dim=-1)
    contact_cost = contact_cost_scale * 1.0 / (1.0 + contact * contact)

    total_reward = actions_reward + pose_reward + vel_reward + ang_reward + ang_vel_reward + contact_cost

    # terminal costs
    total_reward = torch.where(pose <= goal_radius, total_reward + goal_bonus, total_reward)

    # adjust reward for dead agents
    total_reward = torch.where(contact > contact_lim, torch.ones_like(total_reward) * death_cost, total_reward)

    # update targets
    update_conds = (pose <= goal_radius) | (check_waypoints(obs_buf[:, :2]+path[targets],path,targets))
    targets = torch.where(update_conds,targets+1,targets)

    # reset agents
    reset_conds = (contact > contact_lim) | (progress_buf >= max_episode_length - 1) | (targets > len(path)-2)
    reset = torch.where(reset_conds, torch.ones_like(reset_buf), reset_buf) #last arg "reset"

    # reset targets
    targets = torch.where(reset_conds,1,targets)

    """
    print(pose_reward)
    print(vel_reward)
    print(ang_reward)
    print(ang_vel_reward)
    print(-contact_cost)
    print(total_reward)
    print()
    """

    ## TODO
    # - add power usage terms
    # - timeout bootstrapping (see ETH parallel walking paper)

    return total_reward, reset, targets

@torch.jit.script
def compute_observations(
    root_states,
    path,
    targets,
    goal_vel,
    goal_ang_vel,
    dt,
    actions,
    num_bodies
    ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int) -> Tensor

    # x y th vx vy vth gx gy gth ang2target gv fx fy fth

    position = root_states[::num_bodies, 0:3]
    rotation = root_states[::num_bodies, 3:7]
    velocity = root_states[::num_bodies, 7:10]
    ang_velocity = root_states[::num_bodies, 10:13]

    pose_error = position[:, :2] - path[targets]

    _,_,yaw = get_euler_xyz(rotation)
    heading = normalize_angle(yaw).unsqueeze(-1)

    dot_prod = torch.einsum('ij,ij->i',position[:, :2],path[targets])
    dot_prod /= torch.linalg.norm(position[:, :2]+1e-7,dim=-1)
    dot_prod /= torch.linalg.norm(path[targets]+1e-7,dim=-1)
    dot_prod = torch.clamp(dot_prod,-1+1e-7,1-1e-7)
    target_angs = torch.arccos(dot_prod).view(-1,1)

    heading_err = heading - target_angs

    vel_error = torch.linalg.norm(velocity[:,:2],dim=-1).view(-1,1) - goal_vel #[1]

    ang_vel_error = torch.linalg.norm(ang_velocity[:,2],dim=-1) - goal_ang_vel #[1]

    # obs_buf shapes: 2, 1, 1, 1, num_acts(3)
    obs = torch.cat((pose_error,vel_error,
        heading_err,ang_vel_error,actions),dim=-1) #8

    return obs