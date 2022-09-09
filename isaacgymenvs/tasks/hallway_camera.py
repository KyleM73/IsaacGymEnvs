from typing import Dict, Any, Tuple

import gym
from gym import spaces

import numpy as np
import os
import torch
import imageio

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

def set_np_formatting():
    """ formats numpy print """
    np.set_printoptions(edgeitems=3, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=100, formatter=None)

class HallwayCamera(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        set_np_formatting()
        self.cfg = cfg

        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.velocity_cost_scale = self.cfg["env"]["velocityCost"]
        self.ang_velocity_cost_scale = self.cfg["env"]["angularVelocityCost"]
        self.img_cost_scale = self.cfg["env"]["imgCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.boundary = self.cfg["env"]["boundary"] #GET RID OF
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

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 13
        self.cfg["env"]["numActions"] = 3

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        self.img_obs_space = spaces.Box(low=0,high=1,shape=(self.img_w,self.img_h,self.img_d))
        self.vec_obs_space = self.obs_space
        self.obs_space = spaces.Dict(
            spaces={
            "vec" : self.vec_obs_space,
            "img" : self.img_obs_space,
            })

        if self.viewer != None:
            cam_pos = gymapi.Vec3(*self.cfg["viewer"]["pos"])
            cam_target = gymapi.Vec3(*self.cfg["viewer"]["target"])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) #[num_bodies,13]
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)

        self._net_cf = self.gym.acquire_net_contact_force_tensor(self.sim) #[num_bodies,3]
        self.net_cf = gymtorch.wrap_tensor(self._net_cf)
        ##TODO
        # replace cf with force sensors
        # on every human link

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        #print(self.root_tensor.size()) #[4,13] with 2 envs
        #print(self.net_cf.size()) #[num_envs*num_bodies,3]

        self.root_pos = self.root_tensor[:, 0:3] #positions
        self.root_ori = self.root_tensor[:, 3:7] #quaternions
        self.root_lvel = self.root_tensor[:, 7:10] #linear velocities
        self.root_avel = self.root_tensor[:, 10:13] #angular velocities
        self.initial_root_state = self.root_tensor.clone()
        self.initial_root_state[:, 7:13] = 0  #set velocities to zero

        self.targets = to_torch([0, self.max_range, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_angs = to_torch([0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        self.goal_vel = self.cfg["env"]["velocityGoal"] * torch.ones(self.num_envs,1,device=self.device)
        self.potentials = to_torch([-self.max_range/self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        #print(self.root_tensor)

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
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -5, 0)
        upper = gymapi.Vec3(spacing, 30, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/bumpybot.urdf"

        walls_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        walls_file = "urdf/walls.urdf"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)
            walls_file = self.cfg["env"]["asset"].get("wallsFileName", walls_file)
        
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        walls_path = os.path.join(walls_root, walls_file)
        walls_root = os.path.dirname(walls_path)
        walls_file = os.path.basename(walls_path)

        asset_options = gymapi.AssetOptions()
        #asset_options.collapse_fixed_joints = True
        #asset_options.fix_base_link = True
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        asset_bodies = self.gym.get_asset_rigid_body_count(asset)
        #asset = self.gym.create_box(self.sim, 0.2794, 0.2794, 0.635, asset_options)
        
        walls_options = gymapi.AssetOptions()
        walls_options.collapse_fixed_joints = True
        walls_options.fix_base_link = True
        walls = self.gym.load_asset(self.sim, walls_root, walls_file, walls_options)
        walls_bodies = self.gym.get_asset_rigid_body_count(walls)

        self.num_bodies = asset_bodies + walls_bodies

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.5, self.up_axis_idx))
        start_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        walls_pose = gymapi.Transform()
        walls_pose.p = gymapi.Vec3(0,-0.5,0.5)
        walls_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        camera_props = gymapi.CameraProperties()
        camera_props.width = self.img_w
        camera_props.height = self.img_h
        camera_props.enable_tensors = True

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(0,0.25,0.3)
        local_transform.r = gymapi.Quat.from_euler_zyx(0, 0, np.pi/2)

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

            self.gym.attach_camera_to_body(camera_handle, env, handle, local_transform, gymapi.FOLLOW_TRANSFORM) #gymapi.FOLLOW_POSITION doesnt rotate with robot

            self.envs.append(env)
            self.handles.append(handle)
            self.wall_handles.append(walls_handle)
            self.camera_handles.append(camera_handle)

            cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, self.img_type)
            torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
            self.cam_tensors.append(torch_cam_tensor)

    def allocate_buffers(self):
        super().allocate_buffers()
        self.img_tensor = torch.zeros(self.num_envs,self.img_w,self.img_h,self.img_d,device=self.device)
        self.prev_img_tensor = torch.zeros(self.num_envs,self.img_w,self.img_h,self.img_d,device=self.device)

    def _get_images(self):
        #img_dir = "tmp_imgs"
        #if not os.path.exists(img_dir):
        #    os.mkdir(img_dir)
        capture = torch.where(self.progress_buf % 10 == 0,1,0)
        if torch.sum(capture) != 0:
            self.gym.fetch_results(self.sim,True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            for i in range(self.num_envs):
                if capture[i]:
                    img = self.cam_tensors[i].view(self.img_w,self.img_h,self.img_d)
                    self.prev_img_tensor[i,...] = self.img_tensor[i,...]
                    self.img_tensor[i,...] = self._normalize_image(img)
                    #fname = os.path.join(img_dir, "cam-%04d.png" % i)
                    #imageio.imwrite(fname, 255*self._normalize_image(img).cpu().numpy())
            self.gym.end_access_image_tensors(self.sim)

    def _normalize_image(self,img):
        # [W,H,C]
        img[img < -self.cam_max_range] = -self.cam_max_range
        img = -torch.abs(img/(img.min() + 1e-4)) + 1
        return img

    def _compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf = compute_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.velocity_cost_scale,
            self.ang_velocity_cost_scale,
            self.goal_vel,
            self.death_cost,
            self.max_episode_length,
            self.num_bodies,
            self.net_cf,
            self.img_tensor,
            self.prev_img_tensor,
            self.img_cost_scale
            )

    def _compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:] = compute_observations(
            self.root_tensor,
            self.targets,
            self.target_angs,
            self.goal_vel,
            self.potentials,
            self.prev_potentials,
            self.dt,
            self.actions,
            self.max_range,
            self.num_bodies
            )

    def reset_idx(self, env_ids):

        positions = torch.cat((
            torch_rand_float(-0.2, 0.2, (len(env_ids), 2), device=self.device),
            torch.zeros(len(env_ids),11,device=self.device)
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
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        # apply  forces
        self.actions = actions.to(self.device).clone()

        #forces = torch.zeros_like(self.actions)
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

        self._get_images()
        self._compute_observations()
        self._compute_reward(self.actions)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        out = super().step(actions)
        vec_obs = out[0]["obs"] #self.obs_dict
        self.obs_dict["obs"] = {"vec":vec_obs,"img":self.img_tensor}
        return self.obs_dict,*out[1:]

    def reset(self):
        self.obs_dict = super().reset()
        self.obs_dict["obs"] = {"vec":torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device),"img":self.img_tensor}

        return self.obs_dict

    def reset_done(self):
        self.obs_dict,done_env_ids = super().reset_done()
        self.obs_dict["obs"] = {"vec":torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device),"img":self.img_tensor}

        return self.obs_dict, done_env_ids

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_reward(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    velocity_cost_scale,
    ang_velocity_cost_scale,
    goal_vel,
    death_cost,
    max_episode_length,
    num_bodies,
    net_cf,
    img,
    prev_img,
    img_diff_cost_scale
    ):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, Tensor, float, float, int, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward from the direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 2]) * heading_weight
    heading_reward = torch.where(torch.abs(obs_buf[:, 2]) < 0.2, heading_weight_tensor, - heading_weight * obs_buf[:, 2])

    actions_cost = 0 * actions_cost_scale * torch.sum(actions ** 2, dim=-1)

    velocity_cost = velocity_cost_scale * torch.squeeze((torch.linalg.norm(obs_buf[:, 3:5], dim=-1).view(-1,1) - goal_vel) ** 2)

    ang_velocity_cost = ang_velocity_cost_scale * obs_buf[:,6]**2

    # cost from image diff
    img_cost = img_diff_cost_scale * torch.sum(torch.abs(img - prev_img),dim=(1,2))[:,0]

    # reward for duration of being alive
    alive_reward = torch.ones_like(potentials) * 2.0
    progress_reward = potentials - prev_potentials

    total_reward = progress_reward + alive_reward + heading_reward \
        - actions_cost - velocity_cost - ang_velocity_cost - img_cost

    # adjust reward for dead agents
    total_reward = torch.where(torch.sum(torch.abs(net_cf[::num_bodies,:2])) > 0.1, torch.ones_like(total_reward) * death_cost, total_reward)
    #total_reward = torch.where(torch.abs(obs_buf[:, 0]) > boundary, torch.ones_like(total_reward) * death_cost, total_reward)
    #total_reward = torch.where(get_euler_xyz(obs_buf[:, 3:7])[0] < 0.3, torch.ones_like(total_reward) * death_cost, total_reward)

    # reset agents
    reset = torch.where(torch.sum(torch.abs(net_cf[::num_bodies,:2])) > 0.1, torch.ones_like(reset_buf), reset_buf)
    #reset = torch.where(torch.abs(obs_buf[:, 0]) > boundary, torch.ones_like(reset_buf), reset_buf)
    #reset = torch.where(obs_buf[:, 2] < 0.3, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf) #last arg "reset"

    return total_reward, reset

@torch.jit.script
def compute_observations(
    root_states,
    targets,
    target_angs,
    goal_vel,
    potentials, 
    prev_potentials,
    dt,
    actions,
    max_range,
    num_bodies
    ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, int) -> Tuple[Tensor, Tensor, Tensor]

    # x y th vx vy vth gx gy gth ang2target gv fx fy fth

    position = root_states[::num_bodies, 0:3]
    rotation = root_states[::num_bodies, 3:7]
    velocity = root_states[::num_bodies, 7:10]
    ang_velocity = root_states[::num_bodies, 10:13]

    _,_,yaw = get_euler_xyz(rotation)

    to_target = targets - position
    to_target[:, 2] = 0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    # Normalize Observations
    position_norm = position / max_range
    targets_norm = targets / max_range
    heading = normalize_angle(yaw).unsqueeze(-1) / np.pi
    #angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)

    # obs_buf shapes: 2, 1, 2, 1, 2, 1, 1, num_acts(3)
    obs = torch.cat((position_norm[:,:2].view(-1, 2), heading, velocity[:,:2].view(-1,2),
            ang_velocity[:,2].view(-1,1), targets_norm[:,:2].view(-1,2), target_angs[:,0].view(-1,1),
            goal_vel, actions), dim=-1) #13

    return obs, potentials, prev_potentials_new

@torch.jit.script
def image_diff(img1,img2):
    # type: (Tensor, Tensor) -> Tensor
    return torch.sum(torch.abs(img1 - img2),dim=(1,2))
