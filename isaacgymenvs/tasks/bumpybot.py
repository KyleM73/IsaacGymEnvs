from typing import Dict, Any, Tuple

import gym
from gym import spaces

import numpy as np
import os
import torch
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime

import EnvCreator

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.utils.waypoint_checker import *
from isaacgymenvs.utils.np_formatting import *
from isaacgymenvs.tasks.base.vec_task import VecTask

class Bumpybot(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        set_np_formatting()
        self.cfg = cfg

        self.dt = self.cfg["sim"]["dt"]
        self.test = self.cfg["sim"]["test"]

        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.pose_cost_scale = self.cfg["env"]["poseCost"]
        self.ang_cost_scale = self.cfg["env"]["angCost"]
        self.velocity_cost_scale = self.cfg["env"]["velocityCost"]
        self.ang_velocity_cost_scale = self.cfg["env"]["angularVelocityCost"]
        self.contact_cost = self.cfg["env"]["contactCost"]
        self.contact_lim = self.cfg["env"]["contactLimit"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.reward_scale = self.cfg["env"]["rewardScale"]
        
        self.force_scale = self.cfg["env"]["forceScale"]
        self.torque_scale = self.cfg["env"]["torqueScale"]

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
        if self.cfg["image"]["fixCamera"]:
            self.camera_mode = gymapi.FOLLOW_POSITION
        else:
            self.camera_mode = gymapi.FOLLOW_TRANSFORM
        self.record = self.cfg["viewer"]["captureVideo"]
        self.record_freq = self.cfg["viewer"]["captureVideoFreq"]

        if self.record:
            self._set_fig()
            self.fps = int((self.dt * self.update_freq)**-1)
            self.resets = -1
            self.writer = animation.FFMpegWriter(fps=self.fps)
            self.date = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M")
            self.video_dir = "runs/{name}/videos/{date}".format(name=self.cfg["name"],date=self.date)
            if not os.path.exists(self.video_dir):
                os.makedirs(self.video_dir)

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 8
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
            #self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(-2, 12, 0))

        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) #[num_bodies,13]
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)

        #self._fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
        #self.fsdata = gymtorch.wrap_tensor(self._fsdata)

        #self._netcf = self.gym.acquire_net_contact_force_tensor(self.sim)
        #self.netcf = gymtorch.wrap_tensor(self._netcf)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        #self.gym.refresh_net_contact_force_tensor(self.sim)

        self.initial_root_state = self.root_tensor.clone()
        self.initial_root_state[:, 7:13] = 0  #set velocities to zero
        self.prev_root = self.root_tensor.clone()

        self.path.insert(0,tuple(self.cfg["env"]["path"]["start"])) #append start location to beginning of path
        self.path.insert(-1,tuple(self.cfg["env"]["path"]["target"])) #append target location to beginning of path
        self.path = torch.tensor([list(p) for p in self.path]).to(self.device) #[n_wayptsx2]

        self.targets = torch.ones(self.num_envs,dtype=torch.long).to(self.device) #target waypoint index in sef.path. size: [num_envs]

        self.goal_vel = self.cfg["env"]["velocityGoal"] * torch.ones(self.num_envs,1,device=self.device)
        self.goal_ang_vel = self.cfg["env"]["angVelocityGoal"] * torch.ones(self.num_envs,1,device=self.device)
        self.goal_radius = self.cfg["env"]["goalRadius"]
        self.goal_bonus = self.cfg["env"]["goalBonus"]

        self.steps = 0
        self.step_inc = self.cfg["env"]["controlFrequencyInv"]

        self._get_images()
        self.init_img_tensor = self.img_tensor.clone()
                
    def _set_fig(self):
        try:
            plt.close("all")
        except:
            pass
        self.fig,self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=None,hspace=None)
        self.ax.set_axis_off()
        self.frames = []

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

        human_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        human_file = "urdf/human/human.urdf"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)
            occ_file = self.cfg["env"]["asset"].get("occupancyFileName", occ_file)
            human_file = self.cfg["env"]["asset"].get("humanFileName", human_file)
        
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        occ_path = os.path.join(occ_root, occ_file)
        occ_root = os.path.dirname(occ_path)

        env_c = EnvCreator.humanEnvCreator(occ_path)
        walls_path = env_c.get_urdf_fast(occ_root)
        walls_root = os.path.dirname(walls_path)
        walls_file = os.path.basename(walls_path)

        human_path = os.path.join(human_root,human_file)
        human_root = os.path.dirname(human_path)
        human_file = os.path.basename(human_path)

        start = self.cfg["env"]["path"]["start"]
        target = self.cfg["env"]["path"]["target"]
        filter_dist = self.cfg["env"]["path"]["filterDist"]
        self.path = env_c.get_path(start,target,filter_dist)

        human_loc_data_path = occ_root+"/human_loc_data_"+os.path.basename(occ_path).split(".")[0]+".txt"
        self.num_humans = self.cfg["env"]["asset"]["numHumans"]
        self.num_human_samples = self.cfg["env"]["asset"]["numSamples"]
        if not os.path.exists(human_loc_data_path):
            print("Generating human location data...")
            self.loc_data = torch.tensor(env_c.generate_loc_data(start,target,self.num_humans,n=self.num_human_samples,output_dir=occ_root))
            print("Done.")
        else:
            from ast import literal_eval
            print("Retrieving human location data...")
            with open(human_loc_data_path,"r") as f:
                self.loc_data = torch.tensor([list(literal_eval(line)) for line in f])
                if self.loc_data.size() != (self.num_human_samples,self.num_humans,3):
                    if self.loc_data.size()[0] < self.num_human_samples or self.loc_data.size()[1] < self.num_humans:
                        print("Retrieval failed.")
                        print("Generating human location data...")
                        self.loc_data = torch.tensor(env_c.generate_loc_data(start,target,self.num_humans,n=self.num_human_samples,output_dir=occ_root))
            print("Done.")

        self.num_assets = 0

        asset_options = gymapi.AssetOptions()
        asset_options.collapse_fixed_joints = True
        #asset_options.fix_base_link = True
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        asset_bodies = self.gym.get_asset_rigid_body_count(asset)
        self.num_assets += 1
        
        #base_idx = self.gym.find_asset_rigid_body_index(asset, "base")
        #sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
        #sensor_props = gymapi.ForceSensorProperties()
        #sensor_props.enable_forward_dynamics_forces = False
        #sensor_props.enable_constraint_solver_forces = True
        #sensor_props.use_world_frame = True #false??
        #sensor_idx = self.gym.create_asset_force_sensor(asset, base_idx, sensor_pose, sensor_props)

        walls_options = gymapi.AssetOptions()
        walls_options.collapse_fixed_joints = True
        walls_options.fix_base_link = True
        walls = self.gym.load_asset(self.sim, walls_root, walls_file, walls_options)
        walls_bodies = self.gym.get_asset_rigid_body_count(walls)
        self.num_assets += 1

        human_options = gymapi.AssetOptions()
        human_options.collapse_fixed_joints = True
        human_options.fix_base_link = True
        human = self.gym.load_asset(self.sim, human_root, human_file, human_options)
        human_bodies = self.gym.get_asset_rigid_body_count(human)
        self.num_assets += self.num_humans

        self.num_bodies = asset_bodies + walls_bodies + self.num_humans*human_bodies

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.635/2, self.up_axis_idx))
        start_pose.r = gymapi.Quat.from_euler_zyx(0, 0, np.pi/2) #face y axis

        walls_pose = gymapi.Transform()
        walls_pose.p = gymapi.Vec3(0,0,0)
        walls_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        camera_props = gymapi.CameraProperties()
        camera_props.width = self.img_w
        camera_props.height = self.img_h
        camera_props.use_collision_geometry = False
        camera_props.enable_tensors = True

        camera_pose = gymapi.Transform()
        camera_pose.p = gymapi.Vec3(0,0.25,0.3) # get real values from robot
        camera_pose.r = gymapi.Quat.from_euler_zyx(0, 0, np.pi/2)

        if self.test and self.cfg["env"]["path"]["showPath"]:
            path_path = env_c.path2urdf(self.path,occ_root)
            path_root = os.path.dirname(path_path)
            path_file = os.path.basename(path_path)

            path_options = gymapi.AssetOptions()
            path_options.collapse_fixed_joints = True
            path_options.fix_base_link = True
            path = self.gym.load_asset(self.sim, path_root, path_file, path_options)
            path_bodies = self.gym.get_asset_rigid_body_count(path)

            self.num_bodies += path_bodies
            self.num_assets += 1

            path_pose = gymapi.Transform()
            path_pose.p = gymapi.Vec3(0,0,0) # waypoints above env so camera cant see them
            path_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        self.handles = []
        self.wall_handles = []
        self.camera_handles = []
        self.cam_tensors = []
        self.envs = []
        self.human_handles = []

        if self.test and self.cfg["env"]["asset"]["setLoc"]:
            human_idx = self.cfg["env"]["asset"]["testIdx"]*np.ones(self.num_envs,dtype=int)
        else:
            self.rng = np.random.default_rng()
            human_idx = self.rng.integers(low=0,high=self.num_human_samples,size=self.num_envs)

        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            handle = self.gym.create_actor(env, asset, start_pose, "bumpybot", i)
            walls_handle = self.gym.create_actor(env, walls, walls_pose, "walls", i) # take this out of loop and use  -1 as last arg so all actors can collide with same walls
            camera_handle = self.gym.create_camera_sensor(env, camera_props)

            self.gym.attach_camera_to_body(camera_handle, env, handle, camera_pose, self.camera_mode) #gymapi.FOLLOW_TRANSFORM,gymapi.FOLLOW_POSITION doesnt rotate with robot

            human_handles_inner = []
            for h in range(self.num_humans):
                loc = self.loc_data[human_idx[i],h,:]
                human_pose = gymapi.Transform()
                human_pose.p = gymapi.Vec3(loc[0],loc[1],1.12) #human height, put in config
                human_pose.r = gymapi.Quat.from_euler_zyx(np.pi/2,0,loc[2])

                human_handle = self.gym.create_actor(env,human,human_pose,"human_"+str(h),i)
                human_handles_inner.append(human_handle)

                #props_shape = self.gym.get_actor_rigid_shape_properties(env, human_handle)
                #for b in range(len(props_shape)):
                #    props_shape[b].filter = b
                #self.gym.set_actor_rigid_shape_properties(env, human_handle, props_shape)


            self.envs.append(env)
            self.handles.append(handle)
            self.wall_handles.append(walls_handle)
            self.camera_handles.append(camera_handle)
            self.human_handles.append(human_handles_inner)

            cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, self.img_type)
            torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
            self.cam_tensors.append(torch_cam_tensor)

            props_shape = self.gym.get_actor_rigid_shape_properties(env, handle)
            props_shape[0].rolling_friction = 0.0
            props_shape[0].torsion_friction = 0.0
            props_shape[0].friction = 0.0
            props_shape[0].restitution = 0.0
            self.gym.set_actor_rigid_shape_properties(env, handle, props_shape)

            if self.test and self.cfg["env"]["path"]["showPath"]:
                self.gym.create_actor(env, path, path_pose, "path", i)

        self.mass = self.gym.get_actor_rigid_body_properties(env,handle)[0].mass

    def allocate_buffers(self):
        super().allocate_buffers()
        self.img_tensor = torch.zeros(self.num_envs,self.img_w,self.img_h,self.img_d,device=self.device)
        self.frames_in_contact = torch.zeros_like(self.progress_buf,device=self.device)

    def _get_images(self):
        if self.steps % self.update_freq:
            return
        #img_dir = "tmp_imgs"
        #if not os.path.exists(img_dir):
        #    os.mkdir(img_dir)
        self.gym.fetch_results(self.sim,True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for i in range(self.num_envs):
            img = self.cam_tensors[i].view(self.img_w,self.img_h,self.img_d)
            self.img_tensor[i,...] = self._normalize_image(img)
            #fname = os.path.join(img_dir, "cam-%04d.png" % i)
            #imageio.imwrite(fname, 255*self._normalize_image(img).cpu().numpy())
        self.gym.end_access_image_tensors(self.sim)

        if self.record and not self.resets % self.record_freq:
            self.frames.append([self.ax.imshow(255*self.img_tensor[0,...].cpu().numpy(),animated=True,cmap="gray")])

    def _normalize_image(self,img):
        return normalize_image(img,self.cam_max_range)

    def _compute_reward(self, actions):
        self.contact_forces = get_contact_force(
            self.root_tensor[::self.num_assets,7:9],
            self.prev_root[::self.num_assets,7:9],
            self.dt*self.control_freq_inv,
            self.dt,
            self.actions[:,:2],
            self.mass
            )

        self.rew_buf[:], self.reset_buf[:], self.targets[:], self.frames_in_contact[:] = compute_reward(
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
            self.contact_forces,
            self.contact_cost,
            self.contact_lim,
            self.death_cost,
            self.path,
            self.targets,
            self.frames_in_contact,
            self.reward_scale
            )

    def _compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        #self.gym.refresh_force_sensor_tensor(self.sim)
        #self.gym.refresh_net_contact_force_tensor(self.sim)

        self.obs_buf[:] = compute_observations(
            self.root_tensor,
            self.path,
            self.targets,
            self.goal_vel,
            self.goal_ang_vel,
            self.dt,
            self.actions,
            self.num_assets
            )

    def reset_idx(self, env_ids):

        vels = torch.cat((
            torch.zeros(len(env_ids),7,device=self.device), #pose,quat
            torch_rand_float(-0.1, 0.1, (len(env_ids),2), device=self.device), #vx,vy
            torch.zeros(len(env_ids),4,device=self.device) #vy,v_ang
            ),dim=-1)
        pose = torch.zeros_like(self.initial_root_state)

        pose[self.num_assets*env_ids] = vels

        random_root = self.initial_root_state + pose

        env_ids_int32 = (self.num_assets*env_ids).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(random_root),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.img_tensor[env_ids] = self.init_img_tensor[env_ids]
        #self.fsdata[env_ids] = 0

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        # apply  forces
        self.prev_root[:] = self.root_tensor.clone()

        self.actions = actions.to(self.device).clone()

        forces = torch.zeros((self.num_envs*self.num_bodies, 3), device="cuda:0", dtype=torch.float)
        forces[::self.num_bodies, :2] = self.force_scale * self.actions[:, :2]

        torques = torch.zeros((self.num_envs*self.num_bodies, 3), device="cuda:0", dtype=torch.float)
        torques[::self.num_bodies, 2] = self.torque_scale * self.actions[:, 2]

        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

            if self.record and 0 in env_ids:
                if len(self.frames) > 1:
                    if self.test:
                        self.vname = "{dir}/{name}_test.mp4".format(dir=self.video_dir,name=self.cfg["name"])
                        self.record = False
                    else:
                        self.vname = "{dir}/{name}_train{reset}.mp4".format(dir=self.video_dir,name=self.cfg["name"],reset=self.resets)
                    ani = animation.ArtistAnimation(self.fig,self.frames,interval=int(1000/self.fps),blit=True,repeat=False)
                    ani.save(self.vname,writer=self.writer)
                    self._set_fig()
                    self.frames = []
                self.resets += 1

        self._get_images()
        self._compute_observations()
        self._compute_reward(self.actions)

        self.steps += self.step_inc

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        out = super().step(actions)
        vec_obs = out[0]["obs"] #self.obs_dict
        self.obs_dict["obs"] = {
            "vec":vec_obs,
            "img":self.img_tensor
            }
        return self.obs_dict,*out[1:]

    def reset(self):
        self.obs_dict = super().reset()
        self.obs_dict["obs"] = {
            "vec":torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device),
            "img":self.img_tensor
            }

        return self.obs_dict

    def reset_done(self):
        self.obs_dict,done_env_ids = super().reset_done()
        self.obs_dict["obs"] = {
        "vec":torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device),
        "img":self.img_tensor
        }

        return self.obs_dict, done_env_ids


#####################################################################
###=========================jit functions=========================###
#####################################################################
def rew_func(value,scale=1):
    # type: (Tensor, float) -> Tensor
    return 1.0 / (1.0 + scale * value * value)

def rew_func_tensor(value,scale):
    # type: (Tensor, Tensor) -> Tensor
    return 1.0 / (1.0 + scale * value * value)

@torch.jit.script
def get_contact_force(vel,vel_p,dt,dt_action,action,mass):
    # type: (Tensor, Tensor, float, float, Tensor, float) -> Tensor
    # vel : [num_envs 2]
    # vel_p : [num_envs 2]
    # dt : float
    # dt_action : float
    # action : [num_envs 2]
    # mass : float
    vel_ = vel_p + dt_action * action # updated velocity after action
    a = (vel - vel_) / dt
    return mass * a

@torch.jit.script
def normalize_image(img,cam_max_range):
    # [W,H,C]
    img[img < -cam_max_range] = -cam_max_range
    img = -torch.abs(img/(-cam_max_range)) + 1
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
    contact_forces,
    contact_cost_scale,
    contact_lim,
    death_cost,
    path,
    targets,
    frames_in_contact,
    reward_scale
    ):
    # type: (Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, Tensor, float, float, float, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    actions = torch.linalg.norm(obs_buf[:, 5:],dim=-1)
    actions_reward = rew_func(actions,actions_cost_scale)

    pose = torch.linalg.norm(obs_buf[:, :2],dim=-1)
    pose_reward = rew_func(pose,pose_cost_scale)

    vel = torch.abs(obs_buf[:, 2])
    vel_reward = rew_func(vel,velocity_cost_scale)

    ang = torch.abs(obs_buf[:, 3])
    ang_reward = rew_func(ang,ang_cost_scale)

    ang_vel = torch.abs(obs_buf[:, 4])
    ang_vel_reward = rew_func(ang_vel,ang_velocity_cost_scale)

    contact = torch.linalg.norm(contact_forces,dim=-1)
    zero_contact_idx = torch.argwhere(contact < 1e-4).flatten()
    contact_idx = torch.argwhere(contact > 1e-4).flatten()
    frames_in_contact[zero_contact_idx] = 0
    frames_in_contact[contact_idx] += 1
    contact_cost = contact_cost_scale * frames_in_contact * frames_in_contact
    contact_reward = rew_func_tensor(contact,contact_cost)

    total_reward = actions_reward + pose_reward + vel_reward + ang_reward + ang_vel_reward + contact_reward
    total_reward /= 6 #normalize to 1
    total_reward *= reward_scale
    #max reward = reward_scale * steps * 1 + len(path) * 100

    # terminal costs
    total_reward = torch.where(pose <= goal_radius, total_reward + goal_bonus, total_reward)

    # adjust reward for dead agents
    total_reward_ = total_reward
    total_reward = torch.where(contact > contact_lim, torch.ones_like(total_reward) * death_cost, total_reward)

    # update targets
    update_conds = (pose <= goal_radius) | (check_waypoints(obs_buf[:, :2]+path[targets],path,targets))
    targets_ = targets
    targets = torch.where(update_conds,targets+1,targets)

    # reset agents
    reset_conds = (contact > contact_lim) | (progress_buf >= max_episode_length - 1) | (targets > len(path)-2)
    reset = torch.where(reset_conds, torch.ones_like(reset_buf), reset_buf) #last arg "reset"
    targets = torch.where(reset_conds,torch.ones_like(targets),targets)

    # handle first-step error
    err_conds = (progress_buf < 2) & (contact > contact_lim)
    err_idx = torch.argwhere(err_conds).flatten()
    total_reward[err_idx] = total_reward_[err_idx]
    reset[err_idx] = 0
    targets[err_idx] = targets_[err_idx]

    #print(fsdata)
    #if torch.any(contact > 1e-5):
    #    print()
    #    print()
        
    ## TODO
    # - add power usage terms
    # - timeout bootstrapping (see ETH parallel walking paper)

    return total_reward, reset, targets, frames_in_contact

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

    dot_prod = torch.einsum('ij,ij->i',position[:, :2],path[targets+1]) #look at next waypoint
    dot_prod /= torch.linalg.norm(position[:, :2]+1e-7,dim=-1)
    dot_prod /= torch.linalg.norm(path[targets+1]+1e-7,dim=-1)
    dot_prod = torch.clamp(dot_prod,-1+1e-7,1-1e-7)
    target_angs = torch.arccos(dot_prod).view(-1,1)

    heading_err = heading - target_angs

    vel_error = torch.linalg.norm(velocity[:,:2],dim=-1).view(-1,1) - goal_vel #[1]

    ang_vel_error = torch.linalg.norm(ang_velocity[:,2],dim=-1) - goal_ang_vel #[1]

    # obs_buf shapes: 2, 1, 1, 1, num_acts(3)
    obs = torch.cat((pose_error,vel_error,
        heading_err,ang_vel_error,actions),dim=-1) #8

    return obs