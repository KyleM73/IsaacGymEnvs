from typing import Dict, Any, Tuple

import gym
from gym import spaces

import numpy as np
import os
import sys
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
from isaacgymenvs.utils.img_utils import rgba2rgb
from isaacgymenvs.tasks.base.vec_task import VecTask

class Bumpybot_blind(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render=True):
        set_np_formatting()
        self.cfg = cfg

        self.dt = self.cfg["sim"]["dt"]
        self.test = self.cfg["sim"]["test"]
        self.fancyTest = self.cfg["viewer"]["fancyTest"]["test"]
        self.set_location = self.cfg["env"]["asset"]["setLoc"]
        
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.pose_cost_scale = self.cfg["env"]["poseCost"]
        self.ang_cost_scale = self.cfg["env"]["angCost"]
        self.velocity_cost_scale = self.cfg["env"]["velocityCost"]
        self.ang_velocity_cost_scale = self.cfg["env"]["angularVelocityCost"]
        self.contact_cost = self.cfg["env"]["contactCost"]
        self.contact_lim = self.cfg["env"]["contactLimit"]
        self.contact_thresh = self.cfg["env"]["contactThreshold"]
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
            if self.img_d > 3:
                self.img_d = 3
        self.fstack_num = self.cfg["image"]["fstack"]
        self.channels = self.img_d*self.fstack_num
        self.cam_max_range = self.cfg["image"]["range"]
        self.cam_min_range = self.cfg["image"]["minRange"]
        self.update_freq = self.cfg["image"]["updateFreq"]
        if self.cfg["image"]["fixCamera"]:
            self.camera_mode = gymapi.FOLLOW_POSITION
        else:
            self.camera_mode = gymapi.FOLLOW_TRANSFORM
        self.FOV = self.cfg["image"]["FOV"]

        self.record = self.cfg["viewer"]["captureVideo"]
        self.record_freq = self.cfg["viewer"]["captureVideoFreq"]
        self.record_once = self.cfg["viewer"]["captureOnce"]
        if not self.test: self.record_once =  False
        self.rgb_h = self.cfg["viewer"]["height"]
        self.rgb_w = self.cfg["viewer"]["width"]

        if self.record:
            if self.fancyTest:
                self.fancyTest_h = [self.cfg["viewer"]["fancyTest"]["cam_0"]["height"]]
                self.fancyTest_w = [self.cfg["viewer"]["fancyTest"]["cam_0"]["width"]]
                self.fancyTest_pos = [None]
                self.fancyTest_tgt = [None]
                self.fancyTest_num = 1
                for k,v in self.cfg["viewer"]["fancyTest"]["cams"].items():
                    self.fancyTest_h.append(v["height"])
                    self.fancyTest_w.append(v["width"])
                    self.fancyTest_pos.append(v["pos"])
                    self.fancyTest_tgt.append(v["target"])
                    self.fancyTest_num += 1

            self._set_fig()
            self.fps = int((self.dt * self.update_freq)**-1)
            #if self.test:
            #    self.fps *= 2
            self.resets = -1
            self.writer = animation.FFMpegWriter(fps=self.fps) 
            vdir = self.cfg["videoDir"]
            if vdir is "":
                self.date = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M")
                self.video_dir = "runs/{name}/videos/{date}_{name}".format(name=self.cfg["name"],date=self.date)
            else:
                self.video_dir = "runs/{name}/videos/{dir}".format(name=self.cfg["name"],dir=vdir)
            if not os.path.exists(self.video_dir):
                os.makedirs(self.video_dir)

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 15
        self.cfg["env"]["numActions"] = 2 #scale, th, th_heading, humandist, rwalldist, lwalldist

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(*self.cfg["viewer"]["pos"])
            cam_target = gymapi.Vec3(*self.cfg["viewer"]["target"])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            #self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(-2, 12, 0))

        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) #[num_bodies,13]
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)

        #self._fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
        #self.fsdata = gymtorch.wrap_tensor(self._fsdata)

        self._net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(self._net_cf)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        #self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.initial_root_state = self.root_tensor.clone()
        self.initial_root_state[:, 7:13] = 0  #set velocities to zero
        self.prev_root = self.initial_root_state.clone()
        self.prev_actions = None

        self.path.insert(0,tuple(self.cfg["env"]["path"]["start"])) #append start location to beginning of path
        self.path.insert(-1,tuple(self.cfg["env"]["path"]["target"])) #append target location to beginning of path
        self.path = torch.tensor([list(p) for p in self.path]).to(self.device) #[n_wayptsx2]

        self.targets = torch.ones(self.num_envs,dtype=torch.long).to(self.device) #target waypoint index in sef.path. size: [num_envs]

        self.goal_vel = self.cfg["env"]["velocityGoal"] * torch.ones(self.num_envs,1,device=self.device)
        self.max_vel = self.cfg["env"]["velocityMax"]
        self.goal_ang_vel = self.cfg["env"]["angVelocityGoal"] * torch.ones(self.num_envs,1,device=self.device)
        self.goal_radius = self.cfg["env"]["goalRadius"]
        self.goal_bonus = self.cfg["env"]["goalBonus"]

        self.steps = 0
        self.step_inc = self.cfg["env"]["controlFrequencyInv"]

        self._get_images()

    def _set_fig(self):
        try:
            plt.close("all")
        except:
            pass

        self.fig_rgb,self.ax_rgb = plt.subplots()
        self.fig_rgb.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=None,hspace=None)
        self.ax_rgb.set_axis_off()
        self.frames_rgb = []

        if self.fancyTest:
            self.fancyTest_figs = []
            self.fancyTest_axes = []
            self.fancyTest_frames = [[] for _ in range(self.fancyTest_num)]
            for i in range(self.fancyTest_num):
                fig,ax = plt.subplots()
                fig.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=None,hspace=None)
                ax.set_axis_off()
                self.fancyTest_figs.append(fig)
                self.fancyTest_axes.append(ax)

        self.fig_q,self.ax_q = plt.subplots(3,1)
        self.fig_qdot,self.ax_qdot = plt.subplots(3,1)
        self.fig_qddot,self.ax_qddot = plt.subplots(2,1)
        self.fig_err,self.ax_err = plt.subplots(3,1)
        self.fig_contact,self.ax_contact = plt.subplots(2,1)
        self.fig_rew,self.ax_rew = plt.subplots(1,1)
        self.x_hist,self.y_hist,self.th_hist = [],[],[]
        self.vx_hist,self.vy_hist,self.vth_hist = [],[],[]
        self.av_hist,self.ah_hist = [],[]
        self.xerr_hist,self.yerr_hist,self.therr_hist = [],[],[]
        self.heading_hist,self.contact_hist = [],[]
        self.heading_motion,self.heading_cam = [],[]
        if self.cfg["env"]["asset"]["numHumans"] > 0:
            self.human_contact_hists = [[] for _ in range(self.cfg["env"]["asset"]["numHumans"])]
        self.step_rew = []

    def _fill_fig(self):
        #from root state 
        self.x_hist.append(self.root_tensor[0,0].cpu())
        self.y_hist.append(self.root_tensor[0,1].cpu())
        rotation = self.root_tensor[::self.num_assets,3:7].cpu()
        _,_,yaw = get_euler_xyz(rotation)
        heading = normalize_angle(yaw).unsqueeze(-1) - np.pi/2
        self.th_hist.append(heading[0])
        self.vx_hist.append(self.root_tensor[0,7].cpu())
        self.vy_hist.append(self.root_tensor[0,8].cpu())
        self.vth_hist.append(self.root_tensor[0,12].cpu())

        #from actions
        self.av_hist.append(self.actions[0,0].cpu())
        self.ah_hist.append(self.actions[0,1].cpu())

        #from obs
        self.xerr_hist.append(self.obs_buf[0,0].detach().cpu())
        self.yerr_hist.append(self.obs_buf[0,1].detach().cpu())
        self.therr_hist.append(self.obs_buf[0,8].detach().cpu())
        self.heading_motion.append(self.obs_buf[0,9].detach().cpu())
        self.heading_cam.append(self.obs_buf[0,7].detach().cpu())
        heading_diff = self.obs_buf[0,7].detach().cpu()-self.obs_buf[0,9].detach().cpu()
        heading_diff = (heading_diff + 2*np.pi) % (2*np.pi)
        if heading_diff > np.pi:
            heading_diff -= 2*np.pi
        self.heading_hist.append(heading_diff)
        self.contact_hist.append(self.obs_buf[0,12].detach().cpu())
        if self.num_humans > 0:
            for i in range(self.num_humans):
                # lower body (incuding pelivs) = inds [3,13]
                human_cf = self.net_cf[self.asset_bodies+self.walls_bodies+i*self.human_bodies+3:self.asset_bodies+self.walls_bodies+i*self.human_bodies+13+1,:2]
                self.human_contact_hists[i].append(torch.linalg.norm(torch.sum(human_cf,dim=0)).cpu()/self.contact_lim)

        self.step_rew.append(self.rew_buf[0].detach().cpu())

    def _make_plots(self,label):
        assert isinstance(label,str)
        t = [i for i in range(len(self.x_hist))]

        #fix t0 error
        self.x_hist[0] = self.x_hist[1]
        self.y_hist[0] = self.y_hist[1]
        self.th_hist[0] = self.th_hist[1]
        self.vx_hist[0] = self.vx_hist[1]
        self.vy_hist[0] = self.vy_hist[1]
        self.vth_hist[0] = self.vth_hist[1]
        self.av_hist[0] = self.av_hist[1]
        self.ah_hist[0] = self.ah_hist[1]
        self.xerr_hist[0] = self.xerr_hist[1]
        self.yerr_hist[0] = self.yerr_hist[1]
        self.therr_hist[0] = self.therr_hist[1]
        self.heading_hist[0] = self.heading_hist[1]
        self.heading_motion[0] = self.heading_motion[1]
        self.heading_cam[0] = self.heading_cam[1]
        self.contact_hist[0] = self.contact_hist[1]
        if self.num_humans > 0:
            for i in range(self.num_humans):
                self.human_contact_hists[i][0] = self.human_contact_hists[i][1]
        self.step_rew[0] = self.step_rew[1]

        self.ax_q[0].plot(t,self.x_hist,label="x")
        self.ax_q[0].set_ylabel("x")
        self.ax_q[0].set_xticks([])
        self.ax_q[1].plot(t,self.y_hist,label="y")
        self.ax_q[1].set_ylabel("y")
        self.ax_q[1].set_xticks([])
        self.ax_q[2].plot(t,self.th_hist,label="th")
        self.ax_q[2].set_ylabel("th")
        self.ax_q[2].set_xlabel("steps")
        self.fig_q.suptitle("Pose",fontsize=16)
        #self.fig_q.legend(loc="upper right")
        if label == "train":
            q_fname = "{dir}/train_reset{reset}/pose.png".format(dir=self.video_dir,reset=self.resets)
        else:
            q_fname = "{dir}/test/pose.png".format(dir=self.video_dir,reset=self.resets)
        self.fig_q.savefig(q_fname)

        self.ax_qdot[0].plot(t,self.vx_hist,label="vx")
        self.ax_qdot[0].set_ylabel("vx")
        self.ax_qdot[0].set_xticks([])
        self.ax_qdot[1].plot(t,self.vy_hist,label="vy")
        self.ax_qdot[1].set_ylabel("vy")
        self.ax_qdot[1].set_xticks([])
        self.ax_qdot[2].plot(t,self.vth_hist,label="vth")
        self.ax_qdot[2].set_ylabel("vth")
        self.ax_qdot[2].set_xlabel("steps")
        self.fig_qdot.suptitle("Velocity",fontsize=16)
        #self.fig_qdot.legend(loc="upper right")
        if label == "train":
            qdot_fname = "{dir}/train_reset{reset}/velocity.png".format(dir=self.video_dir,reset=self.resets)
        else:
            qdot_fname = "{dir}/test/velocity.png".format(dir=self.video_dir,reset=self.resets)
        self.fig_qdot.savefig(qdot_fname)

        self.ax_qddot[0].plot(t,self.av_hist,label="av")
        self.ax_qddot[0].set_ylabel("av")
        self.ax_qddot[0].set_xticks([])
        self.ax_qddot[1].plot(t,self.ah_hist,label="ah")
        self.ax_qddot[1].set_ylabel("ah")
        self.ax_qddot[1].set_xticks([])
        self.ax_qddot[1].set_xlabel("steps")
        self.fig_qddot.suptitle("Action",fontsize=16)
        #self.fig_qddot.legend(loc="upper right")
        for ax in self.ax_qddot:
            ax.set_ylim([-1.5,1.5])
        if label == "train":
            qddot_fname = "{dir}/train_reset{reset}/action.png".format(dir=self.video_dir,reset=self.resets)
        else:
            qddot_fname = "{dir}/test/action.png".format(dir=self.video_dir,reset=self.resets)
        self.fig_qddot.savefig(qddot_fname)

        self.ax_err[0].plot(t,self.xerr_hist,label="xerr")
        self.ax_err[0].set_ylabel("x error")
        self.ax_err[0].set_xticks([])
        self.ax_err[1].plot(t,self.yerr_hist,label="yerr")
        self.ax_err[1].set_ylabel("y error")
        self.ax_err[1].set_xticks([])
        self.ax_err[2].plot(t,self.therr_hist,label="therr")
        self.ax_err[2].set_ylabel("th error")
        self.ax_err[2].set_xlabel("steps")
        self.fig_err.suptitle("Error",fontsize=16)
        #self.fig_err.legend(loc="upper right")
        if label == "train":
            err_fname = "{dir}/train_reset{reset}/error.png".format(dir=self.video_dir,reset=self.resets)
        else:
            err_fname = "{dir}/test/error.png".format(dir=self.video_dir,reset=self.resets)
        self.fig_err.savefig(err_fname)
        self.ax_contact[0].plot(t,self.heading_hist,label="heading diff")
        self.ax_contact[0].plot(t,self.heading_motion,label="heading motion")
        self.ax_contact[0].plot(t,self.heading_cam,label="heading cam")
        self.ax_contact[0].set_ylabel("heading error")
        self.ax_contact[0].legend(loc="upper right")
        self.ax_contact[0].set_xticks([])
        self.ax_contact[1].plot(t,self.contact_hist,label="contact",zorder=10)
        #if self.num_humans > 0:
        #    for i in range(self.num_humans):
        #        self.ax_contact[1].plot(t,self.human_contact_hists[i],label="human {}".format(i))
        self.ax_contact[1].plot(t,[0.5 for i in range(len(self.x_hist))],linestyle="--",color="grey")
        self.ax_contact[1].plot(t,[1 for i in range(len(self.x_hist))],linestyle="--",color="grey")
        self.ax_contact[1].set_ylim([0,1.1])
        self.ax_contact[1].set_ylabel("contact ratio")
        self.ax_contact[1].set_xlabel("steps")
        #self.ax_contact[1].legend(loc="upper right")
        self.fig_contact.suptitle("Heading & Contact",fontsize=16)
        #self.fig_contact.legend(loc="upper right")
        if label == "train":
            contact_fname = "{dir}/train_reset{reset}/contact.png".format(dir=self.video_dir,reset=self.resets)
        else:
            contact_fname = "{dir}/test/contact.png".format(dir=self.video_dir,reset=self.resets)
        self.fig_contact.savefig(contact_fname)

        self.ax_rew.plot(t,self.step_rew,label="reward")
        self.ax_rew.set_ylim([0,2.1])
        self.fig_rew.suptitle("Rewards",fontsize=16)
        if label == "train":
            rew_fname = "{dir}/train_reset{reset}/rewards.png".format(dir=self.video_dir,reset=self.resets)
        else:
            rew_fname = "{dir}/test/rewards.png".format(dir=self.video_dir,reset=self.resets)
        self.fig_rew.savefig(rew_fname)

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
            if self.fancyTest:
                asset_file = self.cfg["viewer"]["fancyTest"]["fancyUrdf"]
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
        print("Path: ",self.path)

        self.num_humans = self.cfg["env"]["asset"]["numHumans"]
        if self.set_location and self.num_humans > 0:
            self.num_human_samples = 1
            print("Using set human location data...")
            locs = []
            c = 0
            for h in range(self.num_humans):
                if h < 5:
                    loc = [-1.5+0.5*h,5+0.1*h,0]
                elif h < 10:
                    loc = [1.5-0.5*(h-5),10+0.1*(h-5),0]
                else:
                    c += 1
                locs.append(loc)
            self.num_humans -= c

            self.loc_data = torch.tensor(locs).view(self.num_human_samples,self.num_humans,3)
            print("Done.")
        else:
            human_loc_data_path = occ_root+"/human_loc_data_"+os.path.basename(occ_path).split(".")[0]+".txt"
            self.num_human_samples = self.cfg["env"]["asset"]["numSamples"]
            human_start = start[:]
            human_start[1] += self.cfg["env"]["asset"]["start_offset"]
            human_target = target[:]
            human_target[1] -= self.cfg["env"]["asset"]["end_offset"] #give free space to move into at end of path
            if not os.path.exists(human_loc_data_path):
                print("Generating human location data...")
                self.loc_data = torch.tensor(env_c.generate_loc_data(
                    human_start,human_target,self.num_humans,start_free_zone=0.6,end_free_zone=0,
                    n=self.num_human_samples,output_dir=occ_root))
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
                            self.loc_data = torch.tensor(env_c.generate_loc_data(
                                human_start,human_target,self.num_humans,start_free_zone=0.6,end_free_zone=0,
                                n=self.num_human_samples,output_dir=occ_root))
                print("Done.")

        self.num_assets = 0

        asset_options = gymapi.AssetOptions()
        asset_options.collapse_fixed_joints = True
        asset_options.fix_base_link = False
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.asset_bodies = self.gym.get_asset_rigid_body_count(asset)
        self.num_assets += 1

        #base_idx = self.gym.find_asset_rigid_body_index(asset, "base")
        #sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
        #sensor_props = gymapi.ForceSensorProperties()
        #sensor_props.enable_forward_dynamics_forces = False
        #sensor_props.enable_constraint_solver_forces = True
        #sensor_props.use_world_frame = False
        #sensor_idx = self.gym.create_asset_force_sensor(asset, base_idx, sensor_pose, sensor_props)

        walls_options = gymapi.AssetOptions()
        walls_options.collapse_fixed_joints = True
        walls_options.fix_base_link = True
        walls = self.gym.load_asset(self.sim, walls_root, walls_file, walls_options)
        self.walls_bodies = self.gym.get_asset_rigid_body_count(walls)
        self.num_assets += 1

        human_options = gymapi.AssetOptions()
        human_options.collapse_fixed_joints = True
        human_options.fix_base_link = True
        human = self.gym.load_asset(self.sim, human_root, human_file, human_options)
        self.human_bodies = self.gym.get_asset_rigid_body_count(human)
        self.num_assets += self.num_humans

        self.num_bodies = self.asset_bodies + self.walls_bodies + self.num_humans*self.human_bodies

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.6318758/2, self.up_axis_idx))
        start_pose.r = gymapi.Quat.from_euler_zyx(0,0,np.pi/2) #face y axis

        walls_pose = gymapi.Transform()
        walls_pose.p = gymapi.Vec3(0,0,0)
        walls_pose.r = gymapi.Quat.from_euler_zyx(0,0,0)

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
        #self.fs_handles = []

        if self.test and self.set_location:
            human_idx = self.cfg["env"]["asset"]["testIdx"]*np.ones(self.num_envs,dtype=int)
        else:
            self.rng = np.random.default_rng()
            human_idx = self.rng.integers(low=0,high=self.num_human_samples,size=self.num_envs)

        if self.record:
            camera_props_rgb = gymapi.CameraProperties()
            camera_props_rgb.width = self.rgb_w #self.img_w
            camera_props_rgb.height = self.rgb_h #self.img_h
            camera_props_rgb.use_collision_geometry = False
            camera_props_rgb.enable_tensors = True
            if self.fancyTest:
                fancy_camera_props = []
                for i in range(self.fancyTest_num):
                    prop = gymapi.CameraProperties()
                    prop.height = self.fancyTest_h[i]
                    prop.width = self.fancyTest_w[i]
                    prop.use_collision_geometry = False
                    prop.enable_tensors = True
                    fancy_camera_props.append(prop)

        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            handle = self.gym.create_actor(env, asset, start_pose, "bumpybot", i)
            walls_handle = self.gym.create_actor(env, walls, walls_pose, "walls", i) # take this out of loop and use  -1 as last arg so all actors can collide with same walls

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

            #fs = self.gym.get_actor_force_sensor(env, handle, i)
            #self.fs_handles.append(fs)

            self.envs.append(env)
            self.handles.append(handle)
            self.wall_handles.append(walls_handle)
            self.human_handles.append(human_handles_inner)

            if i==0 and self.record:
                camera_handle_rgb = self.gym.create_camera_sensor(env, camera_props_rgb)
                self.gym.set_camera_location(
                        camera_handle_rgb, 
                        env, 
                        gymapi.Vec3(*self.cfg["viewer"]["pos"]),
                        gymapi.Vec3(*self.cfg["viewer"]["target"]))

                cam_tensor_rgb = self.gym.get_camera_image_gpu_tensor(self.sim,env, camera_handle_rgb, gymapi.IMAGE_COLOR)
                self.torch_cam_tensor_rgb = gymtorch.wrap_tensor(cam_tensor_rgb)

                if self.fancyTest:
                    self.fancy_cam_handles = []
                    self.fancy_cam_tensors = []

                    cam_hndl = self.gym.create_camera_sensor(env,fancy_camera_props[0])
                    self.gym.attach_camera_to_body(cam_hndl, env, handle, camera_pose, self.camera_mode)
                    self.fancy_cam_handles.append(cam_hndl)
                    fancy_cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, cam_hndl, gymapi.IMAGE_COLOR)
                    torch_fancy_cam_tensor = gymtorch.wrap_tensor(fancy_cam_tensor)
                    self.fancy_cam_tensors.append(torch_fancy_cam_tensor)

                    for i in range(1,self.fancyTest_num):
                        cam_hndl = self.gym.create_camera_sensor(env,fancy_camera_props[i])
                        self.gym.set_camera_location(
                            cam_hndl, env, 
                            gymapi.Vec3(*self.fancyTest_pos[i]),
                            gymapi.Vec3(*self.fancyTest_tgt[i]))
                        fancy_cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, cam_hndl, gymapi.IMAGE_COLOR)
                        torch_fancy_cam_tensor = gymtorch.wrap_tensor(fancy_cam_tensor)
                        self.fancy_cam_tensors.append(torch_fancy_cam_tensor)

            props_shape = self.gym.get_actor_rigid_shape_properties(env, handle)
            props_shape[0].rolling_friction = 0.0
            props_shape[0].torsion_friction = 0.0
            props_shape[0].friction = 0.0
            props_shape[0].restitution = 0.0
            self.gym.set_actor_rigid_shape_properties(env, handle, props_shape)

            if self.test and self.cfg["env"]["path"]["showPath"]:
                self.gym.create_actor(env, path, path_pose, "path", i)

        #self.cam_tensors.append(torch_cam_tensor_rgb)
        #self.cam_tensors_torch = torch.cat(([torch.unsqueeze(cam,0) for cam in self.cam_tensors]),dim=0)

        self.mass = self.gym.get_actor_rigid_body_properties(env,handle)[0].mass

    def allocate_buffers(self):
        super().allocate_buffers()
        self.frames_in_contact = torch.zeros_like(self.progress_buf,device=self.device)

    def _get_images(self):
        if self.steps % self.update_freq and not self.test:
            return
        self.gym.fetch_results(self.sim,True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        if self.record and not self.resets % self.record_freq:
            img_rgb = self.torch_cam_tensor_rgb.view(self.rgb_h,self.rgb_w,4).cpu().numpy()
            self.frames_rgb.append([self.ax_rgb.imshow(img_rgb,animated=True)])

            if self.fancyTest:
                for i in range(self.fancyTest_num):
                    fancy_img = self.fancy_cam_tensors[i].view(self.fancyTest_h[i],self.fancyTest_w[i],4).cpu().numpy()
                    self.fancyTest_frames[i].append([self.fancyTest_axes[i].imshow(fancy_img,animated=True)])
       
        self.gym.end_access_image_tensors(self.sim)            

    def _normalize_image(self,img):
        return normalize_image(img,self.cam_max_range,self.cam_min_range)

    def _compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.targets[:], self.frames_in_contact[:] = compute_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions_cost_scale,
            self.pose_cost_scale,
            self.ang_cost_scale,
            self.max_episode_length,
            self.goal_radius,
            self.goal_bonus,
            self.contact_forces,
            self.contact_cost,
            self.contact_lim,
            self.contact_thresh,
            self.death_cost,
            self.path,
            self.targets,
            self.frames_in_contact,
            self.reward_scale,
            self.prev_actions,
            )

    def _compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        #self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.obs_buf[:] = compute_observations(
            self.root_tensor,
            self.path,
            self.targets,
            self.actions,
            self.num_assets,
            self.contact_forces,
            self.contact_lim
            )
        
        if self.record and not self.resets % self.record_freq:
            self._fill_fig()

    def reset_idx(self, env_ids):

        pose = torch.zeros_like(self.initial_root_state)
        vels = torch.cat((
            torch.zeros(len(env_ids),7,device=self.device), #pose,quat
            torch_rand_float(-0.1, 0.1, (len(env_ids),2), device=self.device), #vx,vy
            torch.zeros(len(env_ids),3,device=self.device), #vy,v_ang
            torch_rand_float(-0.5, 0.5, (len(env_ids),1), device=self.device)
            ),dim=-1)
        pose[self.num_assets*env_ids] = vels

        random_root = self.initial_root_state + pose

        env_ids_int32 = (self.num_assets*env_ids).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(random_root),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        #self.img_tensor[env_ids] = self.init_img_tensor[env_ids]

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        # apply  forces
        if self.prev_actions is None:
            self.prev_actions = self.actions = actions.to(self.device).clone() #v_d,th_d,heading_d
        else:
            self.prev_actions = self.actions.clone()
            self.actions = actions.to(self.device).clone() #v_d,th_d,heading_d

        ## TESTING
        #self.actions[:, 0] = 1
        #self.actions[:, 1] = 0
        #self.actions[:, 2] = 0 #-np.pi/2

        self.vel_dx = self.actions[:, 0]*torch.sin((np.pi/2)*self.actions[:, 1])
        sgn_vel_dx = torch.sign(self.vel_dx)
        self.vel_dy = self.actions[:, 0]*torch.cos((np.pi/2)*self.actions[:, 1])
        sgn_vel_dy = torch.sign(self.vel_dy)
        self.heading_d = (np.pi/2)*self.actions[:, 1].view(-1,1)
        heading_dx = torch.sin(self.heading_d)
        sgn_heading_dx = torch.sign(heading_dx)
        heading_dy = torch.cos(self.heading_d)
        sgn_heading_dy = torch.sign(heading_dy)

        self.forces = torch.zeros((self.num_envs*self.num_bodies, 3), device="cuda:0", dtype=torch.float)
        self.torques = torch.zeros((self.num_envs*self.num_bodies, 3), device="cuda:0", dtype=torch.float)

        self.prev_root[:] = self.root_tensor.clone()

        # lines
        n_lines= 2
        colors = torch.zeros(n_lines,3)
        colors[0,0] = 255 #motion heading
        colors[1,1] = 255 #camera heading
        for i in range(self.num_envs):
            verts = torch.zeros(n_lines*2,3)
            verts[::2,:2] = self.root_tensor[::self.num_assets,:2][i]
            verts[1,0] = self.root_tensor[::self.num_assets,0][i] + self.vel_dx[i] + sgn_vel_dx[i]*0.3048
            verts[1,1] = self.root_tensor[::self.num_assets,1][i] + self.vel_dy[i] + sgn_vel_dy[i]*0.3048
            verts[3,0] = self.root_tensor[::self.num_assets,0][i] + heading_dx[i] + sgn_heading_dx[i]*0.3048
            verts[3,1] = self.root_tensor[::self.num_assets,1][i] + heading_dy[i] + sgn_heading_dy[i]*0.3048
            self.gym.add_lines(self.viewer,self.envs[i],n_lines,verts,colors)

        err_vx = self.vel_dx - self.root_tensor[::self.num_assets, 7]
        err_vy = self.vel_dy - self.root_tensor[::self.num_assets, 8]
        err_th = self.heading_d - normalize_angle(get_euler_xyz(self.root_tensor[::self.num_assets,3:7])[2]).unsqueeze(-1) + np.pi/2
        err_th = (err_th + 2*np.pi) % (2*np.pi)
        err_th = torch.where(err_th > np.pi, err_th-2*np.pi,err_th)
        self.forces[::self.num_bodies, 0] = self.force_scale * torch.clamp(err_vx,-torch.ones_like(err_vx),torch.ones_like(err_vx))
        self.forces[::self.num_bodies, 1] = self.force_scale * torch.clamp(err_vy,-torch.ones_like(err_vy),torch.ones_like(err_vy))
        self.torques[::self.num_bodies, 2] = self.torque_scale * torch.clamp(err_th.view(-1),-torch.ones_like(err_th.view(-1)),torch.ones_like(err_th.view(-1)))

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            
            rgb_flag = False
            if self.record and 0 in env_ids:
                if len(self.frames_rgb) > 1:
                    if self.test:
                        rgb_vdir = "{dir}/test".format(dir=self.video_dir)
                        if not os.path.exists(rgb_vdir):
                            os.makedirs(rgb_vdir)
                        rgb_vname = "{dir}/rgb.mp4".format(dir=rgb_vdir)
                    
                        if self.fancyTest:
                            for i in range(self.fancyTest_num):
                                fancy_vname = "{dir}/rgb{n}.mp4".format(dir=rgb_vdir,n=i)
                                ani_fancy = animation.ArtistAnimation(self.fancyTest_figs[i],self.fancyTest_frames[i],interval=int(1000/self.fps),blit=True,repeat=False)
                                ani_fancy.save(fancy_vname,writer=self.writer)

                    else:
                        rgb_vdir = "{dir}/train_reset{reset}".format(dir=self.video_dir,reset=self.resets)
                        if not os.path.exists(rgb_vdir):
                            os.makedirs(rgb_vdir)
                        rgb_vname = "{dir}/rgb.mp4".format(dir=rgb_vdir)

                    ani_rgb = animation.ArtistAnimation(self.fig_rgb,self.frames_rgb,interval=int(1000/self.fps),blit=True,repeat=False)
                    ani_rgb.save(rgb_vname,writer=self.writer)
                    rgb_flag = True

                    self._set_fig()
                    
                if rgb_flag and self.record_once:
                    sys.exit("test simulation complete.")

                self.resets += 1

        self._get_images()
        self._compute_observations()
        self._compute_reward(self.actions)
        self.gym.clear_lines(self.viewer)

        self.steps += self.step_inc

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        cf = torch.zeros_like(self.net_cf[::self.num_bodies, :2])
        for i in range(self.control_freq_inv):
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.ENV_SPACE)
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)
            #self.gym.refresh_actor_root_state_tensor(self.sim)
            #self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

            cf += self.net_cf[::self.num_bodies, :2]

        self.contact_forces = cf/self.control_freq_inv #average forces

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict,self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def reset(self):
        self.obs_dict = super().reset()
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        return self.obs_dict

    def reset_done(self):
        self.obs_dict,done_env_ids = super().reset_done()
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

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
def normalize_image(img,cam_max_range,cam_min_range):
    # type: (Tensor, float, float) -> Tensor
    # [W,H,C]
    img[img < -cam_max_range] = -cam_max_range
    img[img > -cam_min_range] = -cam_min_range
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
    max_episode_length,
    goal_radius,
    goal_bonus,
    contact_forces,
    contact_cost_scale,
    contact_lim,
    contact_thresh,
    death_cost,
    path,
    targets,
    frames_in_contact,
    reward_scale,
    prev_action,
    ):
    # type: (Tensor, Tensor, Tensor, float, float, float, float, float, float, Tensor, float, float, float, float, Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    delta_action = torch.sum(torch.abs(obs_buf[:, 13:] - prev_action[:, :3]),dim=-1) #weight first two vals by pi?
    actions_reward = rew_func(delta_action,actions_cost_scale)

    pose = torch.linalg.norm(obs_buf[:, :2],dim=-1)
    pose_reward = rew_func(pose,pose_cost_scale)

    heading = torch.abs(obs_buf[:, 8])
    heading_reward = rew_func(heading,ang_cost_scale)

    contact = torch.linalg.norm(contact_forces,dim=-1)
    zero_contact_idx = torch.argwhere(contact < 1e-4).flatten()
    contact_idx = torch.argwhere(contact > 1e-4).flatten()
    frames_in_contact[zero_contact_idx] = 0
    frames_in_contact[contact_idx] += 1
    contact_cost = contact_cost_scale * frames_in_contact * frames_in_contact
    contact_reward = rew_func_tensor(contact,contact_cost)
    contact_reward = torch.where(contact > contact_thresh, torch.zeros_like(contact_reward), contact_reward)

    #disable rewards when in contact
    total_reward = contact_reward * ( actions_reward + pose_reward + heading_reward )
    total_reward /= 3 #normalize to 1
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

    ## TODO
    # - add power usage terms
    # - timeout bootstrapping (see ETH parallel walking paper)

    return total_reward, reset, targets, frames_in_contact

@torch.jit.script
def compute_observations(
    root_states,
    path,
    targets,
    actions,
    num_assets,
    contact_forces,
    contact_lim
    ):
    # type: (Tensor, Tensor, Tensor, Tensor, int, Tensor, float) -> Tensor

    # x y th vx vy vth gx gy gth ang2target gv fx fy fth

    position = root_states[::num_assets, 0:3]
    rotation = root_states[::num_assets, 3:7]
    velocity = root_states[::num_assets, 7:10]
    ang_velocity = root_states[::num_assets, 12]

    pose_error = position[:, :2] - path[targets]

    _,_,yaw = get_euler_xyz(rotation)
    heading = normalize_angle(yaw).unsqueeze(-1) - np.pi/2

    next_pose_error = position[:, :2] - path[targets+1]
    xaxis = torch.zeros_like(next_pose_error)
    xaxis[:,0] = 1

    #dot_prod = torch.einsum('ij,ij->i',position[:, :2],path[targets+1]) #look at next waypoint
    dot_prod = torch.einsum('ij,ij->i',next_pose_error,xaxis)
    dot_prod /= torch.linalg.norm(next_pose_error+1e-7,dim=-1)
    dot_prod /= torch.linalg.norm(xaxis+1e-7,dim=-1)
    dot_prod = torch.clamp(dot_prod,-1+1e-7,1-1e-7)
    target_angs = torch.arccos(dot_prod).view(-1,1) - np.pi/2
    heading_err = heading - target_angs #where camera should be pointed

    motion_heading = normalize_angle(torch.atan2(velocity[:, 0],velocity[:, 1])).unsqueeze(-1) #.view(-1,1)

    contact_norm = torch.linalg.norm(contact_forces,dim=-1).view((-1,1))
    contact_scaled = contact_forces / (contact_norm + 1e-7)
    contact_ratio = contact_norm / (contact_lim + 1e-7)

    obs = torch.cat((
        pose_error, next_pose_error, #0,1, 2,3
        velocity[:, :2], ang_velocity.view(-1,1), #4,5, 6
        heading, heading_err, #7, 8
        motion_heading, #9
        contact_scaled, contact_ratio, #10,11, 12
        actions[:, :2] #13,14
        ),dim=-1)

    return obs #[-1,15]