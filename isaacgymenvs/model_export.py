import onnx
import isaacgym
import torch

from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path

from isaacgymenvs.utils.reformat import omegaconf_to_dict

@hydra.main(config_name="config", config_path="./cfg")
def export_model(cfg: DictConfig):
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs

    if cfg.checkpoint:
        onnx_fname = cfg.checkpoint.split(".")[0]+".onnx"
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    else:
        onnx_fname = "runs/{task}/nn/{task}.onnx".format(task=cfg.task.name)

    cfg.task.env.numEnvs = 1
    cfg.train.params.config.minibatch_size = 20
    cfg.headless = True
    cfg.task.viewer.captureVideo = False

    def create_env_thunk(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, cfg.task_name, cfg.task.env.numEnvs, 
            cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
            cfg.headless, cfg.multi_gpu, cfg.capture_video,
            cfg.force_render, cfg, **kwargs)
        return envs

    # register the rl-games adapter to use inside the runner
    vecenv.register("RLGPU",
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {
        "vecenv_type": "RLGPU",
        "env_creator": create_env_thunk,
    })

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    runner = Runner()
    runner.load(rlg_config_dict)

    agent = runner.create_player()
    agent.restore(cfg.checkpoint)
    net = agent.model

    from rl_games.algos_torch.models import BaseModelNetwork
    class ModelWrapper(BaseModelNetwork):
        def __init__(self, model, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self._model = model
            
        def forward(self,input_dict):
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            """
            just model export doesn"t work. Looks like onnx issue with torch distributions
            thats why we are exporting only neural network
            """
            return self._model.a2c_network(input_dict)
    
    obs = {}
    for k, v in agent.observation_space.spaces.items():
        obs[k] = v.sample()

    
    inputs = {
        "obs" : {
            "img" : torch.from_numpy(obs["img"]).unsqueeze(0).to(agent.device),
            "vec" : torch.from_numpy(obs["vec"]).unsqueeze(0).to(agent.device),
            },
        #"rnn_states" : None, "seq_length" : 1, "bptt_len" : 0
    }

    import rl_games.algos_torch.flatten as flatten
    with torch.no_grad():
        net = ModelWrapper(
                agent.model,
                obs_shape=agent.obs_shape,
                normalize_value=agent.normalize_value,
                normalize_input=agent.normalize_input,
                value_size=agent.value_size
                ).to(agent.device).eval()

        #traced = torch.jit.trace(net,inputs,check_trace=True)
        adapter = flatten.TracingAdapter(net, inputs,allow_non_tensor=True)
        traced = torch.jit.trace(adapter, adapter.flattened_inputs,check_trace=True)

        print(net(inputs))
        print(traced(*adapter.flattened_inputs))
        
    torch.onnx.export(traced, adapter.flattened_inputs, onnx_fname, verbose=True, input_names=["obs"], output_names=["mu","log_std", "value"])

    # check that the model is well formed
    #onnx_model = onnx.load(onnx_fname)
    #onnx.checker.check_model(onnx_model)

if __name__=="__main__":
    export_model()