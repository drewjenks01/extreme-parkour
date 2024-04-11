from pathlib import Path
import isaacgym

from params_proto.hyper import Sweep

from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym_learn.ppo_cse import RunnerArgs
from go1_gym_learn.ppo_cse.actor_critic import AC_Args
from go1_gym_learn.ppo_cse.ppo import PPO_Args

from go1_gym_experiment.jaynes_sweeps.eipo_parkour.train import RunCfg

terrain = Cfg.terrain
commands = Cfg.commands
domain_rand = Cfg.domain_rand
env = Cfg.env
reward_scales = Cfg.reward_scales
rewards = Cfg.rewards
curriculum_thresholds = Cfg.curriculum_thresholds
normalization = Cfg.normalization
init_state = Cfg.init_state
control = Cfg.control

with Sweep(RunCfg, PPO_Args, RunnerArgs, AC_Args, terrain, commands, domain_rand, env, reward_scales, rewards, curriculum_thresholds, normalization, init_state, control) as sweep:
    # set up the sweep here
    # with sweep.zip:
    #     # idx = 0
    #     # energy_coeffs = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
    #     energy_coeffs = [0.001, 0.002, 0.005]
    #     # num_seeds = 3
    #     # reward_scales.energy_analytic = [energy_coeffs[idx] for _ in range(num_seeds)]
    #     # RunCfg.experiment_job_type = [f"teleport_{energy_coeffs[idx]}" for _ in range(num_seeds)]
    #     # reward_scales.energy_analytic = [0.0001, 0.0002, 0.0005]
    #     reward_scales.energy_analytic = energy_coeffs
    #     RunCfg.experiment_job_type = [f"{energy_coeff}" for energy_coeff in energy_coeffs]
    # reward_scales.lin_vel_z = 1.0
    RunCfg.experiment_group = "eipo_parkour"
    env.recording_width_px = 360
    env.recording_height_px = 240

sweep.save(f"{Path(__file__).stem}.jsonl")
print(f"sweep: {sweep}")