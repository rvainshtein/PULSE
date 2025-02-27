import glob
import subprocess
import time
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import List, Union, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.syntax import Syntax
from os.path import dirname, abspath


@dataclass
class WandbConfig:
    entity: str = "phys_inversion"
    project: str = "eval_results_debug_pulse"


@dataclass
class PerturbationsConfig:
    gravity_z: float = -9.81
    list_gravity_z_mult: List[float] = field(
        default_factory=lambda: [0.25, 0.4, 0.6, 0.75, 1, 1.15, 1.3, 1.5, 1.6, 1.75, 2])
    friction: float = 1
    list_friction: List[float] = field(
        default_factory=lambda: [0.1 , 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82, 0.91, 1.])

    mass_multiplier: dict = field(default_factory=lambda: {})


@dataclass
class EvalConfig:
    checkpoint_paths: List[str] = field(default_factory=lambda: ["results/long_jump_pose/last.ckpt"])
    checkpoint_paths_glob: str = field(default="")  # Use this to override and glob paths dynamically
    gpu_ids: List[int] = field(default_factory=lambda: [2])
    more_options: str = field(default="++algo_type=PULSE ++prior=False")
    log_eval_results: bool = field(default=True)
    wandb: WandbConfig = WandbConfig()
    opt: List[str] = field(default_factory=lambda: ["eval"])
    # include_envs: List[str] = field(default_factory=lambda: ["pm_reach", "pm_direction_facing", "pm_longjump", "pm_direction", "pm_strike"])
    include_envs: List[str] = field(default_factory=lambda: ["pm_direction_facing"])
    num_envs: int = field(default=1024)
    games_per_env: int = field(default=1)
    use_perturbations: bool = field(default=False)
    perturbations: PerturbationsConfig = field(default_factory=lambda: PerturbationsConfig())
    record_dir: str = field(default="output/eval_videos")
    termination: bool = field(default=False)


env_name_dict = {"pm_direction_facing": "direction_facing", "pm_longjump": "long_jump", "pm_direction": "steering",
                 "pm_strike": "strike", "pm_reach": "reach"}


def build_command(config: DictConfig, checkpoint: Path, gpu_id: int, env_name: str):
    opt = config.opt
    more_options = config.more_options
    pulse_dir = dirname(dirname(abspath(__file__)))
    if config.use_perturbations is False:
        wandb_project = f"FINALLY__{env_name_dict[env_name]}_debug"
    else:
        wandb_project = f"PERTURB__{env_name_dict[env_name]}_debug"
    seed = Path(checkpoint).name[-1]
    cmd = []
    cmd.extend(
        [
            f" export PYTHONPATH={pulse_dir} &&",
            f" CUDA_VISIBLE_DEVICES={gpu_id} python phc/run_hydra.py headless=True",
            f" +exp={env_name}",
            f" +eval_checkpoint_path={str(Path(checkpoint) / 'Humanoid.pth')}",
            f" +wandb.wandb_entity={config.wandb.entity} ++wandb.project={wandb_project}",
            f" +opt=[{','.join(opt)}]",
            f" learning.params.config.player.games_num={config.num_envs * config.games_per_env}",
            f" env.num_envs={config.num_envs}",
            f" ++use_perturbations={config.use_perturbations}",
            f" ++seed={seed}",
            f" {more_options}",
        ]
    )

    if config.termination:
        cmd.append("++env.enable_height_termination=True")
    if config.log_eval_results:
        cmd.append("++algo.log_eval_results=True")
    if config.use_perturbations:
        cmd.append(f"++env.perturbations.friction={config.perturbations.friction}")
        cmd.append(f"++env.config.perturbations.friction={config.perturbations.friction}")
        cmd.append(f"++env.perturbations.gravity_z={config.perturbations.gravity_z}")
        cmd.append(f"++env.config.perturbations.gravity_z={config.perturbations.gravity_z}")

    return cmd


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config: DictConfig):
    console = Console()
    # Merge default config with CLI arguments
    default_cfg = OmegaConf.structured(EvalConfig())
    config = OmegaConf.merge(default_cfg, config)

    # Resolve checkpoint paths
    env_paths = glob.glob(config.checkpoint_paths_glob, recursive=False)

    gpu_ids = config.gpu_ids
    gpu_cycle = cycle(gpu_ids) if len(gpu_ids) > 1 else None
    processes = []
    if not config.use_perturbations:
        config.perturbations.friction = [1]
        config.perturbations.gravity_z_mult = [1]
    perturbations = []
    for friction in config.perturbations.list_friction:
        perturbations.append({"friction": friction, "gravity_z_mult": 1})
    for gravity_z_mult in config.perturbations.list_gravity_z_mult:
        perturbations.append({"friction": 1, "gravity_z_mult": gravity_z_mult})

    for perturbation in perturbations:
        config.perturbations.friction = round(perturbation["friction"], 2)
        config.perturbations.gravity_z = round(-9.81*perturbation["gravity_z_mult"], 2)
        for env_dir in env_paths:
            env_name = "_".join(Path(env_dir).resolve().name.split("_")[:-1])
            checkpoints_paths = glob.glob(str(Path(env_dir) / "*"), recursive=False)
            for checkpoint in checkpoints_paths:
                checkpoint = Path(checkpoint).resolve()
                gpu_id = next(gpu_cycle) if gpu_cycle else gpu_ids[0]
                cmd = build_command(config, checkpoint, gpu_id, env_name)
                if env_name not in config.include_envs:
                    continue
                cmd_print = Syntax(' '.join(cmd), "bash", theme="monokai", line_numbers=False, word_wrap=True)

                if len(gpu_ids) == 1:
                    console.print(f"[bold blue]Running sequentially on GPU {gpu_id}:[/bold blue]")
                    console.print(cmd_print)
                    subprocess.run(' '.join(cmd), shell=True)
                else:
                    while len(processes) >= len(gpu_ids):
                        processes = [p for p in processes if p.poll() is None]  # Remove finished processes
                        time.sleep(1)
                    console.print(f"[bold blue]Running on GPU {gpu_id}:[/bold blue]")
                    console.print(cmd_print)
                    processes.append(subprocess.Popen(' '.join(cmd), shell=True))

    # Wait for all remaining processes to finish
    for p in processes:
        p.wait()


if __name__ == '__main__':
    main()
