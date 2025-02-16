import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.syntax import Syntax


@dataclass
class WandbConfig:
    entity: str = "phys_inversion"
    project: str = "eval_results"


@dataclass
class EvalConfig:
    envs: List[str] = field(default_factory=lambda: [])
    checkpoints: List[str] = field(default_factory=lambda: [])
    more_options: str = field(default="")
    wandb: WandbConfig = WandbConfig()
    opt: List[str] = field(default_factory=lambda: ["eval"])
    num_envs: int = field(default=128)
    games_per_env: int = field(default=1)


def build_command(config: DictConfig, env_name, checkpoint: Path):
    opt = config.opt
    more_options = config.more_options
    cmd = (
        f" export PYTHONPATH=/home/stav/dev/PULSE &&"
        f" python phc/run_hydra.py headless=True"
        f" +exp=pm_{env_name} "
        f" +eval_checkpoint_path={checkpoint}"
        f" +wandb.wandb_entity={config.wandb.entity} +wandb.wandb_project={config.wandb.project}"
        f" +opt=[{','.join(opt)}]"
        f" learning.params.config.player.games_num={config.num_envs * config.games_per_env}"
        f" env.num_envs={config.num_envs}"
        f" {more_options}"
    )
    return cmd


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config: DictConfig):
    console = Console()

    # Merge default config with CLI arguments
    default_cfg = OmegaConf.structured(EvalConfig())
    config = OmegaConf.merge(default_cfg, config)
    # Resolve checkpoint paths
    for env_name, checkpoint in zip(config.envs, config.checkpoints):
        cmd = build_command(config, env_name, checkpoint)

        cmd_print = Syntax(cmd, "bash", theme="monokai", line_numbers=False, word_wrap=True)
        console.print(f"[bold blue]Running sequentially:[/bold blue]")
        console.print(cmd_print)
        subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    main()
