import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union, Dict

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
    checkpoints: List[List[str]] = field(default_factory=lambda: [[]])
    more_options: str = field(default="")
    wandb: WandbConfig = WandbConfig()
    opts: List[str] = field(default_factory=lambda: ["eval"])
    num_envs: int = field(default=1024)
    games_per_env: int = field(default=1)


def build_command(config: DictConfig, env_name, checkpoint: List[Path]):
    opts = config.opts
    more_options = config.more_options
    checkpoint_string = "\',\'".join(checkpoint)
    checkpoint_string = f"[\'{checkpoint_string}\']"
    cmd = (
        f" export PYTHONPATH=/home/stav/dev/PULSE &&"
        f" python phc/run_hydra.py headless=True"
        f" +exp=pm_{env_name}"
        f" env.models={checkpoint_string}"
        f" +wandb.wandb_entity={config.wandb.entity} +wandb.wandb_project={config.wandb.project}"
        f" +opt=[{','.join(opts)}]"
        f" learning.params.config.player.games_num={config.num_envs * config.games_per_env}"
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
    for env_name, checkpoints in zip(config.envs, config.checkpoints):
        cmd = build_command(config, env_name, checkpoints)

        cmd_print = Syntax(cmd, "bash", theme="monokai", line_numbers=False, word_wrap=True)
        console.print(f"[bold blue]Running sequentially:[/bold blue]")
        console.print(cmd_print)
        subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    main()
