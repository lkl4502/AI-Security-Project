import os
import yaml
import wandb
import torch
import joblib
import lightning.pytorch as pl

from addict import Dict
from dotenv import load_dotenv
from dataset import DiabetesDataModule
from trainer import TrainerWrapper

# from inferencer import Inferencer
from argparse import ArgumentParser


def load_config(path):
    with open(path) as file:
        config_dict = yaml.safe_load(file)
    return Dict(config_dict)


def main(args):
    # Config 로드
    config = load_config(args.config)
    print(f"Loaded Config for model: {config.model.name}")

    # Wandb logging 설정
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        entity=config.wandb.team_name,
        project=config.wandb.project_name,
        name=config.wandb.experiment_name,
        config={k: v for k, v in config.items() if k != "wandb"},
        mode="disabled" if args.dev_run else None,
    )

    pl.seed_everything(config.system.random_seed, workers=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False  # 입력 크기 고정이면 최적화 알고리즘
        torch.backends.cudnn.enabled = True  # cuDNN 활성화, GPU 가속

    # Dataset 준비
    dm = DiabetesDataModule(config, args.dev_run)
    dm.setup()

    # Training
    trainer_wrapper = TrainerWrapper(config, dm)
    if trainer_wrapper.model_name == "AE":
        model, trainer = trainer_wrapper.dl_fit()
        trainer_wrapper.test(trainer=trainer)
    else:
        model_path = trainer_wrapper.ml_cv_fit()
        model = joblib.load(model_path)
        trainer_wrapper.test(model=model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="",
        help="config file (.yml) containing the hyper-parameters for training. ",
    )
    parser.add_argument("--dev_run", default=False, action="store_true")
    args = parser.parse_args()

    main(args)
