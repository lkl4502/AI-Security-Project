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
    """YAML 설정 파일을 로드하여 딕셔너리 객체로 반환"""
    with open(path) as file:
        config_dict = yaml.safe_load(file)
    return Dict(config_dict)


def main(args):
    # Config 로드
    config = load_config(args.config)
    print(f"Loaded Config for model: {config.model.name}")

    # Wandb logging 설정
    load_dotenv()  # wandb login을 위한 환경변수 로드
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        entity=config.wandb.team_name,
        project=config.wandb.project_name,
        name=config.wandb.experiment_name,
        config={
            k: v for k, v in config.items() if k != "wandb"
        },  # wandb 설정 제외하고 기록
        mode="disabled" if args.dev_run else None,  # dev_run일 경우 logging 끄기
    )

    # seed 고정
    pl.seed_everything(config.system.random_seed, workers=True)

    # GPU 가속 설정
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

    # Dataset 준비
    dm = DiabetesDataModule(config, args.dev_run)
    dm.setup()

    # Training
    trainer_wrapper = TrainerWrapper(config, dm)
    if trainer_wrapper.model_name == "AE":  # 딥러닝 모델 (AE)
        model, trainer = trainer_wrapper.dl_fit()
        trainer_wrapper.test(trainer=trainer)
    else:  # 머신러닝 모델 (LR, DT, RF, IF)
        model_path = trainer_wrapper.ml_cv_fit()  # 교차 검증 학습
        model = joblib.load(model_path)
        trainer_wrapper.test(model=model)


if __name__ == "__main__":
    # 인자 파싱
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="",
        help="config file (.yml) containing the hyper-parameters for training. ",
    )
    parser.add_argument("--dev_run", default=False, action="store_true")
    args = parser.parse_args()

    main(args)
