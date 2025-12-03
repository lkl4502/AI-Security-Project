import os
import json
import wandb
import joblib
import numpy as np
import os.path as osp
import lightning.pytorch as pl

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from model import DiabetesAutoEncoder, get_sklearn_model, evaluate_and_log_sklearn
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from sklearn.metrics import (
    f1_score as sk_f1,
    recall_score as sk_recall,
    precision_score as sk_precision,
    roc_auc_score,
    accuracy_score,
    average_precision_score,
)


class TrainerWrapper:
    def __init__(self, config, data_module):
        self.config = config
        self.dm = data_module
        self.model_name = config.model.name
        self.save_dir = osp.join(config.logging.save_dir, config.wandb.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def ml_cv_fit(self, n_splits=5):
        print(f"\n[Cross Validation] Start {n_splits}-Fold CV for {self.model_name}")
        train_features, train_targets = self.dm.get_full_train_data()

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.config.system.random_seed
        )

        metrics_list = {
            "f1": [],
            "recall": [],
            "precision": [],
            "auroc": [],
            "auprc": [],
            "accuracy": [],
        }

        base_model = get_sklearn_model(self.config)
        best_score = -1.0
        best_model_path = ""

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_features, train_targets)
        ):
            print(f">> Fold {fold+1}/{n_splits}...", end=" ")

            fold_train_features, fold_val_features = (
                train_features.iloc[train_idx],
                train_features.iloc[val_idx],
            )
            fold_train_targets, fold_val_targets = (
                train_targets[train_idx],
                train_targets[val_idx],
            )

            model = clone(base_model)

            if self.model_name == "IF":
                fold_train_normal_features = fold_train_features[
                    fold_train_targets == 0
                ]
                model.fit(fold_train_normal_features)
            else:
                model.fit(fold_train_features, fold_train_targets)

            fold_save_path = os.path.join(
                self.save_dir, f"{self.model_name}-fold{fold + 1}.pkl"
            )
            joblib.dump(model, fold_save_path)

            if self.model_name == "IF":
                scores = -model.score_samples(fold_val_features)
                preds = np.where(model.predict(fold_val_features) == -1, 1, 0)
            else:
                preds = model.predict(fold_val_features)
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(fold_val_features)[:, 1]
                else:
                    scores = preds

            current_score = average_precision_score(fold_val_targets, scores)

            metrics_list["f1"].append(sk_f1(fold_val_targets, preds))
            metrics_list["recall"].append(sk_recall(fold_val_targets, preds))
            metrics_list["precision"].append(
                sk_precision(fold_val_targets, preds, zero_division=0.0)
            )
            metrics_list["auroc"].append(roc_auc_score(fold_val_targets, scores))
            metrics_list["auprc"].append(current_score)
            metrics_list["accuracy"].append(accuracy_score(fold_val_targets, preds))

            print("Done.")

            if current_score > best_score:
                best_score = current_score
                best_model_path = fold_save_path
                print(f"   -> New Best Model found! (AUPRC: {best_score:.4f})")

        save_data = {
            "model_name": self.model_name,
            "n_splits": n_splits,
            "metrics": {},
        }

        txt_lines = []
        txt_lines.append(f"=== {n_splits}-Fold CV Results ({self.model_name}) ===")
        txt_lines.append(f"Experiment: {self.config.wandb.experiment_name}")
        txt_lines.append("-" * 30)

        # 5. 결과 요약 출력
        print(f"\n=== {n_splits}-Fold CV Results ({self.model_name}) ===")
        for k, v in metrics_list.items():
            mean_score = np.mean(v)
            std_score = np.std(v)

            log_str = f"{k.upper()}: {mean_score:.4f} (+/- {std_score:.4f})"
            print(log_str)

            txt_lines.append(log_str)

            save_data["metrics"][k] = {
                "mean": float(mean_score),
                "std": float(std_score),
                "raw_scores": [float(x) for x in v],  # 각 Fold의 점수도 보관
            }

            # WandB에 평균 점수 기록
            if wandb.run is not None:
                wandb.log(
                    {
                        f"{self.model_name}/cv_mean_{k}": mean_score,
                        f"{self.model_name}/cv_std_{k}": std_score,
                    }
                )

        json_path = os.path.join(self.save_dir, f"cv_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4)

        # 2. TXT 저장
        txt_path = os.path.join(self.save_dir, f"cv_results.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

        print(f"\n>> CV Results saved to:")
        print(f"   - {json_path}")
        print(f"   - {txt_path}")
        print(f"\n>> Best Model Path: {best_model_path} (Score: {best_score:.4f})")

        return best_model_path

    def dl_fit(self):
        print(f"Start Training: {self.model_name}")

        # 1. Deep Learning (Autoencoder)
        if self.model_name == "AE":
            input_dim = self.dm.train_features.shape[1]
            model = DiabetesAutoEncoder(self.config, input_dim)

            wandb_logger = WandbLogger()
            lr_monitor = LearningRateMonitor(logging_interval="step")

            checkpoint_callback = ModelCheckpoint(
                dirpath=self.save_dir,
                filename="{epoch:02d}-{val_auprc:.4f}",
                every_n_epochs=1,
                save_top_k=self.config.logging.save_top_k,  # 3
                monitor=self.config.logging.monitor,
                mode=self.config.logging.mode,
            )

            trainer = pl.Trainer(
                max_epochs=self.config.model.epochs,
                accelerator=self.config.system.device,
                devices=1,
                logger=wandb_logger,
                callbacks=[checkpoint_callback, lr_monitor],
                deterministic=True,
            )

            trainer.fit(model, self.dm)

            return model, trainer
        else:
            raise ValueError(
                f"Model '{self.model_name}' is a Machine Learning Model. "
                "Please use 'wrapper.ml_cv_fit()' for training and evaluation."
            )
        # 2. Machine Learning (Sklearn)
        # else:
        #     model = get_sklearn_model(self.config)

        #     if self.model_name == "IF":
        #         # Isolation Forest는 정상 데이터로 학습하는 것이 일반적
        #         print(">> Training Isolation Forest with Normal Data only...")
        #         train_features = self.dm.get_normal_data()
        #         model.fit(train_features)
        #     else:
        #         # Supervised Learning
        #         print(f">> Training {self.model_name} with Full Data...")
        #         train_features, _, train_targets, _ = self.dm.get_sklearn_data()
        #         model.fit(train_features, train_targets)

        #     # 모델 저장
        #     save_path = os.path.join(self.save_dir, f"{self.model_name}.pkl")
        #     joblib.dump(model, save_path)
        #     print(f"Model saved to {save_path}")
        #     return model, None

    def test(self, model=None, trainer=None):
        print(f"\n[Test Stage] Start Testing: {self.model_name}")

        # 1. DL
        if self.model_name == "AE":
            if trainer is None:
                print("Warning: Trainer is None. Attempting to create new trainer...")
                trainer = pl.Trainer(accelerator=self.config.system.device, devices=1)
            # 자동으로 가장 좋은 Checkpoint 로드
            trainer.test(datamodule=self.dm, ckpt_path="best")

        # 2. ML
        else:
            if model is None:
                raise ValueError(f"model is Empty.")

            # Test Set 평가
            _, test_features, _, test_targets = self.dm.get_sklearn_data()
            evaluate_and_log_sklearn(
                model,
                test_features,
                test_targets,
                model_name=self.model_name,
            )
