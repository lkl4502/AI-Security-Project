import os
import torch
import wandb
import numpy as np
import lightning.pytorch as pl

from torch import nn

# AutoEncoer 평가용
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.functional import (
    accuracy,
    f1_score,
    recall,
    precision,
    precision_recall_curve,
)

# Scikit-Learn -> ML 모델 및 Metric
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score as sk_f1,
    recall_score as sk_recall,
    precision_score as sk_precision,
    roc_auc_score,
    accuracy_score,
    average_precision_score,
)


# Deep Learning Model (Lightning)
class DiabetesAutoEncoder(pl.LightningModule):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = self.config.model.hidden_dim

        # encoder -> 입력 데이터를 저차원으로 압축
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.hidden_dim),
            nn.ReLU(),
        )

        # decoder -> 압축 데이터를 원래 차원으로 복원
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.input_dim),
            nn.Sigmoid(),  # MinMaxScaler 전제
        )

        # 재구성 오차 -> 입력과 복원한 데이터 간의 차이
        self.loss_fn = nn.MSELoss(reduction="none")

        # Metric 변수 추가
        self.auroc = BinaryAUROC()  # Validation 용
        self.auprc = BinaryAveragePrecision()  # Validation 용
        self.test_step_outputs = []  # Test 용

    def forward(self, x):
        """순전파: Encoder -> Decoder"""
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        """학습 단계: 정상 데이터만으로 재구성 학습"""
        features = batch[0]
        predictions = self(features)
        loss = torch.mean(self.loss_fn(predictions, features))

        # Wandb logging
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """검증 단계: 정상/비정상 데이터를 모두 사용하여 이상 탐지 성능 평가"""
        features, labels = batch
        predictions = self(features)

        scores = torch.mean(self.loss_fn(predictions, features), dim=1)
        loss = torch.mean(scores)

        # AUROC/AUPRC Metric 계산
        self.auroc(scores, labels)
        self.auprc(scores, labels)

        # Wandb logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auprc", self.auprc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """테스트 단계: 점수와 라벨 수집"""
        features, labels = batch
        predictions = self(features)
        scores = torch.mean(self.loss_fn(predictions, features), dim=1)
        self.test_step_outputs.append({"scores": scores, "labels": labels})

    def on_test_epoch_end(self):
        """테스트 에폭 종료 후 최종 평가 (Threshold 결정 및 지표 계산)"""
        outputs = self.test_step_outputs
        all_scores = torch.cat([x["scores"] for x in outputs])
        all_labels = torch.cat([x["labels"] for x in outputs]).long()

        # 1. AUROC / AUPRC
        self.auroc.reset()
        self.auprc.reset()
        auroc = self.auroc(all_scores, all_labels)
        auprc = self.auprc(all_scores, all_labels)
        self.log(f"{self.config.model.name}/test_auroc", auroc)
        self.log(f"{self.config.model.name}/test_auprc", auprc)

        precisions, recalls, thresholds = precision_recall_curve(
            all_scores, all_labels, "binary"
        )
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

        if self.config.threshold.type == "F1-max":  # f1이 최대인 지점 threshold 설정
            best_idx = torch.argmax(f1_scores)
            if best_idx >= len(thresholds):
                best_idx = -1
            best_threshold = thresholds[best_idx]

        elif self.config.threshold.type == "Top-k":  # 상위 k개 이상치
            k = int(len(all_scores) * self.config.threshold.contamination)
            top_val, _ = torch.topk(all_scores, k)
            best_threshold = top_val[-1]

        preds = (all_scores >= best_threshold).long()

        metrics = {
            f"{self.config.model.name}/test_accuracy": accuracy(
                preds, all_labels, task="binary"
            ),
            f"{self.config.model.name}/test_f1": f1_score(
                preds, all_labels, task="binary"
            ),
            f"{self.config.model.name}/test_recall": recall(
                preds, all_labels, task="binary"
            ),
            f"{self.config.model.name}/test_precision": precision(
                preds, all_labels, task="binary"
            ),
            f"{self.config.model.name}/test_threshold": best_threshold,
        }

        self.log_dict(metrics)

    def configure_optimizers(self):
        """옵티마이저 및 스케줄러 설정"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.model.lr)

        scheduler_config = {
            "scheduler": getattr(torch.optim.lr_scheduler, self.config.scheduler.type)(
                optimizer, **self.config.scheduler.params
            ),
            "monitor": self.config.scheduler.monitor,
            "interval": self.config.scheduler.interval,
            "frequency": self.config.scheduler.frequency,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


# Machine Learning Model
def get_sklearn_model(config):
    """설정 파일에 따라 Scikit-Learn 모델 인스턴스 반환"""
    name = config.model.name
    params = config.model
    seed = config.system.random_seed

    if name == "LR":
        return LogisticRegression(
            class_weight="balanced", random_state=seed, max_iter=params.max_iter
        )
    elif name == "DT":
        return DecisionTreeClassifier(
            class_weight="balanced", max_depth=params.max_depth, random_state=seed
        )
    elif name == "RF":
        return RandomForestClassifier(
            n_estimators=params.n_estimators,
            class_weight="balanced",
            max_depth=params.max_depth,
            random_state=seed,
            n_jobs=params.n_jobs,
        )
    elif name == "IF":
        return IsolationForest(
            contamination=params.contamination, random_state=seed, n_jobs=params.n_jobs
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def evaluate_and_log_sklearn(
    model, test_features, test_targets, model_name, save_dir, use_wandb=True
):
    """
    Scikit-Learn 모델 평가 및 WandB 로깅 함수
    DL 모델과 동일한 Metric을 사용하여 비교 가능하게 함
    """
    print(f"\n--- Evaluating {model_name} ---")

    # Score 및 Predict 추출
    if isinstance(model, IsolationForest):
        # Isolation Forest: -1(Anomaly), 1(Normal)
        raw_preds = model.predict(test_features)
        preds = np.where(raw_preds == -1, 1, 0)

        # Score Samples: 높을수록 Normal -> 부호 반전하여 Anomaly Score로 사용
        scores = -model.score_samples(test_features)
    else:
        # Supervised Models (RF, LR, DT)
        preds = model.predict(test_features)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(test_features)[:, 1]  # Class 1 확률
        else:
            scores = preds  # 확률 미지원시 0/1 값

    # Metric 계산 (Scikit-Learn 함수 사용)
    metrics = {
        f"{model_name}/test_accuracy": accuracy_score(test_targets, preds),
        f"{model_name}/test_f1": sk_f1(test_targets, preds),
        f"{model_name}/test_recall": sk_recall(test_targets, preds),
        f"{model_name}/test_precision": sk_precision(
            test_targets, preds, zero_division=0.0
        ),
        f"{model_name}/test_auroc": roc_auc_score(test_targets, scores),
        f"{model_name}/test_auprc": average_precision_score(test_targets, scores),
    }

    # 콘솔 출력 및 WandB 로깅
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if use_wandb and wandb.run is not None:
        wandb.log(metrics)

    txt_lines = []
    for k, v in metrics.items():
        txt_lines.append(f"{k.upper()} : {v:.4f}")

    txt_path = os.path.join(save_dir, "test_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
