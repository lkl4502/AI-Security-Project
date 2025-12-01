import torch
import wandb
import numpy as np
import pytorch_lightning as pl

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


# --- Deep Learning Model (Lightning) ---
class DiabetesAutoEncoder(pl.LightningModule):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = self.config.model.hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.input_dim),
            nn.Sigmoid(),  # MinMaxScaler 전제
        )
        self.loss_fn = nn.MSELoss(reduction="none")

        # Metric 변수 추가
        self.auroc = BinaryAUROC()  # Val
        self.auprc = BinaryAveragePrecision()  # Val
        self.test_step_outputs = []  # Test

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        features = batch[0]
        predictions = self(features)
        loss = torch.mean(self.loss_fn(predictions, features))

        # Wandb logging
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
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
        features, labels = batch
        predictions = self(features)
        scores = torch.mean(self.loss_fn(predictions, features), dim=1)

        return {"scores": scores, "labels": labels}

    def on_test_epoch_end(self, outputs):
        all_scores = torch.cat([x["scores"] for x in outputs])
        all_labels = torch.cat([x["labels"] for x in outputs]).long()

        # 1. AUROC / AUPRC
        self.auroc.reset()
        self.auprc.reset()
        auroc = self.auroc(all_scores, all_labels)
        auprc = self.auprc(all_scores, all_labels)
        self.log(f"{self.config.model.name}/auroc", auroc)
        self.log(f"{self.config.model.name}/auprc", auprc)

        precisions, recalls, thresholds = precision_recall_curve(
            all_scores, all_labels, "binary"
        )
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

        if self.config.threshold.type == "F1-max":
            best_idx = torch.argmax(f1_scores)
            if best_idx >= len(thresholds):
                best_idx = -1
            best_threshold = thresholds[best_idx]

        elif self.config.threshold.type == "Top-k":
            k = int(len(all_scores) * self.config.threshold.contamination)
            top_val, _ = torch.topk(all_scores, k)
            best_threshold = top_val[-1]

        preds = (all_scores >= best_threshold).long()

        metrics = {
            f"{self.config.model.name}/accuracy": accuracy(
                preds, all_labels, task="binary"
            ),
            f"{self.config.model.name}/f1": f1_score(preds, all_labels, task="binary"),
            f"{self.config.model.name}/recall": recall(
                preds, all_labels, task="binary"
            ),
            f"{self.config.model.name}/precision": precision(
                preds, all_labels, task="binary"
            ),
            f"{self.config.model.name}/threshold": best_threshold,
        }

        self.log_dict(metrics)

    def configure_optimizers(self):
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


# --- Machine Learning Model Factory ---
def get_sklearn_model(config):
    name = config.model.name
    params = config.model
    seed = config.system.random_seed

    if name == "LR":
        return LogisticRegression(
            class_weight="balanced", random_state=seed, max_iter=params.max_iter
        )
    elif name == "DT":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed)
    elif name == "RF":
        return RandomForestClassifier(
            n_estimators=params.n_estimators,
            class_weight="balanced",
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
    model, test_features, test_targets, model_name, use_wandb=True
):
    """
    Scikit-Learn 모델을 평가하고 딥러닝 모델과 동일한 Metric을 WandB에 로깅하는 함수
    """
    print(f"\n--- Evaluating {model_name} ---")

    # 1. Score 및 Predict 추출
    if isinstance(model, IsolationForest):
        # Isolation Forest: -1(Anomaly), 1(Normal)
        raw_preds = model.predict(test_features)
        preds = np.where(raw_preds == -1, 1, 0)

        # Score Samples: 높을수록 Normal -> 부호 반전하여 Anomaly Score로 사용
        scores = -model.score_samples(test_features)

        # 시각화를 위한 0~1 정규화
        s_min, s_max = scores.min(), scores.max()
        probs = (scores - s_min) / (s_max - s_min + 1e-8)

    else:
        # Supervised Models (RF, LR, DT)
        preds = model.predict(test_features)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(test_features)[:, 1]  # Class 1 확률
            scores = probs
        else:
            probs = preds  # 확률 미지원시 0/1 값
            scores = preds

    # 2. Metric 계산 (Scikit-Learn 함수 사용)
    metrics = {
        f"{model_name}/accuracy": accuracy_score(test_targets, preds),
        f"{model_name}/f1": sk_f1(test_targets, preds),
        f"{model_name}/recall": sk_recall(test_targets, preds),
        f"{model_name}/precision": sk_precision(test_targets, preds),
        f"{model_name}/auroc": roc_auc_score(test_targets, scores),
        f"{model_name}/auprc": average_precision_score(test_targets, scores),
    }

    # 3. 콘솔 출력 및 WandB 로깅
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if use_wandb and wandb.run is not None:
        wandb.log(metrics)
