import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from torch import nn


class Inferencer:
    def __init__(self, config, model, data_module):
        self.config = config
        self.model = model
        self.dm = data_module
        self.model_name = config["model"]["name"]

    def evaluate(self):
        print(f"\n--- Evaluating {self.model_name} ---")
        X_test = self.dm.X_test
        y_test = self.dm.y_test.values.ravel()

        y_pred = []
        y_prob = None  # ROC용

        # 1. Autoencoder Evaluation
        if self.model_name == "autoencoder":
            self.model.eval()
            self.model.freeze()

            # Reconstruction Error 계산
            X_tensor = torch.FloatTensor(X_test.values)
            recons = self.model(X_tensor)
            mse_loss = nn.MSELoss(reduction="none")
            losses = mse_loss(recons, X_tensor).mean(dim=1).numpy()

            # Threshold 설정 (Train 정상 데이터 기준)
            X_train_normal = torch.FloatTensor(self.dm.get_normal_data().values)
            train_recons = self.model(X_train_normal)
            train_losses = mse_loss(train_recons, X_train_normal).mean(dim=1).numpy()
            threshold = np.mean(train_losses) + 2 * np.std(train_losses)

            print(f"Anomaly Threshold: {threshold:.5f}")
            y_pred = [1 if l > threshold else 0 for l in losses]
            y_prob = losses  # Error 자체가 Anomaly Score

        # 2. Isolation Forest
        elif self.model_name == "isolation_forest":
            y_pred_raw = self.model.predict(X_test)
            y_pred = [1 if x == -1 else 0 for x in y_pred_raw]
            y_prob = -self.model.score_samples(X_test)  # Anomaly Score

        # 3. Supervised Classifiers
        else:
            y_pred = self.model.predict(X_test)
            if hasattr(self.model, "predict_proba"):
                y_prob = self.model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred

        # Metric 출력
        print(classification_report(y_test, y_pred))

        # 시각화 실행
        self.plot_results(y_test, y_pred, y_prob)

    def plot_results(self, y_true, y_pred, y_prob=None):
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix ({self.model_name})")
        plt.show()

        # ROC Curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(6, 5))
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (area = {roc_auc:.2f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {self.model_name}")
            plt.legend(loc="lower right")
            plt.show()
