import torch
import pandas as pd
import pytorch_lightning as pl

from ucimlrepo import fetch_ucirepo
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class DiabetesDataModule(pl.LightningDataModule):
    def __init__(self, config, dev_run):
        super().__init__()
        self.config = config
        self.dev_run = dev_run
        self.scaler = (
            MinMaxScaler() if config.model.name == "autoencoder" else StandardScaler()
        )

    def prepare_data(self):
        # 데이터 다운로드 (최초 1회만 실행됨)
        fetch_ucirepo(id=self.config.data.uci_id)

    def setup(self):
        print(">> Loading Data...")
        cdc_diabetes = fetch_ucirepo(id=self.config.data.uci_id)

        total_features = cdc_diabetes.data.features
        total_targets = cdc_diabetes.data.targets

        # Train/Val/Test Split
        # Test 먼저 분리
        train_val_features, test_features, train_val_targets, test_targets = (
            train_test_split(
                total_features,
                total_targets,
                test_size=self.config.data.test_size,
                random_state=seed,
                stratify=total_targets,
            )
        )

        # Val 분리 및 Val 비율 재계산
        val_ratio = self.config.data.val_size / (1 - self.config.data.test_size)
        train_features, val_features, train_targets, val_targets = train_test_split(
            train_val_features,
            train_val_targets,
            test_size=val_ratio,
            random_state=seed,
            stratify=train_val_targets,
        )

        # Scaling
        self.train_features = pd.DataFrame(
            self.scaler.fit_transform(train_features), columns=train_features.columns
        )
        self.val_features = pd.DataFrame(
            self.scaler.transform(val_features), columns=val_features.columns
        )
        self.test_features = pd.DataFrame(
            self.scaler.transform(test_features), columns=test_features.columns
        )

        self.train_targets = train_targets.reset_index(drop=True)
        self.val_targets = val_targets.reset_index(drop=True)
        self.test_targets = test_targets.reset_index(drop=True)

        # Imbalance Handling (Sampling)
        method = self.config.data.sampling_method
        ratio = self.config.data.sampling_ratio
        seed = self.config.system.random_seed

        if method == "under":
            before_len = len(self.train_features)
            sampler = RandomUnderSampler(
                sampling_strategy=ratio,
                random_state=seed,
            )
            self.train_features, self.train_targets = sampler.fit_resample(
                self.train_features, self.train_targets
            )
            print(
                f">> UnderSampling Applied. Train Size: {before_len} -> {len(self.train_features)}"
            )

        elif method == "smote":
            before_len = len(self.train_features)
            sampler = SMOTE(
                sampling_strategy=ratio,
                random_state=seed,
            )
            self.train_features, self.train_targets = sampler.fit_resample(
                self.train_features, self.train_targets
            )
            print(
                f">> SMOTE Applied. Train Size: {before_len} -> {len(self.train_features)}"
            )
        else:
            print(">> No Sampling Applied.")

        if self.dev_run:
            idx = self.config.data.batch_size
            self.train_features = self.train_features.iloc[:idx]
            self.val_features = self.val_features.iloc[:idx]
            self.test_features = self.test_features.iloc[:idx]
            self.train_targets = self.train_targets.iloc[:idx]
            self.val_targets = self.val_targets.iloc[:idx]
            self.test_targets = self.test_targets.iloc[:idx]

        # DataFrame/Series 형태로 저장 (Scikit-learn용)
        # Deep Learning용 Tensor 변환은 train_dataloader에서 처리하거나 여기서 미리 변환

    def get_sklearn_data(self):
        """Scikit-Learn 모델용 데이터 반환"""
        return (
            self.train_features,
            self.test_features,
            self.train_targets.values.ravel(),
            self.test_targets.values.ravel(),
        )

    def train_dataloader(self):
        # Autoencoder 학습용
        if self.config.model.name == "AE":
            data = self.train_features[self.train_targets.iloc[:, 0] == 0].values
        else:
            data = self.train_features.values  # 일반 Classification DL 모델일 경우

        dataset = TensorDataset(torch.FloatTensor(data))
        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )

    def val_dataloader(self):
        # train_dataloader와 다른 이유
        # validation 단계에서 성능 평가를 위해 Label과 Anomaly 데이터도 필요하기 때문
        dataset = TensorDataset(
            torch.FloatTensor(self.val_features.values),
            torch.FloatTensor(self.val_targets.values),
        )

        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        dataset = TensorDataset(
            torch.FloatTensor(self.test_features.values),
            torch.FloatTensor(self.test_targets.values),
        )

        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=False,
        )
