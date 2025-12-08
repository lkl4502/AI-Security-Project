import torch
import pandas as pd
import lightning.pytorch as pl

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

        # AutoEncoder는 0 ~ 1 사이 값(Sigmoid)을 출력하므로 MinMaxScaler 사용
        # 머신러닝 모델은 StandardScaler 사용
        self.scaler = MinMaxScaler() if config.model.name == "AE" else StandardScaler()

    def prepare_data(self):
        """데이터 다운로드 (최초 1회만 실행됨)"""
        fetch_ucirepo(id=self.config.data.uci_id)

    def setup(self, stage=None):
        """데이터 로드, 전처리, 분할, 스케일링 수행"""

        if hasattr(self, "train_features") and self.train_features is not None:
            print(">> Already Prepare Data...")
            return

        print(">> Loading Data...")
        seed = self.config.system.random_seed
        cdc_diabetes = fetch_ucirepo(id=self.config.data.uci_id)

        total_features = cdc_diabetes.data.features
        total_targets = cdc_diabetes.data.targets

        # 중복제거
        combined = total_features.copy()
        combined["target"] = total_targets

        combined = combined.drop_duplicates().reset_index(drop=True)

        total_targets = combined["target"]
        total_features = combined.drop(columns=["target"])

        # EDA를 통한 중요도 낮은 column 제거
        dropped_cols = ["Fruits", "AnyHealthcare", "NoDocbcCost", "Sex"]
        total_features.drop(columns=dropped_cols, inplace=True, errors="ignore")

        # Train / Val / Test Split
        # Test 먼저 분리
        target_ratio = self.config.data.get("target_ratio", None)
        if (
            target_ratio is not None
        ):  # target ratio가 설정되어 있으면 target에서 abnormal 데이터 비율 조정
            normal_features = total_features[total_targets == 0]
            abnormal_features = total_features[total_targets == 1]

            normal_targets = total_targets[total_targets == 0]
            abnormal_targets = total_targets[total_targets == 1]

            n_test = int(len(total_targets) * self.config.data.test_size)

            abnormal_n_test = int(n_test * target_ratio)
            normal_n_test = n_test - abnormal_n_test

            # test sampling
            test_normal_features = normal_features.sample(
                n=normal_n_test, random_state=seed
            )
            test_abnormal_features = abnormal_features.sample(
                n=abnormal_n_test, random_state=seed
            )

            test_normal_targets = normal_targets.loc[test_normal_features.index]
            test_abnormal_targets = abnormal_targets.loc[test_abnormal_features.index]

            # test set
            test_features = pd.concat([test_normal_features, test_abnormal_features])
            test_targets = pd.concat([test_normal_targets, test_abnormal_targets])

            # train_val set (나머지)
            train_val_features = total_features.drop(test_features.index)
            train_val_targets = total_targets.drop(test_targets.index)
        else:
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
        self.train_features = pd.DataFrame(  # 학습 데이터에만 !! fit_transform !!
            self.scaler.fit_transform(train_features), columns=train_features.columns
        )
        self.val_features = pd.DataFrame(
            self.scaler.transform(val_features), columns=val_features.columns
        )
        self.test_features = pd.DataFrame(
            self.scaler.transform(test_features), columns=test_features.columns
        )

        # reset index
        self.train_targets = train_targets.reset_index(drop=True)
        self.val_targets = val_targets.reset_index(drop=True)
        self.test_targets = test_targets.reset_index(drop=True)

        # Imbalance Handling (Sampling)
        method = self.config.data.sampling_method
        ratio = self.config.data.sampling_ratio

        if method == "under":
            # undersampling : 다수 클래스를 감소
            before_len = len(self.train_features)
            before_counts = self.train_targets.value_counts().to_dict()
            sampler = RandomUnderSampler(
                sampling_strategy=ratio,
                random_state=seed,
            )
            self.train_features, self.train_targets = sampler.fit_resample(
                self.train_features, self.train_targets
            )
            after_counts = pd.Series(self.train_targets).value_counts().to_dict()
            print(
                f">> [UnderSampling Applied]\n"
                f" - Sampling Strategy: {ratio}\n"
                f" - Before Sampling (class counts): {before_counts}\n"
                f" - After  Sampling (class counts): {after_counts}\n"
                f" - Train Size: {before_len} -> {len(self.train_features)}"
            )

        elif method == "smote":
            # oversampling : 소수 클래스 데이터를 합성하여 늘리는 방식
            before_len = len(self.train_features)
            before_counts = self.train_targets.value_counts().to_dict()
            sampler = SMOTE(
                sampling_strategy=ratio,
                random_state=seed,
            )
            self.train_features, self.train_targets = sampler.fit_resample(
                self.train_features, self.train_targets
            )
            after_counts = pd.Series(self.train_targets).value_counts().to_dict()
            print(
                f">> [SMOTE Applied]\n"
                f" - Sampling Strategy: {ratio}\n"
                f" - Before Sampling (class counts): {before_counts}\n"
                f" - After  Sampling (class counts): {after_counts}\n"
                f" - Train Size: {before_len} -> {len(self.train_features)}"
            )
        else:
            print(">> No Sampling Applied.")

        # dev_run일 때는 빠른 실행을 위해서 데이터 크기를 배치 사이즈로 줄여서 실행
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

    def get_full_train_data(self):
        """Cross Validation 등을 위해 Train과 Val 데이터를 합쳐서 반환"""
        train_val_features = pd.concat(
            [self.train_features, self.val_features], axis=0
        ).reset_index(drop=True)
        train_val_targets = pd.concat(
            [self.train_targets, self.val_targets], axis=0
        ).reset_index(drop=True)
        return train_val_features, train_val_targets.values.ravel()

    def get_normal_data(self):
        """AutoEncoder 학습용: 정상 데이터(Label 0)만 반환"""
        if isinstance(self.train_targets, pd.DataFrame):
            return self.train_features[self.train_targets.iloc[:, 0] == 0]
        return self.train_features[self.train_targets == 0]

    def get_sklearn_data(self):
        """Scikit-Learn 모델용 데이터 반환 (DataFrame, Numpy Array)"""
        return (
            self.train_features,
            self.test_features,
            self.train_targets.values.ravel(),
            self.test_targets.values.ravel(),
        )

    def train_dataloader(self):
        # Autoencoder는 비지도 학습(이상치 탐지)이므로 정상 데이터만으로 학습
        if self.config.model.name == "AE":
            data = self.get_normal_data().values
        else:
            data = self.train_features.values  # 일반 Classification DL 모델일 경우

        # AE는 입력 자체가 타겟이 되므로 features만 텐서로 변환
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
            torch.LongTensor(self.val_targets.values.ravel()),
        )

        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        # 테스트 데이터셋 로더 (Feature + Label)
        dataset = TensorDataset(
            torch.FloatTensor(self.test_features.values),
            torch.LongTensor(self.test_targets.values.ravel()),
        )

        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=False,
        )
