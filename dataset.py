import torch
import pandas as pd

from ucimlrepo import fetch_ucirepo
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class DiabetesDataset(Dataset):
    def __init__(self, config, is_train):
        super().__init__()

        self.config = config
        self.is_train = is_train
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.scaler = (
            MinMaxScaler()
            if config["model"]["name"] == "autoencoder"
            else StandardScaler()
        )

        print(">> Loading Data...")
        cdc_diabetes = fetch_ucirepo(id=self.config["data"]["uci_id"])
        X = cdc_diabetes.data.features
        y = cdc_diabetes.data.targets

        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        method = self.config["data"]["sampling_method"]
        ratio = self.config["data"]["sampling_ratio"]

        if method == "under":
            sampler = RandomUnderSampler(
                sampling_strategy=ratio,
                random_state=self.config["system"]["random_seed"],
            )
            X_res, y_res = sampler.fit_resample(X_scaled, y)
        elif method == "smote":
            sampler = SMOTE(
                sampling_strategy=ratio,
                random_state=self.config["system"]["random_seed"],
            )
            X_res, y_res = sampler.fit_resample(X_scaled, y)
        else:
            X_res, y_res = X_scaled, y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_res,
            y_res,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["system"]["random_seed"],
            stratify=y_res,
        )

        # DataFrame/Series 형태로 저장 (Scikit-learn용)
        # Deep Learning용 Tensor 변환은 train_dataloader에서 처리하거나 여기서 미리 변환

    def __len__(self):
        return len(self.X_train) if self.is_train else len(self.X_test)

    def __getitem__(self, index):
        if self.is_train:
            X = self.X_train.iloc[index].values.astype("float32")

            if isinstance(self.y_train, pd.DataFrame):
                y = self.y_train.iloc[index].values.astype("float32")
            else:
                y = float(self.y_train.iloc[index])
        else:
            X = self.X_test.iloc[index].values.astype("float32")

            if isinstance(self.y_test, pd.DataFrame):
                y = self.y_test.iloc[index].values.astype("float32")
            else:
                y = float(self.y_test.iloc[index])

        return torch.tensor(X), torch.tensor(y)
