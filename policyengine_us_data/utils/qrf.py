try:
    from quantile_forest import RandomForestQuantileRegressor
except ImportError:
    pass
import pandas as pd
import numpy as np
import pickle


class QRF:
    categorical_columns: list = None
    encoded_columns: list = None
    output_columns: list = None

    def __init__(self, seed=0, file_path=None):
        self.seed = seed

        if file_path is not None:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            self.seed = data["seed"]
            self.categorical_columns = data["categorical_columns"]
            self.encoded_columns = data["encoded_columns"]
            self.output_columns = data["output_columns"]
            self.qrf = data["qrf"]

    def fit(self, X, y, **qrf_kwargs):
        self.categorical_columns = X.select_dtypes(include=["object"]).columns
        X = pd.get_dummies(
            X, columns=self.categorical_columns, drop_first=True
        )
        self.encoded_columns = X.columns
        self.output_columns = y.columns
        self.qrf = RandomForestQuantileRegressor(
            random_state=self.seed, **qrf_kwargs
        )
        self.qrf.fit(X, y)

    def predict(self, X, count_samples=10, mean_quantile=0.5):
        X = pd.get_dummies(
            X, columns=self.categorical_columns, drop_first=True
        )
        X = X[self.encoded_columns]
        pred = self.qrf.predict(
            X, quantiles=list(np.linspace(0, 1, count_samples))
        )
        random_generator = np.random.default_rng(self.seed)
        a = mean_quantile / (1 - mean_quantile)
        input_quantiles = (
            random_generator.beta(a, 1, size=len(X)) * count_samples
        )
        input_quantiles = input_quantiles.astype(int)
        if len(pred.shape) == 2:
            predictions = pred[np.arange(len(pred)), input_quantiles]
        else:
            predictions = pred[np.arange(len(pred)), :, input_quantiles]
        return pd.DataFrame(predictions, columns=self.output_columns)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "seed": self.seed,
                    "categorical_columns": self.categorical_columns,
                    "encoded_columns": self.encoded_columns,
                    "output_columns": self.output_columns,
                    "qrf": self.qrf,
                },
                f,
            )
