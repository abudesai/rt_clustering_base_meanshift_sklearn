import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
from sklearn.cluster import MeanShift, estimate_bandwidth

warnings.filterwarnings("ignore")

model_fname = "model.save"

MODEL_NAME = "clustering_base_mean_shift"


class ClusteringModel:
    def __init__(self, bandwidth, verbose=False, **kwargs) -> None:
        self.bandwidth = bandwidth
        self.verbose = verbose

        self.model = self.build_model()

    def build_model(self):
        model = MeanShift(bin_seeding=True, bandwidth=self.bandwidth)
        return model

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    def evaluate(self, x_test):
        """Evaluate the model and return the loss and metrics"""
        raise NotImplementedError

    def save(self, model_path):
        joblib.dump(self, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path):
        clusterer = joblib.load(os.path.join(model_path, model_fname))
        return clusterer


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    model = ClusteringModel.load(model_path)
    return model


def get_data_based_model_params(data):
    bandwidth = estimate_bandwidth(data, quantile=0.05)
    return {"bandwidth": bandwidth}
