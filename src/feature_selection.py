from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np


def select_features(trainingSet: np.array, trainingLabels: np.array):
    x = trainingSet[:, :]
    y_h1n1 = trainingLabels[:, 0]
    y_seasonal = trainingLabels[:, 1]
    h1n1_features = _select_best_features(x, y_h1n1)
    seasonal_features = _select_best_features(x, y_seasonal)
    return h1n1_features, seasonal_features


def _select_best_features(x: np.array, y: np.array):
    sfs = SFS(CatBoostClassifier(n_estimators=100, verbose=False),
              k_features=x.shape[1],
              forward=True,
              floating=False,
              verbose=2,
              scoring='roc_auc',
              cv=5,
              n_jobs=-1)
    sfs = sfs.fit(x, y)
    sfdf = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
    print(sfdf)
    return _get_features_of_max_avg_score(sfdf)


def _get_features_of_max_avg_score(resultTable: pd.DataFrame):
    [maxValue], = np.where(resultTable.avg_score == resultTable.avg_score.max())
    return resultTable.loc[maxValue, 'feature_idx']
