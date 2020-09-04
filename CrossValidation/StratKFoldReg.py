# Stratified-kfold for regression
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection


def create_folds(data):
    # We create a new column called kfold and fill it with -1
    data["kfold"] = -1
    # the nesxt step ins to randomize the rows of the data
    num_bins = int(np.floor(1+np.log2(len(data))))
    # bin targets
    data.iloc[:"bins"] = pd.cut(data["target"], bins=num_bins, labels=False)
    # initiate the kFold class from model selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t, v) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    # drio the bins column
    data = data.drop("bins", axis=1)

    # return the data
if __name__ == "__main__":
    # we create a sample dataset with 15000 samples
    # and 100 features and 1 target
    X, y = datasets.make_regression(
        n_samples=15000, n_features=100, n_targets=1
    )
    df = pd.Dataframe(X, columns=[f"f_{i}" for i in range9X.shape[1]])
    df.loc[:, "target"] = y
    # create createfold
    df = create_folds(df)
