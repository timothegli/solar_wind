from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class FeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_new = X_df.copy()
        X_df_new = compute_rolling_std(X_df_new, 'Beta', '2h')
        return X_df_new


class FeatureReductor():

    def __init__(self, num_features):
        self.num_features = num_features
        self.selector = SelectKBest(mutual_info_classif, k=self.num_features)
    
    def fit(self, X_train, y_train):
        self.selector.fit(X_train, y_train)
    
    def transform(self, X):
        return self.selector.transform(X)
    
    def fit_transform(self, X_train, y_train):
        self.fit(X_train, y_train)
        return self.transform(X_train)


class Classifier(BaseEstimator):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        y_pred = self.model.predict_proba(X)
        y_smooth = pd.Series(y_pred[:, 1]).\
            rolling(12, min_periods=0, center=True).quantile(0.9)
        y_out = np.array([1-y_smooth, y_smooth]).T
        return y_out
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)


def get_estimator():

    feature_extractor = FeatureExtractor()

    feature_reductor = FeatureReductor(5)

    standard_scaler = StandardScaler()

    classifier = Classifier()

    pipe = make_pipeline(feature_extractor, feature_reductor, 
                         standard_scaler, classifier)
    return pipe


def compute_rolling_std(data, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    data : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling mean from
    time_indow : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = '_'.join([feature, time_window, 'std'])
    data[name] = data[feature].rolling(time_window, center=center).std()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
