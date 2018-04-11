"""
Pipeline is a generic ML pipeline containing basic functions for cleaning, processing, and evaluating a model.

Current Implementations:
    - K-Nearest-Neighbors

@author: Justin Cohler
"""
from abc import ABC, abstractmethod
import pandas as pd
import sklearn

class Pipeline(ABC):
    """
    Pipeline is a generic ML pipeline containing basic functions for cleaning, processing, and evaluating a model.

    Current Implementations:
        - K-Nearest-Neighbors
    """

    def ingest(self, source):
        """Return a pandas dataframe of the data from a given source string."""
        return pd.read_csv(source)

    def distribution(self, data):
        """Return the distribution in the dataframe."""
        return data.describe()

    def correlation(self, *fields):
        """Return the correlation matrix between the given fields."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, data):
        """
        Return an updated df, filling missing values for all fields.

        (Uses mean to fill in missing values)
        """
        raise NotImplementedError

    @abstractmethod
    def discretize(self, data, field, bins=None, labels=None):
        """Return a discretized Series of the given field."""
        raise NotImplementedError

    def dummify(self, data, categorical):
        """Return an updated dataframe with binary/dummy fields from the given categorical field."""
        return data.join(pd.get_dummies(data[categorical]))

    @abstractmethod
    def build_classifier(self, data):
        """Return a built classifier specific to the implementation.

        (e.g. Logistic Regression, Decision Trees)
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_classifier(self, data):
        """Return evaluation for the implemented classifier."""
        raise NotImplementedError
