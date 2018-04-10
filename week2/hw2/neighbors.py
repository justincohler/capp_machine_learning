"""
Neighbors is a basic ML pipeline implementation using a K-Nearest-Neighbors model.

@author: Justin Cohler
"""
from interface import implements
from pipeline import Pipeline
import pandas as pd
import sklearn

class Neighbors(implements(Pipeline)):
    """Implement ML pipeline using a K-Nearest-Neighbors model."""

    def __init__(self):
        """Set up k-nearest-neighbor specific vars."""
        pass

    def ingest(self, source):
        """Return a pandas dataframe of the data from a given source string."""
        return pd.read_csv(source)

    def distribution(self, data):
        """Return the distribution in the dataframe."""
        return data.describe()

    def correlation(self, *fields):
        """Return the correlation matrix between the given fields."""
        pass

    def preprocess(self, data):
        """
        Return an updated df, filling missing values for all fields.

        (Uses mean to fill in missing values)
        """
        return data.fillna(data.mean())

    def discretize(self, data, field):
        """Return an updated dataframe with a discretized version of the given field."""
        pass

    def dummify(self, data, categorical):
        """Return an updated dataframe with binary/dummy fields from the given categorical field."""
        pass

    def build_classifier(self, data):
        """Return a built classifier specific to the implementation.

        (e.g. Logistic Regression, Decision Trees).
        """
        pass

    def evaluate_classifier(self, data):
        """Return evaluation for the implemented classifier."""
