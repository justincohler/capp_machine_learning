"""
Pipeline is a generic ML pipeline containing basic functions for cleaning, processing, and evaluating a model.

Current Implementations:
    - K-Nearest-Neighbors

@author: Justin Cohler
"""
from interface import Interface

class Pipeline(Interface):
    """
    Pipeline is a generic ML pipeline containing basic functions for cleaning, processing, and evaluating a model.

    Current Implementations:
        - K-Nearest-Neighbors
    """

    def __init__(self):
        """Set up generic vars."""
        pass

    def ingest(self, source):
        """Return a pandas dataframe of the data from a given source string."""
        pass

    def distribution(self, data):
        """Return the distribution in the dataframe."""
        pass

    def correlation(self, *fields):
        """Return the correlation matrix between the given fields."""
        pass

    def preprocess(self, data):
        """
        Return an updated df, filling missing values for all fields.

        (Uses mean to fill in missing values)
        """
        pass

    def discretize(self, data, field):
        """Return an updated dataframe with a discretized version of the given field."""
        pass

    def dummify(self, data, categorical):
        """Return an updated dataframe with binary/dummy fields from the given categorical field."""
        pass

    def build_classifier(self, data):
        """Return a built classifier specific to the implementation.

        (e.g. Logistic Regression, Decision Trees)
        """
        pass

    def evaluate_classifier(self, data):
        """Return evaluation for the implemented classifier."""
