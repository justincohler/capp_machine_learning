"""
Neighbors is a basic ML pipeline implementation using a K-Nearest-Neighbors model.

@author: Justin Cohler
"""
from pipeline import Pipeline
from abc import ABCMeta
import pandas as pd
import sklearn

class Neighbors(Pipeline):
    """Implement ML pipeline using a K-Nearest-Neighbors model."""

    def __init__(self):
        """Set up k-nearest-neighbor specific globals."""
        super().__init__()

    def preprocess(self, data):
        """
        Return an updated df, filling missing values for all fields.

        (Uses mean to fill in missing values)
        """
        return data.fillna(data.mean())

    def discretize(self, data, field, bins=None, labels=None):
        """Return a discretized Series of the given field."""
        if not bins and not labels:
            series = pd.qcut(data[field], q=4)
        elif not labels and bins != None:
            series = pd.qcut(data[field], q=bins)
        elif not bins and labels != None:
            series = pd.qcut(data[field], q=len(labels), labels=labels)
        elif bins != len(labels):
            raise IndexError("Bin size and label length must be equal.")
        else:
            series = pd.qcut(data[field], q=bins, labels=labels)

        return series

    def build_classifier(self, data):
        """Return a built classifier specific to the implementation.

        (e.g. Logistic Regression, Decision Trees).
        """
        pass

    def evaluate_classifier(self, data):
        """Return evaluation for the implemented classifier."""
