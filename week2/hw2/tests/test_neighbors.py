"""
Unit Tests for the Neighbors Pipeline.

@author: Justin Cohler
"""
import unittest
from week2.hw2.src.neighbors import Neighbors
import os

class TestNeighbors(unittest.TestCase):
    """Unit Tests for the Neighbors Pipeline."""

    def setUp(self):
        """Set up global vars for tests."""
        self.neighbors = Neighbors()
        self.dirname = os.path.dirname(__file__)
        self.csv = os.path.join(self.dirname, '../data/credit-data.csv')
        self.data = self.neighbors.ingest(self.csv)

    def test_ingest(self):
        """Test the ingest() function."""
        self.data = self.neighbors.ingest(self.csv)
        self.assertIn('zipcode', self.data)

        print(self.data.tail())

    def test_distribution(self):
        """Test the distribution() function."""
        self.assertIsNotNone(self.data)
        distribution_data = self.neighbors.distribution(self.data)

        self.assertIsNotNone(distribution_data)

    def test_preprocess(self):
        """Test the preprocess() function."""
        nullcheck = self.data['MonthlyIncome'].isnull()
        self.assertIn(True, nullcheck)

        data = self.data
        data = self.neighbors.preprocess(data)

        nullcheck = data['MonthlyIncome'].isnull()
        self.assertEqual(0, len(nullcheck[nullcheck==True]))

    def test_discretize(self):
        """Test the discretize() function."""
        series1 = self.neighbors.discretize(self.data, "DebtRatio")
        series2 = self.neighbors.discretize(self.data, "DebtRatio", labels=['excellent', 'good', 'fair', 'poor'])
        series3 = self.neighbors.discretize(self.data, "DebtRatio", 4, ['excellent', 'good', 'fair', 'poor'])

        self.assertEqual(len(series1), len(series2))
        self.assertEqual(len(series1), len(series3))
        self.assertEqual(len(series1), len(self.data['DebtRatio']))

        try:
            series4 = self.neighbors.discretize(self.data, "DebtRatio", 5, ['excellent', 'good', 'fair', 'poor'])
        except IndexError as e:
            print(str(e))
            self.assertIn("Bin size and label length must be equal", e.args[0])

    def test_dummify(self):
        """Test the dummify() function."""
        data = self.data
        data['DebtClassification'] = self.neighbors.discretize(self.data, "DebtRatio", labels=['High Debt', 'Above Average Debt', 'Below Average Debt', 'Low Debt'])
        data = self.neighbors.dummify(data, 'DebtClassification')
        self.assertIn('High Debt', data)

    def test_classify(self):
        """Test the classify() function."""
        data = self.data
        data['DebtClassification'] = self.neighbors.discretize(self.data, "DebtRatio", labels=['High Debt', 'Above Average Debt', 'Below Average Debt', 'Low Debt'])
        data = self.neighbors.dummify(data, 'DebtClassification')

        features = ['High Debt', 'Above Average Debt', 'Below Average Debt', 'Low Debt']
        target = 'SeriousDlqin2yrs'
        kwargs = {"n_neighbors": 3}
        classifier, _, _= self.neighbors.classify(data, features, target, **kwargs)

        self.assertIsNotNone(classifier)

    def test_evaluate_classifier_smallk(self):
        """Test evaluate_classifier on a small k."""
        data = self.data
        data = self.neighbors.preprocess(data)
        data['DebtClassification'] = self.neighbors.discretize(self.data, "DebtRatio", labels=['High Debt', 'Above Average Debt', 'Below Average Debt', 'Low Debt'])
        data = self.neighbors.dummify(data, 'DebtClassification')

        features = ['High Debt', 'Above Average Debt', 'Below Average Debt', 'Low Debt']
        target = 'SeriousDlqin2yrs'
        kwargs = {"n_neighbors": 3}
        classifier, test_features, test_target = self.neighbors.classify(data, features, target, **kwargs)
        prediction = self.neighbors.predict(classifier, test_features)
        evaluation = self.neighbors.evaluate_classifier(prediction, test_target)

        print("Accuracy score for {} neighbors: {}".format(3, evaluation))

    @unittest.skip
    def test_classify_cross_validate(self):
        """Test the classify_cross_validate() function for a range of neighborsets."""
        data = self.data
        data = self.neighbors.preprocess(data)
        data['DebtClassification'] = self.neighbors.discretize(self.data, "DebtRatio", labels=['High Debt', 'Above Average Debt', 'Below Average Debt', 'Low Debt'])
        data = self.neighbors.dummify(data, 'DebtClassification')

        features = ['High Debt', 'Above Average Debt', 'Below Average Debt', 'Low Debt']
        target = 'SeriousDlqin2yrs'
        kwargs = {"folds": 10, "k_range": [5, 10, 25, 50, 100]}

        self.neighbors.classify_cross_validate(data, features, target, **kwargs)


if __name__ == '__main__':
    unittest.main()
