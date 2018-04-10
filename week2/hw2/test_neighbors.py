"""
Unit Tests for the Neighbors Pipeline.

@author: Justin Cohler
"""
import unittest
from neighbors import Neighbors
class TestNeighbors(unittest.TestCase):
    """Unit Tests for the Neighbors Pipeline."""

    def setUp(self):
        """Set up global vars for tests."""
        self.neighbors = Neighbors()
        self.csv = r'C:\Users\Justin Cohler\workspaces\capp30254_ML\week2\hw2\credit-data.csv'
        self.data = self.neighbors.ingest(self.csv)

    def test_ingest(self):
        """Test the ingest() function."""
        self.data = self.neighbors.ingest(self.csv)
        self.assertIn('zipcode', self.data)

        print(self.data.tail())

    def test_distribution(self):
        """Test the distribution() function."""
        self.assertIsNotNone(self.data)
        distribution = self.neighbors.distribution(self.data)

        print(distribution)

    def test_preprocess(self):
        """Test the preprocess() function."""
        nullcheck = self.data['MonthlyIncome'].isnull()
        self.assertIn(True, nullcheck)

        data = self.data
        data = self.neighbors.preprocess(data)

        nullcheck = data['MonthlyIncome'].isnull()
        self.assertEqual(0, len(nullcheck[nullcheck==True]))

if __name__ == '__main__':
    unittest.main()
