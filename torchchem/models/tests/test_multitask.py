import unittest
from torchchem.models import MultitaskClassification
from torchchem.models import MultitaskRegression


class TestMultitaskClassification(unittest.TestCase):

  def test_classification_init(self):
    """Test that classification init can happen."""
    n_tasks = 1
    n_features = 100
    model = MultitaskClassification(n_tasks, n_features)
    assert model

  def test_regression_init(self):
    """Test that regression init can happen."""
    n_tasks = 1
    n_features = 100
    model = MultitaskRegression(n_tasks, n_features)
    assert model
