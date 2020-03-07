"""
Tests for molecular attention transformer.
"""
import unittest
from torchchem.models.transformer import make_model


class TestMAT(unittest.TestCase):

  def test_mat(self):
    """Simple test that initializes and fits a MAT."""
    model = make_model(5)

    
