"""
A utilities file with useful utilities.
"""
import numpy as np

def from_one_hot(y, axis=1):
  """Transorms label vector from one-hot encoding.

    y: np.ndarray
      A vector of shape [n_samples, num_classes]
    """
  return np.argmax(y, axis=axis)
