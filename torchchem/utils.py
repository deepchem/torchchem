"""
A utilities file with useful utilities.
"""
import numpy as np

def log(string, verbose=True):
  """Print string if verbose."""
  if verbose:
    print(string)

def from_one_hot(y, axis=1):
  """Transorms label vector from one-hot encoding.

    y: np.ndarray
      A vector of shape [n_samples, num_classes]
    """
  return np.argmax(y, axis=axis)
