"""
This file implements the base model for torchchem classes
"""

import shutil
import torch
import time
import numpy as np
import tempfile
from torchchem.utils import log

class Model(object):
  """
  Abstract base class for different ML models.
  """

  def __init__(self,
               model_instance=None,
               model_dir=None,
               verbose=True,
               **kwargs):
    """Abstract class for all models.
    Parameters:
    -----------
    model_instance: object
      Wrapper around ScikitLearn/Keras/Tensorflow model object.
    model_dir: str
      Path to directory where model will be stored.
    """
    self.model_dir_is_temp = False
    if model_dir is not None:
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
      model_dir = tempfile.mkdtemp()
      self.model_dir_is_temp = True
    self.model_dir = model_dir
    self.model_instance = model_instance
    self.model_class = model_instance.__class__

    self.verbose = verbose

  def __del__(self):
    if 'model_dir_is_temp' in dir(self) and self.model_dir_is_temp:
      shutil.rmtree(self.model_dir)

  def fit_on_batch(self, X, y, w):
    """
    Updates existing model with new information.
    """
    raise NotImplementedError(
        "Each model is responsible for its own fit_on_batch method.")

  def predict_on_batch(self, X, **kwargs):
    """
    Makes predictions on given batch of new data.

    Parameters
    ----------
    X: np.ndarray
      Features
    """
    raise NotImplementedError(
        "Each model is responsible for its own predict_on_batch method.")

  def reload(self):
    """
    Reload trained model from disk.
    """
    raise NotImplementedError(
        "Each model is responsible for its own reload method.")

  def save(self):
    """Dispatcher function for saving.

    Each subclass is responsible for overriding this method.
    """
    raise NotImplementedError

  def fit(self, dataset, nb_epoch=10, batch_size=50, **kwargs):
    """
    Fits a model on data in a Dataset object.
    """
    raise NotImplementedError

  def predict(self, dataset, transformers=[], batch_size=None):
    """
    Uses self to make predictions on provided Dataset object.

    Returns:
      y_pred: numpy ndarray of shape (n_samples,)
    """
    raise NotImplementedError

  #def evaluate(self, dataset, metrics, transformers=[], per_task_metrics=False):
  #  """
  #  Evaluates the performance of this model on specified dataset.

  #  Parameters
  #  ----------
  #  dataset: dc.data.Dataset
  #    Dataset object.
  #  metric: deepchem.metrics.Metric
  #    Evaluation metric
  #  transformers: list
  #    List of deepchem.transformers.Transformer
  #  per_task_metrics: bool
  #    If True, return per-task scores.

  #  Returns
  #  -------
  #  dict
  #    Maps tasks to scores under metric.
  #  """
  #  raise NotImplementedError

  def get_task_type(self):
    """
    Currently models can only be classifiers or regressors.
    """
    raise NotImplementedError

  def get_num_tasks(self):
    """
    Get number of tasks.
    """
    raise NotImplementedError


class MultitaskModel(Model):

  def __init__(self,
               layer_sizes=[1000],
               weight_init_stddevs=[.02],
               bias_init_consts=[1.],
               penalty=0.0,
               penalty_type="l2",
               dropouts=[0.5],
               learning_rate=.001,
               momentum=.9,
               optimizer="adam",
               batch_size=50,
               pad_batches=False,
               verbose=True,
               seed=None,
               **kwargs):
    """Constructs the computational graph.

    This function constructs the computational graph for the model. It relies
    subclassed methods (build/cost) to construct specific graphs.

    Parameters
    ----------
    layer_sizes: list
      List of layer sizes.
    weight_init_stddevs: list
      List of standard deviations for weights (sampled from zero-mean
      gaussians). One for each layer.
    bias_init_consts: list
      List of bias initializations. One for each layer.
    penalty: float
      Amount of penalty (l2 or l1 applied)
    penalty_type: str
      Either "l2" or "l1"
    dropouts: list
      List of dropout amounts. One for each layer.
    learning_rate: float
      Learning rate for model.
    momentum: float
      Momentum. Only applied if optimizer=="momentum"
    optimizer: str
      Type of optimizer applied.
    batch_size: int
      Size of minibatches for training.GraphConv
    verbose: True 
      Perform logging.
    seed: int
      If not none, is used as random seed for tensorflow. 
    """
    # Save hyperparameters
    self.layer_sizes = layer_sizes
    self.weight_init_stddevs = weight_init_stddevs
    self.bias_init_consts = bias_init_consts
    self.penalty = penalty
    self.penalty_type = penalty_type
    self.dropouts = dropouts
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.optimizer = optimizer
    self.batch_size = batch_size
    self.pad_batches = pad_batches
    self.verbose = verbose
    self.seed = seed

    self.build()
    self.optimizer = self.get_training_op()

  def add_training_cost(self, outputs, labels, weights):
    weighted_costs = []  # weighted costs for each example
    for task in range(self.n_tasks):
      weighted_cost = self.cost(outputs[task], labels[:, task],
                                weights[:, task])
      weighted_costs.append(weighted_cost)
    loss = torch.cat(weighted_costs).sum()
    # weight decay
    if self.penalty > 0.0:
      for variable in self.regularizaed_variables:
        loss += self.penalty * 0.5 * variable.mul(variable).sum()
    return loss

  def get_training_op(self):
    """Get training op for applying gradients to variables.

    Subclasses that need to do anything fancy with gradients should override
    this method.

    Returns:
    An optimizer
    """
    if self.optimizer == "adam":
      train_op = torch.optim.Adam(self.trainables, lr=self.learning_rate)
    elif self.optimizer == 'adagrad':
      train_op = torch.optim.Adagrad(self.trainables, lr=self.learning_rate)
    elif self.optimizer == 'rmsprop':
      train_op = torch.optim.RMSprop(
          self.trainables, lr=self.learning_rate, momentum=self.momentum)
    elif self.optimizer == 'sgd':
      train_op = torch.optim.SGD(self.trainables, lr=self.learning_rate)
    else:
      raise NotImplementedError('Unsupported optimizer %s' % self.optimizer)
    return train_op

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          log_every_N_batches=50,
          checkpoint_interval=10,
          **kwargs):
    """Fit the model.

    Parameters
    ---------- 
    dataset: dc.data.Dataset
      Dataset object holding training data 
    nb_epoch: 10
      Number of training epochs.
    max_checkpoints_to_keep: int
      Maximum number of checkpoints to keep; older checkpoints will be deleted.
    log_every_N_batches: int
      Report every N batches. Useful for training on very large datasets,
      where epochs can take long time to finish.
    checkpoint_interval: int
      Frequency at which to write checkpoints, measured in epochs

    Raises
    ------
    AssertionError
      If model is not in training mode.
    """
    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    log("Training for %d epochs" % nb_epoch, self.verbose)
    for epoch in range(nb_epoch):
      avg_loss, n_batches = 0., 0
      for ind, (X_b, y_b, w_b, ids_b) in enumerate(
          # Turns out there are valid cases where we don't want pad-batches
          # on by default.
          dataset.iterbatches(self.batch_size, pad_batches=self.pad_batches)):
        if ind % log_every_N_batches == 0:
          log("On batch %d" % ind, self.verbose)
        # Run training op.
        self.optimizer.zero_grad()
        X_b_input = torch.autograd.Variable(torch.cuda.FloatTensor(X_b))
        y_b_input = torch.autograd.Variable(torch.cuda.FloatTensor(y_b))
        w_b_input = torch.autograd.Variable(torch.cuda.FloatTensor(w_b))
        outputs = self.forward(X_b_input, training=True)
        loss = self.add_training_cost(outputs, y_b_input, w_b_input)
        loss.backward()
        self.optimizer.step()
        avg_loss += loss
        n_batches += 1
      avg_loss = float(avg_loss.data.cpu().numpy()) / n_batches
      log('Ending epoch %d: Average loss %g' % (epoch, avg_loss), self.verbose)
    time2 = time.time()
    print("TIMING: model fitting took %0.3f s" % (time2 - time1), self.verbose)
    ############################################################## TIMING

  def predict(self, dataset, transformers=[]):
    """
    Uses self to make predictions on provided Dataset object.

    Returns:
      y_pred: numpy ndarray of shape (n_samples,)
    """
    y_preds = []
    n_tasks = self.n_tasks
    for (X_batch, _, _, ids_batch) in dataset.iterbatches(
        self.batch_size, deterministic=True):
      n_samples = len(X_batch)
      y_pred_batch = self.predict_on_batch(X_batch)
      assert y_pred_batch.shape == (n_samples, n_tasks)
      y_pred_batch = undo_transforms(y_pred_batch, transformers)
      y_preds.append(y_pred_batch)
    y_pred = np.vstack(y_preds)

    # The iterbatches does padding with zero-weight examples on the last batch.
    # Remove padded examples.
    n_samples = len(dataset)
    y_pred = np.reshape(y_pred, (n_samples, n_tasks))
    # Special case to handle singletasks.
    if n_tasks == 1:
      y_pred = np.reshape(y_pred, (n_samples,))
    return y_pred

  def predict_proba(self, dataset, transformers=[], n_classes=2):
    y_preds = []
    n_tasks = self.n_tasks
    for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(
        self.batch_size, deterministic=True):
      n_samples = len(X_batch)
      y_pred_batch = self.predict_proba_on_batch(X_batch)
      assert y_pred_batch.shape == (n_samples, n_tasks, n_classes)
      y_pred_batch = undo_transforms(y_pred_batch, transformers)
      y_preds.append(y_pred_batch)
    y_pred = np.vstack(y_preds)
    # The iterbatches does padding with zero-weight examples on the last batch.
    # Remove padded examples.
    n_samples = len(dataset)
    y_pred = y_pred[:n_samples]
    y_pred = np.reshape(y_pred, (n_samples, n_tasks, n_classes))
    return y_pred

  def build(self):
    raise NotImplementedError('Must be overridden by concrete subclass')

  def forward(self, X, training=False):
    raise NotImplementedError('Must be overridden by concrete subclass')

  def cost(self, logit, label, weight):
    raise NotImplementedError('Must be overridden by concrete subclass')

  def predict_on_batch(self, X_batch):
    raise NotImplementedError('Must be overridden by concrete subclass')

  def predict_proba_on_batch(self, X_batch):
    raise NotImplementedError('Must be overridden by concrete subclass')
