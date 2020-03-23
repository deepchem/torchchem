#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 03:15:05 2020

@author: zqwu
"""

import numpy as np
import pandas as pd
import csv
import torch as t
import os
import pickle
import copy
import torch_geometric as tg
import tempfile

class MolDataset(tg.data.Dataset):
  def __init__(self,
               mol_graphs=[],
               data_batch_size=1000,
               cache_size=10000,
               root=None, 
               transform=None, 
               pre_transform=None,
               pre_filter=None):
    """
    Molecule dataset

    Parameters
    ----------
    mol_graphs : list of tg.data.Data, optional
      Molecular graphs
    data_batch_size : int, optional
      Number of graphs in each data batch file. The default is 1000.
    cache_size : int, optional
      Size of cache. The default is 10000.
    root : str, optional
      Root directory where the data batches are saved.
      If None, a temp folder will be used.
    transform : callable, optional
    pre_transform : callable, optional
    pre_filter : callable, optional

    """
    if root is None:
      root = tempfile.mkdtemp()
    root = os.path.abspath(root)
    super(MolDataset, self).__init__(root=root,
                                     transform=transform,
                                     pre_transform=pre_transform,
                                     pre_filter=pre_filter)
    self.file_mapping = {}
    self.cache = {}
    self.data_batch_size = data_batch_size
    self.cache_size = cache_size
    self.num_graphs = 0
    self.num_files = 0
    self.add_graph_batch(mol_graphs)
  
  @property
  def raw_file_names(self):
    """
    Returns
    -------
    list
      File paths of saved data batches.

    """
    return [os.path.join(self.root, "%d.pt" % i) \
            for i in range(self.num_files)]
  
  @property
  def current_file(self):
    """
    Returns
    -------
    str
      File path of the next data batch.

    """
    return os.path.join(self.root, "%d.pt" % self.num_files)
  
  def save(self):
    self.cache = {}
    with open(os.path.join(self.root, "meta.pkl"), 'wb') as f:
      pickle.dump(self, f)
    return
  
  @staticmethod
  def load(root):
    meta_path = os.path.join(root, "meta.pkl")
    assert os.path.exists(meta_path)
    mol_dataset = pickle.load(open(meta_path, 'rb'))
    assert os.path.abspath(mol_dataset.root) == os.path.abspath(root)
    return mol_dataset
    
  def add_graph_batch(self, gs):
    """
    Add a list of new graphs
    
    Parameters
    ----------
    gs : list of tg.data.Data
      Molecular graphs

    """
    batch_save = {}
    for g in gs:
      self.file_mapping[self.num_graphs] = self.current_file
      batch_save[self.num_graphs] = g
      self.num_graphs += 1
      if len(batch_save) > self.data_batch_size:
        t.save(batch_save, self.current_file)
        self.num_files += 1
        batch_save = {}
    if len(batch_save) > 0:
      t.save(batch_save, self.current_file)
      self.num_files += 1
    assert len(self.file_mapping) == self.num_graphs
    self.save()
    return
    
  def add_graph(self, g):
    """
    Add a single graph

    Parameters
    ----------
    g : tg.data.Data
      Molecular graph

    """
    
    p = self.raw_file_names[-1]
    batch_save = t.load(p)
    self.file_mapping[self.num_graphs] = p
    self.num_graphs += 1
    batch_save[self.num_graphs] = g
    t.save(batch_save, p)
    assert len(self.file_mapping) == self.num_graphs
    self.save()
    return
  
  def update_cache(self, file_path):
    """
    Add a batch of graphs into cache

    Parameters
    ----------
    file_path : str
      File path of the cached data batch.

    """
    new_dat = t.load(file_path)
    if len(self.cache) < self.cache_size:
      self.cache.update(new_dat)
    else:
      self.cache = {}
      self.cache.update(new_dat)
      
  def len(self):
    return self.num_graphs

  def get(self, idx):
    """
    Fetch the idx-th graph from cache

    Parameters
    ----------
    idx : int
      indice
      
    Returns
    -------
    tg.data.Data
      Molecular graph

    """
    if not idx in self.cache:
      self.update_cache(self.file_mapping[idx])
    return self.cache[idx]

  def reorder_with_indices(self, indices=None, new_root_dir=None):
    """
    Reorder the data batches according to new indices
    Usually used after shuffle
    
    Parameters
    ----------
    indices : list, optional
      If given, the new order of dataset.
      If None, use the default indices.
    new_root_dir : str, optional
      Root directory to save the reordered dataset.
      If None, a temp folder will be used.

    Returns
    -------
    new_dataset : MolDataset
      New dataset with reordered graphs

    """
    if indices is None:
      indices = self.indices()
    if new_root_dir is None:
      new_root_dir = tempfile.mkdtemp()
    assert new_root_dir != self.root

    new_dataset = MolDataset(mol_graphs=[],
                             data_batch_size=self.data_batch_size,
                             cache_size=self.cache_size,
                             root=new_root_dir, 
                             transform=self.transform, 
                             pre_transform=self.pre_transform,
                             pre_filter=self.pre_filter)
    batch_gs = []
    for i in indices:
      batch_gs.append(self.get(i))
      if len(batch_gs) > self.data_batch_size:
        new_dataset.add_graph_batch(batch_gs)
        batch_gs = []
    if len(batch_gs) > 0:
      new_dataset.add_graph_batch(batch_gs)
    return new_dataset
  
  