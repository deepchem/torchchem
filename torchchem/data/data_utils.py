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

from rdkit import Chem
from torchchem.data.mol_graph import mol_to_graph
from torchchem.data.mol_dataset import MolDataset


def load_csv_dataset(path,
                     label_col,
                     data_dir=None,
                     smi_col='smiles',
                     explicit_H=False,
                     use_chirality=False,
                     use_molecular_attributes=False,
                     all_pair_features=False,
                     graph_distance=True):
  """
  Load dataset from a csv file

  Parameters
  ----------
  path : str
    Path to the csv data file.
  label_col : list of str
    Column names for labels/tasks.
  data_dir : str, optional
   Root directory to save the dataset.
   If None, a temp folder will be used.
  smi_col : str, optional
    Column name for SMILES.
  explicit_H : bool, optional
    If H should be explicitly included as nodes. The default is False.
  use_chirality : bool, optional
    If to add chirality features. The default is False.
  use_molecular_attributes : bool, optional
    If to add molecular properties calculated by RDkit. The default is False.
  all_pair_features : bool, optional
    If True, molecules are regarded as densely-connected graph.
    If False, only bonds are regarded as edges. The default is False.
  graph_distance : bool, optional
    If to use graph distance or catesian distance. The default is True.

  Returns
  -------
  dataset : torchchem.data.MolDataset

  """
  df = pd.read_csv(path)
  smiles = np.array(df[smi_col])
  labels = np.array(df[label_col])
  if data_dir is None:
    data_dir = tempfile.mkdtemp()
  dataset = MolDataset(root=data_dir)
  batch_graphs = []
  for i, (smi, l) in enumerate(zip(smiles, labels)):
    if i > 0 and i % 1000 == 0:
      print("Featurized %d molecules" % i)
      dataset.add_graph_batch(batch_graphs)
      batch_graphs = []
    g = mol_to_graph(Chem.MolFromSmiles(smi),
                     explicit_H=explicit_H,
                     use_chirality=use_chirality,
                     use_molecular_attributes=use_molecular_attributes,
                     all_pair_features=all_pair_features,
                     graph_distance=graph_distance)
    g.smi = smi
    w = (l==l) * 1
    y = copy.deepcopy(l)
    y[np.where(y != y)] = 0.
    g.y = t.from_numpy(y).long()
    g.w = t.from_numpy(w).float()
    batch_graphs.append(g)
  dataset.add_graph_batch(batch_graphs)
  return dataset

def load_sdf_dataset(path,
                     label_col,
                     data_dir=None,
                     explicit_H=True,
                     use_chirality=False,
                     use_molecular_attributes=False,
                     all_pair_features=True,
                     graph_distance=False):
  """
  Load dataset from a csv file

  Parameters
  ----------
  path : str
    Path to the sdf data file.
  label_col : list of str
    Column names for labels/tasks.
  data_dir : str, optional
   Root directory to save the dataset.
   If None, a temp folder will be used.
  explicit_H : bool, optional
    If H should be explicitly included as nodes. The default is False.
  use_chirality : bool, optional
    If to add chirality features. The default is False.
  use_molecular_attributes : bool, optional
    If to add molecular properties calculated by RDkit. The default is False.
  all_pair_features : bool, optional
    If True, molecules are regarded as densely-connected graph.
    If False, only bonds are regarded as edges. The default is False.
  graph_distance : bool, optional
    If to use graph distance or catesian distance. The default is True.

  Returns
  -------
  dataset : torchchem.data.MolDataset

  """
  label_csv = path + '.csv'
  label_df = pd.read_csv(label_csv)
  labels = np.array(label_df[label_col])
  
  MolSupp = Chem.SDMolSupplier(path, 
                               sanitize=False,
                               removeHs=not explicit_H,
                               strictParsing=False)
  dataset = MolDataset(root=data_dir)
  batch_graphs = []
  for i, (mol, l) in enumerate(zip(MolSupp, labels)):
    if i > 0 and i % 1000 == 0:
      print("Featurized %d molecules" % i)
      dataset.add_graph_batch(batch_graphs)
      batch_graphs = []
    if mol is None:
      print("W Error loading molecule %d" % i)
    g = mol_to_graph(mol,
                     explicit_H=explicit_H,
                     use_chirality=use_chirality,
                     use_molecular_attributes=use_molecular_attributes,
                     all_pair_features=all_pair_features,
                     graph_distance=graph_distance)
    try:
      g.smi = Chem.MolToSmiles(mol)
    except Exception as e:
      print(e)
      print("W Error generating SMILES for molecule %d" % i)
      g.smi  = ''
    w = (l==l) * 1
    y = copy.deepcopy(l)
    y[np.where(y != y)] = 0.
    g.y = t.from_numpy(y).long()
    g.w = t.from_numpy(w).float()
    batch_graphs.append(g)
  dataset.add_graph_batch(batch_graphs)
  return dataset
  
  
  