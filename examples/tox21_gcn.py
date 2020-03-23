#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 06:50:13 2020

@author: zqwu
"""

import torch
import torch.nn as nn
import torchchem
import numpy as np
import os
from sklearn.metrics import roc_auc_score

# Settings
tox21_path = './data/tox21/tox21.csv'
tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
         'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
lr = 0.001
batch_size = 128
weight_decay = 0.001
n_epochs = 1000
gpu = False

# Load dataset
dataset = torchchem.data.load_csv_dataset(tox21_path, tasks)

# Split dataset
inds = np.arange(len(dataset))
np.random.seed(123)
np.random.shuffle(inds)
train_inds = inds[:int(0.8*len(dataset))]
valid_inds = inds[int(0.8*len(dataset)):]

train_dataset = dataset.index_select(list(train_inds))
valid_dataset = dataset.index_select(list(valid_inds))

# Initialize model
net = torchchem.models.GraphConvolutionNet(
    n_node_features=dataset.num_node_features,
    n_tasks=len(tasks),
    post_op=nn.Sigmoid())

model = torchchem.models.GraphConvolutionModel(
    net, 
    criterion=torchchem.models.WeightedBCEWithLogits(),
    lr=lr,
    weight_decay=weight_decay,
    gpu=gpu)

def evaluate(dataset, model):
  outputs = model.predict(dataset)
  ys = []
  ws = []
  for data in dataset:
    ys.append(data.y.cpu().data.numpy())
    ws.append(data.w.cpu().data.numpy())
  ys = np.stack(ys, 0)
  ws = np.stack(ws, 0)
  scores = []
  for i in range(len(tasks)):
    y_pred = outputs[:, i]
    y = ys[:, i]
    w = ws[:, i]
    scores.append(roc_auc_score(y[np.where(w > 0)], y_pred[np.where(w > 0)]))
  return scores

# Training and evaluation
for i in range(n_epochs):
  model.train_epoch(train_dataset, batch_size=batch_size, shuffle=False)
print(np.mean(evaluate(train_dataset, model)))
print(np.mean(evaluate(valid_dataset, model)))
