import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

from model import ElemNet

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

elements_tl = [
    'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
    'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
    'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'
]


def train(train_X, train_y):

  trainset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
  trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

  model = ElemNet(train_X.shape[1])
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  criterion = nn.MSELoss()
  model.train()

  for epoch in range(EPOCHS):

    train_loss = []

    for i, data in enumerate(trainloader):
      X, y = data
      X = X.to(device)
      y = y.to(device)
      model = model.to(device)

      optimizer.zero_grad()

      output = model(X.float()).reshape(-1)

      loss = criterion(output, y.float())
      train_loss.append(loss.item())

      if i % 200 == 0:
        print("Batch {}".format(i))

      loss.backward()
      optimizer.step()

    print("Loss in epoch {} is {}".format(epoch, np.mean(train_loss)))

  torch.save(model.state_dict(), 'model.pt')


def test(test_X, test_y):

  testset = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y))
  testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

  model = ElemNet(test_X.shape[1])
  model.load_state_dict(torch.load('model.pt'))
  criterion = nn.MSELoss()
  model.eval()

  test_loss = []
  for i, data in enumerate(testloader):
    X, y = data
    X = X.to(device)
    y = y.to(device)
    model = model.to(device)

    output = model(X.float()).reshape(-1)

    loss = criterion(output, y.float())
    test_loss.append(loss.item())
  print("test loss is {}".format(np.mean(test_loss)))


if __name__ == '__main__':
  df_train = pd.read_csv("data/train_set.csv")
  df_test = pd.read_csv("data/test_set.csv")

  labels = df_train.columns[-1]

  train_X = df_train[elements_tl].values
  train_y = df_train[labels].values
  test_X = df_test[elements_tl].values
  test_y = df_test[labels].values

  train(train_X, train_y)
  test(test_X, test_y)
