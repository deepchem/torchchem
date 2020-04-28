import torch
import torch.nn as nn
import torch.nn.functional as F


class ElemNet(nn.Module):

  def __init__(self, input_size):
    super(ElemNet, self).__init__()

    self.linear_1 = nn.Linear(input_size, 1024)
    self.linear_2 = nn.Linear(1024, 1024)
    self.linear_3 = nn.Linear(1024, 1024)
    self.linear_4 = nn.Linear(1024, 1024)

    self.linear_5 = nn.Linear(1024, 512)
    self.linear_6 = nn.Linear(512, 512)
    self.linear_7 = nn.Linear(512, 512)

    self.linear_8 = nn.Linear(512, 256)
    self.linear_9 = nn.Linear(256, 256)
    self.linear_10 = nn.Linear(256, 256)

    self.linear_11 = nn.Linear(256, 128)
    self.linear_12 = nn.Linear(128, 128)
    self.linear_13 = nn.Linear(128, 128)

    self.linear_14 = nn.Linear(128, 64)
    self.linear_15 = nn.Linear(64, 64)

    self.linear_16 = nn.Linear(64, 32)

    self.linear_17 = nn.Linear(32, 1)

    self.activation = nn.ReLU()

  def forward(self, x):
    x = self.activation(self.linear_1(x))
    x = self.activation(self.linear_2(x))
    x = self.activation(self.linear_3(x))
    x = self.activation(self.linear_4(x))
    x = F.dropout(x, p=0.2)

    x = self.activation(self.linear_5(x))
    x = self.activation(self.linear_6(x))
    x = self.activation(self.linear_7(x))
    x = F.dropout(x, p=0.1)

    x = self.activation(self.linear_8(x))
    x = self.activation(self.linear_9(x))
    x = self.activation(self.linear_10(x))
    x = F.dropout(x, p=0.3)

    x = self.activation(self.linear_11(x))
    x = self.activation(self.linear_12(x))
    x = self.activation(self.linear_13(x))
    x = F.dropout(x, p=0.2)

    x = self.activation(self.linear_14(x))
    x = self.activation(self.linear_15(x))

    x = self.activation(self.linear_16(x))

    x = self.activation(self.linear_17(x))

    return x
