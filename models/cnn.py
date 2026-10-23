import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
    self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
    self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
    self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
    self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)

    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(256*4*4, 512)
    self.fc2 = nn.Linear(512, 10)

    self.drop = nn.Dropout(0.1)

    self.nor1 = nn.BatchNorm2d(32)
    self.nor2 = nn.BatchNorm2d(64)
    self.nor3 = nn.BatchNorm2d(128)
    self.nor4 = nn.BatchNorm2d(128)
    self.nor5 = nn.BatchNorm2d(256)
    self.nor6 = nn.BatchNorm2d(256)

  def forward(self, X):
    X = F.relu(self.nor1(self.conv1(X)))
    X = F.relu(self.nor2(self.conv2(X)))
    X = self.pool(X)
    X = F.relu(self.nor3(self.conv3(X)))
    X = F.relu(self.nor4(self.conv4(X)))
    X = self.pool(X)
    X = F.relu(self.nor5(self.conv5(X)))
    X = F.relu(self.nor6(self.conv6(X)))
    X = self.pool(X)
    X = torch.flatten(X, start_dim = 1)
    X = self.fc2(self.drop(F.relu(self.fc1(X))))
    return X
