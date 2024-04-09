import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
from scipy.signal import convolve2d

class FCNet(torch.nn.Module):
  
  """
  Class implementing a simple 2 layer fully connected neural network
  """

  def __init__(self, input_dim = 2, num_classes = 2, hidden_dim = 6):
        super(FCNet, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, num_classes)
        self.softmax = torch.nn.Softmax()

  def forward(self, x):
        h1 = self.linear1(x)
        layer1_with_activ = self.activation(h1)
        h2 = self.linear2(layer1_with_activ)
        output = self.softmax(h2)
        return output

  def get_h1_h2(self, x):
    h1 = self.linear1(x)
    h2 = self.linear2(self.activation(h1))

    return h1, h2
  



class ConvNet(nn.Module):
    
    """
    Class implementing a simple convolutional neural network with 1 fc layer 
    """

    def __init__(self):
      super(ConvNet, self).__init__()

      self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
      self.relu = nn.ReLU()
      self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

      self.conv2 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1)

      self.fc = nn.Linear(7*7*3, 10)  # Assuming input image size is 28x28 and we have 10 classes
      self.softmax = nn.Softmax()

    def forward(self, x):

      x = self.conv1(x) # conv layer 1 output
      x = self.relu(x)
      x = self.maxpool(x)

      x = self.conv2(x) # conv layer 2 output

      x = self.relu(x)
      x = self.maxpool(x)

      x = x.view(x.size(0), -1) # flatten the output for fc layer

      x = self.fc(x)  # fc layer output
      x = self.softmax(x) # apply softmax to get class probabilities

      return x

    def get_h1_h2(self, x):
      h1 = self.conv1(x)
      h2 = self.conv2(self.maxpool(self.relu(h1)))

      return h1, h2