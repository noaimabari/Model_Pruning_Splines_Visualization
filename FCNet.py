import torch

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
        self.h1 = self.linear1(x) 
        layer1_with_activ = self.activation(self.h1)
        self.h2 = self.linear2(layer1_with_activ)
        output = self.softmax(self.h2)
        return output

  def get_h1_h2(self):
    return self.h1, self.h2