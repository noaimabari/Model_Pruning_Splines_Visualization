import matplotlib.pyplot as plt
import torch

def plot_splines(h, N):
  """
  Function to plot splines in FCNet
  input: h (tuple) - output of layer 1 and layer 2
         N (int) - square root of number of datapoints 
  """
  L = 2
  
  grid = torch.meshgrid(torch.linspace(-L, L, N), 
                        torch.linspace(- L, L, N))

  plt.figure(figsize=(12,4))

  plt.subplot(1, 3, 1)
  for k in range(6):
      plt.contour(grid[0].numpy(), grid[1].numpy(), h[0][:, k].detach().numpy().reshape((N, N)), levels=[0], colors='b')
  plt.xticks([])
  plt.yticks([])
  plt.title('layer1')

  plt.subplot(1, 3, 2)
  for k in range(2):
      plt.contour(grid[0].numpy(), grid[1].numpy(), h[1][:, k].detach().numpy().reshape((N, N)), levels=[0], colors='r')
  plt.xticks([])
  plt.yticks([])
  plt.title('layer2')

  plt.subplot(1, 3, 3)
  for k in range(6):
      plt.contour(grid[0].numpy(), grid[1].numpy(), h[0][:, k].detach().numpy().reshape((N, N)), levels=[0], colors='b')
  for k in range(2):
      plt.contour(grid[0].numpy(), grid[1].numpy(), h[1][:, k].detach().numpy().reshape((N, N)), levels=[0], colors='r')
  plt.xticks([])
  plt.yticks([])
  plt.title('layer1+2')

  plt.show()
  
  plt.savefig('mini_test.png')



