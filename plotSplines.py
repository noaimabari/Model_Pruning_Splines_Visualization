import matplotlib.pyplot as plt
import torch

def plot_splines(model):
  
  """
  Function to plot splines

  """
  L = 2
  N = 200
  grid = torch.meshgrid(torch.linspace(-L, L, N),
                  torch.linspace(- L, L, N))
                  
  x = torch.stack([grid[0].reshape(-1), grid[1].reshape(-1)], 1)
  h = model.get_h1_h2(x)

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

  plt.savefig("mini_test.png")
