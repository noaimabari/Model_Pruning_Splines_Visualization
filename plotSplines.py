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



def plot_splines_for_conv2d(model, image_to_study):
  
  """
  Function to plot the splines of each kernel in each layer of a conv net 
  """

  N = 200
  T = 28
  # image_to_study = np.random.randn(T, T) -> we are instead using a test image to study the splines

  direction = torch.randn(T, T)

  alpha, beta = torch.meshgrid(torch.linspace(0, 2, N), torch.linspace(-1, 1, N))
  alpha = alpha.reshape(-1)[:, None, None]
  beta = beta.reshape(-1)[:, None, None]

  x = alpha * image_to_study + beta * direction

  h = model.get_h1_h2(x.unsqueeze(1))

  plt.figure(figsize=(12, 4))

  plt.subplot(1, 2, 1)
  for k in range(4):
      for i in range(T):
          for j in range(T):
              # print(i, j)
              plt.contour(
                  alpha.reshape((N, N)),
                  beta.reshape((N, N)),
                  h[0][:, k, i, j].detach().numpy().reshape((N, N)),
                  levels=[0],
                  colors="b",
              )
  plt.xticks([])
  plt.yticks([])
  plt.title("layer1")

  plt.subplot(1, 2, 2)
  for k in range(3):
      for i in range(T//2):
          for j in range(T//2):
              plt.contour(
                  alpha.reshape((N, N)),
                  beta.reshape((N, N)),
                  h[1][:, k, i, j].detach().numpy().reshape((N, N)),
                  levels=[0],
                  colors="b",
              )
  plt.xticks([])
  plt.yticks([])
  plt.title("layer2")

  plt.savefig("mini_conv.png")

