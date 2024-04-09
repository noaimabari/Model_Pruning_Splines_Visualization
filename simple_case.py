# import necessary libraries

import matplotlib.pyplot as plt
import torch
from torch.optim import *

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from FCNet import FCNet
from plotSplines import plot_splines


def train(model, x, y, optimizer, criterion, callbacks = None):

  model.train()

  optimizer.zero_grad()

  outputs = model(x)

  loss = criterion(outputs, y)

  loss.backward()

  optimizer.step()

  if callbacks is not None:
        for callback in callbacks:
            callback()

  return loss


def evaluate(model, x, y):

  model.eval()
  
  num_samples = 0
  num_correct = 0

  outputs = model(x)

  outputs = outputs.argmax(dim=1)

  # Update metrics
  num_samples += y.size(0)
  num_correct += (outputs == y).sum()

  return (num_correct / num_samples * 100).item()




model = FCNet()
print(model)


## now let's create a handcrafted 2d dataset


# 1
data, labels = make_classification(n_samples=42000, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, class_sep=0.7, random_state=42)

# 2
# data, labels = make_circles(40000, random_state=42)

# Visualize dataset
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.Spectral)
plt.title("Binary Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8)

x = torch.from_numpy(x_train)
y = torch.from_numpy(y_train)

x_eval = torch.from_numpy(x_test)
y_eval = torch.from_numpy(y_test)

x = x.type(torch.float32)
x_eval = x_eval.type(torch.float32)

y = y.type(torch.int64)
y_eval = y_eval.type(torch.int64)

print("Train x: ", x.shape, "\nTrain y: ", y.shape)
print("Test x: ", x_eval.shape, "\nTest y: ", y_eval.shape)


# perform training

num_epochs = 50
criterion = torch.nn.CrossEntropyLoss()
model = FCNet()

optimizer = SGD(
  model.parameters(),
  lr=0.3,
  momentum=0.9,
  weight_decay=5e-4,
)

for epoch in range(num_epochs):
  loss = train(model, x, y, optimizer, criterion)
  print("Loss of model at epoch: ", epoch, "= ", loss.item())
  plot_splines(model)

with torch.no_grad():
  acc = evaluate(model, x_eval, y_eval)
  print("Model accuracy on test dataset = ", acc)