# import necessary libraries

import matplotlib.pyplot as plt
import torch
from torch.optim import *

from sklearn import datasets
from sklearn.model_selection import train_test_split

from FCNet import FCNet
from plotSplines import plot_splines


def train(model, x, y, optimizer, criterion):
  
  model.train()

  optimizer.zero_grad()

  outputs = model(x)

  loss = criterion(outputs, y)

  loss.backward()

  optimizer.step()

  return loss, model.get_h1_h2()


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


## now let's use a real dataset

iris = datasets.load_iris()

## let's only use sepal length and sepal width from iris dataset and only 2 classes 0 and 1

x_train, x_test, y_train, y_test = train_test_split(iris.data[:100, :2], iris.target[:100], train_size = 81)

x = torch.from_numpy(x_train)
y = torch.from_numpy(y_train)

x_eval = torch.from_numpy(x_test)
y_eval = torch.from_numpy(y_test)

x = x.type(torch.float32)
x_eval = x_eval.type(torch.float32)


print("Train x: ", x.shape, "\nTrain y: ", y.shape)
print("Test x: ", x_eval.shape, "\nTest y: ", y_eval.shape)


num_epochs = 5
criterion = torch.nn.CrossEntropyLoss()
N = 9

optimizer = SGD(
  model.parameters(),
  lr=0.15,
  momentum=0.9,
  weight_decay=5e-4,
)

for epoch in range(num_epochs):
  
  loss, h = train(model, x, y, optimizer, criterion)
  print("Loss at epoch ", epoch, " = ", loss.item())
  plot_splines(h, N)


with torch.no_grad():
  acc = evaluate(model, x_eval, y_eval)
  print("Accuracy at model on test data = ", acc)

