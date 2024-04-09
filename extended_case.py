import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import nn

from models import ConvNet
from plotSplines import plot_splines_for_conv2d


## load the dataset

# Define transforms
transform = {
    'train':
      transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,)),
      ]),
    'test':
      transforms.ToTensor(),
}

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform['train'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform['test'])
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)



## randomly displaying 4 samples from the dataset

# Function to show images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images[:4]))
# Print labels
print(' '.join('%5s' % labels[j].item() for j in range(4)))



# Initialize the model, loss function, and optimizer
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# image for plotting spline
dataiter = iter(testloader)
images, labels = next(dataiter)
image_to_study = images[0]

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        cnt = 2
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    
    # # plot spline
    # with torch.no_grad():
    #   model.eval()
    #   plot_splines_for_conv2d(model, image_to_study)

print('Finished Training')



# Evaluate the model on the test dataset
correct = 0
total = 0

with torch.no_grad():
  model.eval()
  for data in testloader:
      images, labels = data
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


# plot spline at the end of training for layer 1 and layer 2 activation maps

with torch.no_grad():
    model.eval()
    plot_splines_for_conv2d(model, image_to_study)