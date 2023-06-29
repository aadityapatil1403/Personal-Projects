# Original file is located at: https://colab.research.google.com/drive/1Hn0lxkIj4GT_JcnEO4M8jmNp2E-Ii8-E


import torch
from torch import nn

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

#random seed allows identical replication / used for NN weights
torch.manual_seed(111)

#ensures code is run on GPU if available
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#preparing training data

#defining a transform function to convert data into a tensor; used when loading data
#changes the range of coefficients from 0-1 to -1-1; originally, coefficients = 0 as image backgrounds are black
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])

#load training data
train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
)

#creating DataLoader to shuffle data and create training samples for NN
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

#plotting training data
real_samples, mnist_labels = next(iter(train_loader))
for i in range(16):
  subplot = plt.subplot(4, 4, i+1)
  plt.imshow(real_samples[i].reshape(28,28), cmap="gray_r")
  plt.xticks([])
  plt.yticks([])

#creating Discriminator class

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        #defining first 3 hidden layers using ReLU activation; dropout used to reduce overfitting
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        #defining ouput layer using Sigmoid activation
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

  #defining forward pass of NN
  def forward(self, x):
  #converting original input (32x1x28x28) to (32x784); each line represents coefficients of a distinct image
    x = x.view(x.size(0), 784)
    output = self.model(x)
    return output

#instantiating Discriminator object and sending to GPU (if available)
discriminator = Discriminator().to(device=device)

#creating Generator class

class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        #defining first 3 hidden layers using ReLU activation
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 784),
        #defining output layer using Tanh activation to ensure output is from -1 to 1
        nn.Tanh(),
    )

    #defining forward pass of NN
  def forward(self, x):
    output = self.model(x)
    output = output.view(x.size(0), 1, 28, 28)
    return output

#instantiating Generator object
generator = Generator().to(device=device)

#defining training parameters

learning_rate = 0.0001
num_epochs = 50
#using binary cross-entropy loss function
loss_fn = nn.BCELoss()

#defining training optimizers by implementing Adam's algorithm
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

#training loop

for epoch in range(num_epochs):
  for n, (real_samples, mnist_labels) in enumerate(train_loader):
    #setting up data to train discriminator
    real_samples = real_samples.to(device=device)
    #attaching a label of '1' to the real data
    real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
    #attaching a random label for latent space data, then feeding into generator
    latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
    generated_samples = generator(latent_space_samples)
    #attaching a label of '0' to the generated data
    generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)
    #concatenating real and generated data into single tensor; will later be fed into discriminator
    all_samples = torch.cat((real_samples, generated_samples))
    all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

    #training discrminator

    #clearing gradients before each step
    discriminator.zero_grad()
    #feeding concatenated data into discrminator
    result_discriminator = discriminator(all_samples)
    #calculating loss
    discriminator_loss = loss_fn(result_discriminator, all_samples_labels)
    #calculating gradient and updating weights
    discriminator_loss.backward()
    #updating weights of discriminator; single optimization step that reevaluates the loss
    discriminator_optimizer.step()

    #setting up data to train generator
    latent_space_samples = torch.randn((batch_size, 100)).to(device=device)

    #training generator

    #clearing gradients before each step
    generator.zero_grad()
    generated_samples = generator(latent_space_samples)
    #storing result of entire model
    result_discriminator_generated = discriminator(generated_samples)
    #calculating loss
    generator_loss = loss_fn(result_discriminator_generated, real_samples_labels)
    generator_loss.backward()
    generator_optimizer.step()

    #displaying loss
    if n == batch_size - 1:
      print(f"Epoch: {epoch} Discriminator Loss: {discriminator_loss}")
      print(f"Epoch: {epoch} Generator Loss: {generator_loss}")

#generate samples
latent_space_samples = torch.randn(batch_size, 100).to(device=device)
generated_samples = generator(latent_space_samples)

#plot generated samples
generated_samples = generated_samples.cpu().detach()
for i in range(16):
  subplot = plt.subplot(4, 4, i+1)
  plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
  plt.xticks([])
  plt.yticks([])

