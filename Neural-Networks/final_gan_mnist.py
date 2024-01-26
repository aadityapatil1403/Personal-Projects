# -*- coding: utf-8 -*-
"""Final_GAN_MNIST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1siSQyMnasdK_8QVC_kVDAg2tc-HHS-x7
"""

# Imports
import math
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Downloading data
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Creating a DataLoader to streamline process of loading and preprocess data
dl = DataLoader(dataset=train_ds,
                shuffle=True,
                batch_size=64)

image_batch = next(iter(dl))

# Define a function to display a batch of images
def display_images(images, n_cols=4, figsize=(12, 6)):
    # Set the style of the plots to 'ggplot' (a popular plotting style)
    plt.style.use('ggplot')

    # Calculate the number of images and the required number of rows for displaying
    n_images = len(images)
    n_rows = math.ceil(n_images / n_cols)

    # Create a figure object with the specified size
    plt.figure(figsize=figsize)

    # Loop over each image to display it
    for i in range(n_images):
        # Create a subplot for each image in the grid
        ax = plt.subplot(n_rows, n_cols, i + 1)

        # Extract the current image
        image = images[i]

        # Rearrange the image dimensions from (C x H x W) to (H x W x C) for display
        image = image.permute(1, 2, 0)

        # Choose the colormap: 'gray' for single channel images, 'viridis' for multi-channel
        cmap = 'gray' if image.shape[2] == 1 else plt.cm.viridis

        # Display the image with the chosen colormap
        ax.imshow(image, cmap=cmap)

        # Remove the x and y ticks for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust the layout for better spacing between images
    plt.tight_layout()

    # Render the plot
    plt.show()

# Display a batch of images in a grid with 8 columns
display_images(images=image_batch[0], n_cols=8)

# Defining the Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_features, out_features):
        # Call the initializer of the parent class (nn.Module)
        super().__init__()

        # Initialize input and output feature dimensions
        self.in_features = in_features
        self.out_features = out_features

        # Define the first fully connected layer
        self.fc1 = nn.Linear(in_features=in_features, out_features=128)
        # Define a LeakyReLU activation with a negative slope of 0.2
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)

        # Define the second fully connected layer
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        # Another LeakyReLU activation with the same negative slope
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)

        # Define the third fully connected layer
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        # A third LeakyReLU activation
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)

        # Define the final fully connected layer that outputs the classification
        self.fc4 = nn.Linear(in_features=32, out_features=out_features)
        # Dropout layer to reduce overfitting during training
        self.dropout = nn.Dropout(0.3)

    # Define the forward pass
    def forward(self, x):
        # Reshape the input batch for processing
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # Sequentially pass the input through the fully connected layers and activations
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.dropout(x)

        # Output the final classification from the last fully connected layer
        logit_out = self.fc4(x)

        return logit_out

# Defining Generator
# Define the Generator class as a subclass of nn.Module
class Generator(nn.Module):
    def __init__(self, in_features, out_features):
        # Initialize the parent class (nn.Module)
        super(Generator, self).__init__()

        # Store input and output feature dimensions
        self.in_features = in_features
        self.out_features = out_features

        # Define the first fully connected layer of the generator
        self.fc1 = nn.Linear(in_features=in_features, out_features=32)
        # Define a LeakyReLU activation function with a negative slope of 0.2
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)

        # Define the second fully connected layer
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        # Another LeakyReLU activation function
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)

        # Define the third fully connected layer
        self.fc3 = nn.Linear(in_features=64, out_features=128)
        # A third LeakyReLU activation function
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)

        # Define the final fully connected layer that outputs the generated image
        self.fc4 = nn.Linear(in_features=128, out_features=out_features)
        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(0.3)

        # Tanh activation function for the output layer
        self.tanh = nn.Tanh()

    # Define the forward pass
    def forward(self, x):
        # Sequentially pass the input through the layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc4(x)

        # Apply the Tanh function to normalize the output
        tanh_out = self.tanh(x)

        return tanh_out

# Function for calculating the loss for real data
def real_loss(predicted_outputs, loss_fn, device):
    # Get the batch size from the shape of the predicted outputs
    batch_size = predicted_outputs.shape[0]

    # Create a target tensor of ones, since the real data should ideally be classified as 1 (real)
    # The tensor is moved to the specified device (e.g., GPU or CPU)
    targets = torch.ones(batch_size).to(device)

    # Calculate the loss between the predicted outputs and the targets
    # The predicted_outputs are squeezed to match the target's dimensions
    real_loss = loss_fn(predicted_outputs.squeeze(), targets)

    # Return the calculated loss
    return real_loss

# Function for calculating the loss for fake data
def fake_loss(predicted_outputs, loss_fn, device):
    # Get the batch size from the shape of the predicted outputs
    batch_size = predicted_outputs.shape[0]

    # Create a target tensor of zeros, since the fake data should ideally be classified as 0 (fake)
    # The tensor is moved to the specified device
    targets = torch.zeros(batch_size).to(device)

    # Calculate the loss between the predicted outputs and the targets
    # The predicted_outputs are squeezed to match the target's dimensions
    fake_loss = loss_fn(predicted_outputs.squeeze(), targets)

    # Return the calculated loss
    return fake_loss

# Sample generation of latent vector
l_size = 100
l = np.random.uniform(-1, 1, size=(16, l_size))
plt.imshow(l, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

# Define the training function for the GAN
def train_minst_gan(d, g, d_optim, g_optim, loss_fn, dl, num_epochs, device, verbose=False):
    print(f'Starting the training on [{device}]...')

    # Preparing a fixed set of latent vectors to monitor the progress of the Generator
    l_size = 100  # Size of the latent vector
    fixed_l = np.random.uniform(-1, 1, size=(16, l_size))
    fixed_l = torch.from_numpy(fixed_l).float().to(device)
    fixed_samples = []  # To store generated samples for visualization
    d_losses = []  # To track the Discriminator's loss
    g_losses = []  # To track the Generator's loss

    # Move both the Discriminator and Generator models to the specified device
    d = d.to(device)
    g = g.to(device)

    # Start the training loop over all epochs
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]:')

        # Set both models to training mode
        d.train()
        g.train()

        # Initialize running loss for both models
        d_running_batch_loss = 0
        g_running_batch_loss = 0

        # Iterate over the data loader
        for curr_batch, (real_images, _) in enumerate(dl):
            real_images = real_images.to(device)  # Move real images to the device

            # --- Train the Discriminator ---
            d_optim.zero_grad()  # Zero out gradients for the Discriminator

            # Process real images
            real_images = (real_images * 2) - 1  # Normalize images to [-1, 1]
            d_real_logits_out = d(real_images)
            d_real_loss = real_loss(d_real_logits_out, loss_fn, device)

            # Process fake images
            with torch.no_grad():
                z = np.random.uniform(-1, 1, size=(dl.batch_size, l_size))
                z = torch.from_numpy(z).float().to(device)
                fake_images = g(z)  # Generate fake images
            d_fake_logits_out = d(fake_images)
            d_fake_loss = fake_loss(d_fake_logits_out, loss_fn, device)

            # Calculate total loss and backpropagate
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()
            d_running_batch_loss += d_loss

            # --- Train the Generator ---
            g_optim.zero_grad()  # Zero out gradients for the Generator

            # Generate fake images and compute loss
            z = np.random.uniform(-1, 1, size=(dl.batch_size, l_size))
            z = torch.from_numpy(z).float().to(device)
            fake_images = g(z)
            g_logits_out = d(fake_images)
            g_loss = real_loss(g_logits_out, loss_fn, device)

            # Backpropagate the loss and update the Generator
            g_loss.backward()
            g_optim.step()
            g_running_batch_loss += g_loss

            # Print training progress
            if curr_batch % 400 == 0 and verbose:
                print(f'\tBatch [{curr_batch:>4}/{len(dl):>4}] - d_batch_loss: {d_loss.item():.6f}\tg_batch_loss: {g_loss.item():.6f}')

        # Calculate and record epoch losses for both models
        d_epoch_loss = d_running_batch_loss.item()/len(dl)
        g_epoch_loss = g_running_batch_loss.item()/len(dl)
        d_losses.append(d_epoch_loss)
        g_losses.append(g_epoch_loss)

        # Print epoch losses
        print(f'epoch_d_loss: {d_epoch_loss:.6f} \tepoch_g_loss: {g_epoch_loss:.6f}')

        # Generate and store fixed samples for visualization
        g.eval()
        fixed_samples.append(g(fixed_l).detach().cpu())

    # Save the generated images for visualization
    with open('fixed_samples.pkl', 'wb') as f:
        pkl.dump(fixed_samples, f)

    return d_losses, g_losses

# Prepare and start the training

# Instantiate the Discriminator and Generator
d = Discriminator(in_features=784, out_features=1)  # Create a Discriminator
g = Generator(in_features=100, out_features=784)    # Create a Generator

# Instantiate optimizers for both models
d_optim = optim.Adam(d.parameters(), lr=0.002)  # Adam Optimizer for the Discriminator
g_optim = optim.Adam(g.parameters(), lr=0.002)  # Adam Optimizer for the Generator

# Instantiate the loss function
loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for GAN training

# Set up the device for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Start the training process
num_epochs = 100
# Train the GAN
d_losses, g_losses = train_minst_gan(d, g, d_optim, g_optim, loss_fn, dl, num_epochs, device, verbose=False)

# Define a function to display generated images for a specific epoch
def show_generated_images(epoch, n_cols=8):
    # Load the saved generated images from the training process
    with open('fixed_samples.pkl', 'rb') as f:
        saved_data = pkl.load(f)  # Load the pickle file containing saved images
    epoch_data = saved_data[epoch-1]  # Retrieve images from the specified epoch

    # Re-scale the images from the range of [-1, 1] back to [0, 1]
    # This is necessary because the Generator's output uses Tanh activation,
    # resulting in values in the range of [-1, 1]
    epoch_data = (epoch_data + 1) / 2

    # Reshape the data to the original image format
    # The shape is determined by the batch size and the dimensions of MNIST images (28x28 pixels)
    # For MNIST, the images are single-channel (grayscale), hence channel is set to 1
    batch_size, channel, height, width = len(epoch_data), 1, 28, 28
    image_batch = epoch_data.view(batch_size, channel, height, width)

    # Display the images using the previously defined display_images function
    # This will show the images in a grid with the specified number of columns and figure size
    display_images(images=image_batch, n_cols=n_cols, figsize=(12, 4))

# Display generated images for Epoch 1
show_generated_images(epoch=1, n_cols=8)

# Display generated images for Epoch 10
show_generated_images(epoch=10, n_cols=8)

# Display generated images for Epoch 50
show_generated_images(epoch=50, n_cols=8)

# Display generated images for Epoch 100
show_generated_images(epoch=100, n_cols=8)