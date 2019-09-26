import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torchvision.models as models
import torchvision
from torchvision import transforms
from PIL import Image
from google.colab import drive
import random
##############

torch.manual_seed(0)

############

# Normalizing the whole dataset and finding std and average
# uncomment lines below to see the average and std for the whole data set

# transformations_full = transforms.Compose([transforms.ToTensor()])
# full_set = torchvision.datasets.ImageFolder("/content/drive/My Drive/dataset/full/",transform = transformations_full)
# full_loader = torch.utils.data.DataLoader(full_set, batch_size=5, shuffle=True)
# def calculate_img_stats_avg(loader):
#     mean = 0.
#     std = 0.
#     nb_samples = 0.
#     for imgs,_ in loader:
#         batch_samples = imgs.size(0)
#         imgs = imgs.view(batch_samples, imgs.size(1), -1)
#         mean += imgs.mean(2).sum(0)
#         std += imgs.std(2).sum(0)
#         nb_samples += batch_samples

#     mean /= nb_samples
#     std /= nb_samples
#     return mean,std
# print(calculate_img_stats_avg(full_loader))

############

def patch_generator(loader):
    i = random.randint(258,1532)##1920
    j = random.randint(258,2174)##2560
    imgs_new = loader[:,:,i:i+256,j:j+256]
    return imgs_new

############

# Specify transforms using torchvision.transforms as transforms
# library

transformations = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean=[0.6730, 0.6604, 0.6644], std=[0.1463, 0.1646, 0.1579])])

######### load the data via pytorch utility tool

train_set = torchvision.datasets.ImageFolder("/content/drive/My Drive/dataset/train/", transform = transformations)
valid_set = torchvision.datasets.ImageFolder("/content/drive/My Drive/dataset/valid/", transform = transformations)
test_set = torchvision.datasets.ImageFolder("/content/drive/My Drive/dataset/test/", transform = transformations)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=5, shuffle=True)

val_loader = torch.utils.data.DataLoader(valid_set, batch_size=5, shuffle=True)
test_loader =  torch.utils.data.DataLoader(test_set, batch_size=5, shuffle=True)

################################## make use of pretrained method(densenet)

model = models.densenet161(pretrained=True)
# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False

################################## Create new classifier for model using torch.nn as nn library

from torch import nn 
classifier_input = model.classifier.in_features
num_labels = 2
classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512 ),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))
# Replace default classifier with new classifier
model.classifier = classifier

###############

# Set the error function using torch.nn as nn library
criterion = nn.NLLLoss()
# Set the optimizer function using torch.optim as optim library
optimizer = torch.optim.Adam(model.classifier.parameters())

##############set up the gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move model to the device specified above
model.to(device)

#########3

############### training the model
epochs = 40
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for input1, labels in train_loader:
        for i in range (1,30):
            inputs = patch_generator(input1)
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear optimizers
            optimizer.zero_grad()
            # Forward pass
            output = model.forward(inputs)
            # Loss
            loss = criterion(output, labels)
            # Calculate gradients (backpropogation)
            loss.backward()
            # Adjust parameters based on gradients
            optimizer.step()
            # Add the loss to the training set's rnning loss
        train_loss += loss.item()*inputs.size(0)

          # Print the progress of our training
        counter += 1
        print(counter, "/", len(train_loader))
        
    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for input1, labels in val_loader:
            for i in range(1,30):
                inputs = patch_generator(input1)
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                output = model.forward(inputs)
                # Calculate Loss
                valloss = criterion(output, labels)
                # Add loss to the validation set's running loss
                ##jash injas

                # Since our model outputs a LogSoftmax, find the real 
                # percentages by reversing the log function
                output = torch.exp(output)
                # Get the top class of the output
                top_p, top_class = output.topk(1, dim=1)
                # See how many of the classes were correct?
                equals = top_class == labels.view(*top_class.shape)
                # Calculate the mean (get the accuracy for this batch)
                # and add it to the running accuracy for this epoch
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            val_loss += valloss.item()*inputs.size(0)
            # Print the progress of our evaluation
            counter += 1
            print(counter, "/", len(val_loader))
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    # Print out the information
    print('Accuracy: ', accuracy/len(val_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
