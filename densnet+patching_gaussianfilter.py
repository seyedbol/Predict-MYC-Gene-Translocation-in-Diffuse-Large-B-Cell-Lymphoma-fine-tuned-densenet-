####new_one
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
import matplotlib.pyplot as plt
from skimage.transform import pyramid_gaussian
import random
!pip install -q tb-nightly
!pip install tb-nightly
from torch.utils.tensorboard import SummaryWriter

##############setting up the seeds

torch.manual_seed(0)
random.seed(0)
writer = SummaryWriter()

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
    i = random.randint(130,1790)##1920
    j = random.randint(130,2430)##2560
    #20 will cover whole image
    imgs_new_1 = loader[:,:,i-128:i+128,j-128:j+128]
#     imgs_new_2 = loader[:,:,i-128:i+128,j-128:j+128]
#     imgs_new_3 = loader[:,:,i-64:i+64,j-64:j+64]
    return imgs_new_1
  
def guassian_filter(image):
#     image = Image.open(path)
#     cols,rows= image.size
#     for in range (image.shape)
#     pyramid0= tuple(pyramid_gaussian(image[0,0,:,:] , downscale=2))
#     pyramid1= tuple(pyramid_gaussian(image[0,1,:,:] , downscale=2))
#     pyramid2= tuple(pyramid_gaussian(image[0,2,:,:] , downscale=2))
#     pyramid = np.dstack((pyramid1[2],pyramid0[2]))
#     pyramid_init = np.dstack((pyramid2[2],pyramid))
    image = image.permute((0,3,2,1))                    
    for i in range(0,image.shape[0],2):
          pyram1 = tuple(pyramid_gaussian(image[i,:,:,:] , downscale=2,multichannel =True))
          ####################################################
          pyram2 = tuple(pyramid_gaussian(image[i+1,:,:,:] , downscale=2,multichannel =True))
          ####################################################
          pyramid_init = np.stack((pyram2[1],pyram1[1]))
          if(i==0):
              A = pyramid_init
          if(i>1):
#               A = np.stack((pyramid_init,A), axis=0)
              A = np.append( pyramid_init , A , axis = 0)
    return A.transpose((0,3,2,1)) 

################

transformations = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean=[0.6730, 0.6604, 0.6644], std=[0.1463, 0.1646, 0.1579])])

######### load the data via pytorch utility tool

train_set = torchvision.datasets.ImageFolder("/content/drive/My Drive/dataset/train/", transform = transformations)
valid_set = torchvision.datasets.ImageFolder("/content/drive/My Drive/dataset/valid/", transform = transformations)
test_set = torchvision.datasets.ImageFolder("/content/drive/My Drive/dataset/test/", transform = transformations)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

val_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=True)
test_loader =  torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

################################## make use of pretrained method(densenet)

model = models.densenet161(pretrained=True)
# model = models.alexnet(pretrained=True)
# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False

################################## Create new classifier for model using torch.nn as nn library

from torch import nn 
classifier_input = model.classifier.in_features
num_labels = 2
classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))
# Replace default classifier with new classifier
model.classifier = classifier

###############

# Set the error function using torch.nn as nn library
criterion = nn.NLLLoss()
# Set the optimizer function using torch.optim as optim library
optimizer = torch.optim.Adam(model.classifier.parameters(),lr=0.01)

##############set up the gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move model to the device specified above
model.to(device)

#########3

############### training the model
epochs = 20
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy_train = 0
    accuracy_valid = 0
    # Training the model
    model.train()
    counter = 0
    for input1, labels in train_loader:
        for i in range (1,30):
            inputs = guassian_filter(patch_generator(input1))
            # Move to  
            inputs  = torch.from_numpy(inputs).float().to(device)
            labels  = labels.to(device)
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
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_loss += loss.item()*inputs.size(0)
        accuracy_train += torch.mean(equals.type(torch.FloatTensor)).item()
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
              
                inputs = guassian_filter(patch_generator(input1))
                inputs  = torch.from_numpy(inputs).float().to(device)
                labels  = labels.to(device)
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                output = model.forward(inputs)
                # Calculate Loss
                valloss = criterion(output, labels)
                # Add loss to the validation set's running loss
                ##jash injas
                val_loss += valloss.item()*inputs.size(0)
                # Since our model outputs a LogSoftmax, find the real 
                # percentages by reversing the log function
                output = torch.exp(output)
                # Get the top class of the output
                top_p, top_class = output.topk(1, dim=1)
                # See how many of the classes were correct?
                equals = top_class == labels.view(*top_class.shape)
                # Calculate the mean (get the accuracy for this batch)
                # and add it to the running accuracy for this epoch
            accuracy_valid += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Print the progress of our evaluation
            counter += 1
            print(counter, "/", len(val_loader))
            
###########################################################################tensor board

    writer.add_scalar('training accuracy_512', accuracy_train/len(train_loader), epoch)
    writer.add_scalar('validation accuracy_512', accuracy_valid/len(val_loader), epoch)
    writer.add_scalar('Loss/train_512', train_loss/len(train_loader.dataset) , epoch)
    writer.add_scalar('Loss/valid_512', val_loss/len(val_loader.dataset), epoch)
    
###########################################################################
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    # Print out the information
    print('Accuracy: ', accuracy_valid/len(val_loader))
    print('Accuracy: ', accuracy_train/len(train_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
torch.save(model.state_dict(),"/content/drive/My Drive/dataset/2nd_P")
%load_ext tensorboard
%tensorboard --logdir=runs
torch.save(model.state_dict(),"/content/drive/My Drive/dataset/2nd_Q_256")