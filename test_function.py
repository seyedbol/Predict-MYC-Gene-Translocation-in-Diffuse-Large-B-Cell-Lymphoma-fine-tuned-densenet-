# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:19:33 2019

@author: EmdadLaptop_77615395
"""

###test the model
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
from torch.utils.tensorboard import SummaryWriter
#############

# random.seed(0)
#############

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
model.load_state_dict(torch.load("/content/drive/My Drive/dataset/Copy of 2nd_Q_512"))
model.cuda()
model.eval()


##############

def patch_generator(loader):
    i = random.randint(66,1854)##1920
    j = random.randint(66,2494)##2560
    #20 will cover whole image
    imgs_new_1 = loader[:,:,i-64:i+64,j-64:j+64]
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
          pyramid_init = np.stack((pyram2[0],pyram1[0]))
          if(i==0):
              A = pyramid_init
          if(i>1):
#               A = np.stack((pyramid_init,A), axis=0)
              A = np.append( pyramid_init , A , axis = 0)
    return A.transpose((0,3,2,1)) 


############################

def process_image(image_path):
    # Load Image
    img = Image.open(image_path)
    # Get the dimensions of the image
    width, height = img.size
    
    #    Turn image into numpy array
    img = np.array(img)
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # Make all values between 0 and 1
    img = img/255
    
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.6730)/0.1463
    img[1] = (img[1] - 0.6604)/0.1646
    img[2] = (img[2] - 0.6644)/0.1579
    
#     # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image.cuda()
# Using our model to predict the label



def predict(image, model):
    # Pass the image through our model
    probs = 0
    classes = 0
    for i in range (0,30):
        inputs = (patch_generator(image))
#         inputs  = torch.from_numpy(inputs).float().to(device)
#         labels  = labels.to(device)
        output = model.forward(inputs)

        # Reverse the log function in our output
        output = torch.exp(output)

        # Get the top predicted class, and the output percentage for
        # that class
        classes = output.topk(1, dim=1)[1]
        if(classes==0):
           probs1 =output.topk(1, dim=1)[0]
        if(classes==1):
           probs1 = 1-output.topk(1, dim=1)[0]
        print(probs1)
        probs =probs+ probs1
           
        
    return probs.item()/30 
    # def show_image(image):
    #     # Convert image to numpy
    #     image = image.numpy()

    #     # Un-normalize the image
    #     image[0] = image[0] * 0.1463 + 0.6730

    #     # Print the image
    #     fig = plt.figure(figsize=(25, 4))
    #     plt.imshow(np.transpose(image[0], (1, 2, 0)))
    # Process Image
image = process_image("/content/drive/My Drive/dataset/test/neg/102_neg.tif")
# Give image to model to predict output
top_prob = predict(image, model)
# Show the image
# show_image(image)
# Print the results
print("The model is ", top_prob*100, "% certain that the image has a predicted class of 1 "  )