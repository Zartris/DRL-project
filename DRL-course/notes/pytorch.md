# Pytorch help

~~~~Python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (28-3)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (10, 26, 26)
        # after one pool layer, this becomes (10, 13, 13)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=2)
        #
        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.2, inplace=False)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=2)
        self.fc1 = nn.Linear(20*6*6, 50)
        self.fc2 = nn.Linear(50, 10)
        # Below is not used yet:
        self.net = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size=2), 
                nn.Dropout2d(p=0.2, inplace=False),
            	nn.ReLU(), 
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size=2), 
                nn.ReLU(),
                Flatten(),
                nn.Linear(4096, 64),
                nn.ReLU(),
                nn.Linear(64, 10))
        ## TODO: Define the rest of the layers:
        # include another conv layer, maxpooling layers, and linear layers
        # also consider adding a dropout layer to avoid overfitting
        

    ## TODO: define the feedforward behavior
    def forward(self, x):
        # one activated conv layer
#         print("input:",x.size())
        x = F.relu(self.pool(self.conv1(x)))
#         print("conv1:",x.size())
        x = F.relu(self.pool(self.dropout(self.conv2(x))))
#         print("conv2:",x.size())
        x = x.view(x.size(0), -1)
#         print("flatten:",x.size())
        x = F.relu(self.dropout(self.fc1(x)))
#         print("fully connected1:",x.size())
        x = F.relu(self.dropout(self.fc2(x)))        
        # final output
#         print("fc2:",x.size())
		# we end up with a tensor of size (batch_size, outputsize) and we want to 
    	# take softmax over the output and not the batch.
        return F.log_softmax(x, dim=1)

    
# instantiate and print your Net
net = Net()
print(net)
~~~~

## Feature Visualization

Sometimes, neural networks are thought of as a black box, given some  input, they learn to produce some output. CNN's are actually learning to  recognize a variety of spatial patterns and you can visualize what each  convolutional layer has been trained to recognize by looking at the  weights that make up each convolutional kernel and applying those one at  a time to a sample image. These techniques are called feature  visualization and they are useful for understanding the inner workings  of a CNN.

In the cell below, you'll see how to extract and visualize the filter  weights for all of the filters in the first convolutional layer.

Note the patterns of light and dark pixels and see if you can tell  what a particular filter is detecting. For example, the filter pictured  in the example below has dark pixels on either side and light pixels in  the middle column, and so it may be detecting vertical edges.

~~~~python
# instantiate your Net
net = Net()

# load the net parameters by name
net.load_state_dict(torch.load('saved_models/fashion_net_ex.pt'))

print(net)
~~~~

~~~~python
# Get the weights in the first conv layer
weights = net.conv1.weight.data
w = weights.numpy()

# for 10 filters
fig=plt.figure(figsize=(20, 8))
columns = 5
rows = 2
for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(w[i][0], cmap='gray')
    
print('First convolutional layer')
plt.show()

weights = net.conv2.weight.data
w = weights.numpy()
~~~~

### Activation Maps

Next, you'll see how to use OpenCV's `filter2D` function to apply these filters to a sample test image and produce a series of **activation maps**  as a result. We'll do this for the first and second convolutional  layers and these activation maps whould really give you a sense for what  features each filter learns to extract.

~~~~python
# obtain one batch of testing images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images = images.numpy()

# select an image by index
idx = 4
img = np.squeeze(images[idx])

# Use OpenCV's filter2D function 
# apply a specific set of filter weights (like the one's displayed above) to the test image

import cv2
plt.imshow(img, cmap='gray')

weights = net.conv1.weight.data
w = weights.numpy()

# 1. first conv layer
# for 10 filters
fig=plt.figure(figsize=(30, 10))
columns = 5*2
rows = 2
for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if ((i%2)==0):
        plt.imshow(w[int(i/2)][0], cmap='gray')
    else:
        c = cv2.filter2D(img, -1, w[int((i-1)/2)][0])
        plt.imshow(c, cmap='gray')
plt.show()

# Same process but for the second conv layer (20, 3x3 filters):
plt.imshow(img, cmap='gray')

# second conv layer, conv2
weights = net.conv2.weight.data
w = weights.numpy()

# 1. first conv layer
# for 20 filters
fig=plt.figure(figsize=(30, 10))
columns = 5*2
rows = 2*2
for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if ((i%2)==0):
        plt.imshow(w[int(i/2)][0], cmap='gray')
    else:
        c = cv2.filter2D(img, -1, w[int((i-1)/2)][0])
        plt.imshow(c, cmap='gray')
plt.show()
~~~~

**Question:  Choose a filter from one of your trained convolutional layers; looking  at these activations, what purpose do you think it plays? What kind of  feature do you think it detects?**

**Answer**:  In the first convolutional layer (conv1), the very first filter,  pictured in the top-left grid corner, appears to detect horizontal  edges. It has a negatively-weighted top row and positively weighted  middel/bottom rows and seems to detect the horizontal edges of sleeves  in a pullover. 

In the second convolutional layer (conv2) the first filter looks like  it may be dtecting the background color (since that is the brightest  area in the filtered image) and the more vertical edges of a pullover.

### Saliency Maps

Salience can be thought of as the importance of something, and for a  given image, a saliency map asks: Which pixels are most important in  classifying this image?

Not all pixels in an image are needed or relevant for classification.  In the image of the elephant above, you don't need all the information  in the image about the background and you may not even need all the  detail about an elephant's skin texture; only the pixels that  distinguish the elephant from any other animal are important.

Saliency maps aim to show these important pictures by computing the  gradient of the class score with respect to the image pixels. A gradient  is a measure of change, and so, the gradient of the class score with  respect to the image pixels is a measure of how much a class score for  an image changes if a pixel changes a little bit.

**Measuring change**

A saliency map tells us, for each pixel in an input image, if we change it's value slightly (by *dp*),  how the class output will change. If the class scores change a lot,  then the pixel that experienced a change, dp, is important in the  classification task.

Looking at the saliency map below, you can see that it identifies the  most important pixels in classifying an image of a flower. These kinds  of maps have even been used to perform image segmentation (imagine the  map overlay acting as an image mask)!



# Handling data

Imports:

~~~~python
# Import things like usual
import numpy as np
import torch

import helper

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
~~~~

First up, we need to get our dataset. This is provided through the `torchvision`
package. The code below will download the MNIST dataset, then create 
training and test datasets for us. Don't worry too much about the 
details here, you'll learn more about this later.

~~~~python
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
# Normalizing means the first(0.5,0.5,0.5) is minus every graysscale pixel with (0.5), the second(0.5,0.5,0.5) is we divide every pixel with (0.5)

# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
~~~~

