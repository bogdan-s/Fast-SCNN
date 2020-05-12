import os  
import torch 
import torch.nn as nn
import torchvision
from torchvision import transforms, utils
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d, Dropout2d, MaxPool2d, ReLU, UpsamplingNearest2d
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

# Hyper parameters
epochs = 5
num_classes = 2
batch_size = 100
learning_rate = 0.001
img_size = 512

masks_train_source = 'D:/Python/DataSets/ADE20K_Filtered/Train/New_Masks/0/'
images_train_source = 'D:/Python/DataSets/ADE20K_Filtered/Train/Images/0/'
masks_validation_source = 'D:/Python/DataSets/ADE20K_Filtered/Validation/New_Masks/0/'
images_validation_source = 'D:/Python/DataSets/ADE20K_Filtered/Validation/Images/0/'

class MyDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
    
    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(img_size, img_size))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        # if random.random() > 0.5:
        #     image = TF.hflip(image)
        #     mask = TF.hflip(mask)

        # Random vertical flipping
        # if random.random() > 0.5:
        #     image = TF.vflip(image)
        #     mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __getitem__(self, idx):
        image_path = self.images[idx]
        masks_path = self.masks[idx]
        img = Image.open(image_path)
        mask = Image.open(masks_path)
        x, y = self.transform(img, mask)
        return x, y
    
    def __len__(self):
        return len(self.images)

# Create Train dateset & loader

train_masks = []
for root, dirs, files in os.walk(masks_train_source):
    for img in files:
        train_masks.append(masks_train_source + img)
        break
train_images = []
for root, dirs, files in os.walk(images_train_source):
    for img in files:
        train_images.append(images_train_source + img)
        break


train_dataset = MyDataset(train_images, train_masks)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f'Train dataset has {len(train_data_loader)} batches of size {batch_size}')

# Create validation dataset & loader

validation_masks = []
for root, dirs, files in os.walk(masks_validation_source):
    for img in files:
        validation_masks.append(masks_validation_source + img)
        break
validation_images = []
for root, dirs, files in os.walk(images_validation_source):
    for img in files:
        validation_images.append(images_validation_source + img)
        break


validation_dataset = MyDataset(validation_images, validation_masks)
validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
print(f'Validation dataset has {len(validation_data_loader)} batches of size {batch_size}')


# helper function to show an image
def matplotlib_imshow(display_list):
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        # plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()

# get some random training images
dataiter = iter(train_data_loader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)
img = transforms.ToPILImage()(images.squeeze_(0))
msk = transforms.ToPILImage()(labels.squeeze_(0))
print(type(transforms.ToPILImage()(images.squeeze_(0))))
# show images
matplotlib_imshow([img, msk])


"""
# Based on https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/unet.py#L19
class UNetMini(Module):

    def __init__(self, num_classes):
        super(UNetMini, self).__init__()

        # Use padding 1 to mimic `padding='same'` in keras,
        # use this visualization tool https://ezyang.github.io/convolution-visualizer/index.html
        self.block1 = Sequential(
            Conv2d(1, 32, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU(),
        )
        self.pool1 = MaxPool2d((2, 2))

        self.block2 = Sequential(
            Conv2d(32, 64, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(),
        )
        self.pool2 = MaxPool2d((2, 2))

        self.block3 = Sequential(
            Conv2d(64, 128, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU()
        )

        self.up1 = UpsamplingNearest2d(scale_factor=2)
        self.block4 = Sequential(
            Conv2d(192, 64, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU()
        )

        self.up2 = UpsamplingNearest2d(scale_factor=2)
        self.block5 = Sequential(
            Conv2d(96, 32, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU()
        )

        self.conv2d = Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        out1 = self.block1(x)
        out_pool1 = self.pool1(out1)

        out2 = self.block2(out_pool1)
        out_pool2 = self.pool1(out2)

        out3 = self.block3(out_pool2)

        out_up1 = self.up1(out3)
        # return out_up1
        out4 = torch.cat((out_up1, out2), dim=1)
        out4 = self.block4(out4)

        out_up2 = self.up2(out4)
        out5 = torch.cat((out_up2, out1), dim=1)
        out5 = self.block5(out5)

        out = self.conv2d(out5)

        return out

model = UNetMini(num_classes).to(device)

# summary(model, input_size=(1, 256, 256))  # (ch, H, W)

# Here is the loss and optimizer definition
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# The training loop
total_steps = len(train_data_loader)
print(f"{epochs} epochs, {total_steps} total_steps per epoch")
for epoch in range(epochs):
    for i, (images, masks) in enumerate(train_data_loader, 1):
        images = images.to(device)
        masks = masks.type(torch.LongTensor)
        masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        softmax = F.log_softmax(outputs, dim=1)
        loss = criterion(softmax, masks)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i) % 100 == 0:
            print (f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}")
"""