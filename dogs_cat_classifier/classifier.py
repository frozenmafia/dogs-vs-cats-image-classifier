import shutil
import re
import os

def get_cwd():
    return os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(get_cwd(),'model_intern.pth')
data_dir = os.path.join(get_cwd(),'dogs-vs-cats')

train_dir = f'{data_dir}\\train'
train_dogs_dir = f'{train_dir}\\dogs'
train_cats_dir = f'{train_dir}\\cats'

#Training model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math

print(torch.__version__)
plt.ion()




data_transforms = {
    'train':transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224,scale=(0.96,1.0),
                                    ratio=(0.95,1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),
    'validate':transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.486,0.406],[0.229,0.224,0.225])
    ])
}
image_dataset = {
    x:datasets.ImageFolder(os.path.join(data_dir,x),
                          data_transforms[x])
    for x in ['train','validate']
}
dataloaders = {
    x:torch.utils.data.DataLoader(
        image_dataset[x],batch_size=4,
        shuffle=True,num_workers=4)
    for x in ['train','validate']
}

dataset_sizes = {
    x:len(image_dataset[x]) for x in ['train','validate']
}
class_names = image_dataset['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# print(class_names) # => ['cats', 'dogs']
# print(f'Train image size: {dataset_sizes["train"]}')
# print(f'Validation image size: {dataset_sizes["validate"]}')
    

def imshow(inp,title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std*inp + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(3,16,5)
        self.conv2 = nn.Conv2d(16,32,5)
        self.conv3 = nn.Conv2d(32,64,5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*53*53,120)
        self.fc2 = nn.Linear(120,10)

        
    def forward(self,x):
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1,32*53*53)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        return x
    



def train_model(training_model,train_loader,criterion,optimizer):
    for epoch in range(2):
        running_loss = 0.0
        for i,data in enumerate(train_loader,0):
            inputs,labels = data
            # print(inputs.shape)
            optimizer.zero_grad()
            
            outputs = training_model(inputs)
#             print(outputs)
#             print(labels)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i%2000 == 1999:
                print(epoch+1,"              ",i+1,"             ",running_loss/2000)
                running_loss = 0.0
                
    print('Finished Training')
    print('Saving the latest params')
    torch.save(training_model.state_dict(),model_path)

def validate(validate_model,validate_dataloader):
    # print(validate_dataloader)
    # validate_model = Network()
    # validate_model.load_state_dict(torch.load(model_path))
    dataiter = iter(validate_dataloader)
    # print(dataiter)
    correct =  0
    total = 0
    for data in enumerate(validate_dataloader,0):
        # print(data.size)
        # print(data)
        _,data = data
        # print(len(data))
        images = data[0]
        labels = data[1]
        # imshow(torchvision.utils.make_grid(images))
        outputs = validate_model(images)
        _,predicted = torch.max(outputs,1)
        total +=labels.size(0)
        correct +=(predicted == labels).sum().item()
        # print('Predicted: ', ' '.join('%5s' % labels[predicted[j]]
        #                       for j in range(4)))
    print('Accuracy of the network on the 2000 images: %d %%'%(100*correct/total))

if __name__ == "__main__":
    print('checkpoint_loaded,getting trained params and intiating to train the CNN')
    training_model = Network()
    training_model.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(training_model.parameters(),lr = 0.001,momentum = 0.9)
    train_model(training_model,dataloaders['train'],criterion,optimizer)
    # torch.save(net.state_dict(),model_path)
    # model_loaded = Network()
    # model_loaded.load_state_dict(torch.load(model_path))
    # model_loaded.eval()
    # print(dataloaders)

    validate_model = Network()
    print('getting accuracy on validation set')
    print('trained models loaded')

    validate_model.load_state_dict(torch.load(model_path))
    validate(validate_model,dataloaders['validate'])