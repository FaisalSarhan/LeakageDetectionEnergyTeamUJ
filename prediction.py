import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import custom_dataloader
from PIL import Image
import xlsxwriter
from torch.optim import lr_scheduler
from predection_gen import image_to_binary


pred_bin = open('predictionimage.bin', 'wb')
pred_bin.close ()
Image_num = int(input ('Please enter image number (from Data_shuffled_and_Blurred file) =  '))
Image_path = 'Data_shuffled_and_Blurred/VRayLightingAnalysis_{}.jpg'.format(Image_num)
try :
	image1 = Image.open(Image_path)
 
except : 
	print ('The Image you are looking for is unavailable')
	Image_num =  input ('Please enter image number (from Data_shuffled_and_Blurred file) =  ')
	image1 = Image.open(Image_path)

image_to_binary(Image_num)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = ('1' ,'0')
PATH = './net.pth'

transform = transforms.Compose([
        transforms.ToPILImage(mode=None),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


trainset = custom_dataloader.Custom_Dataset(224,224,3,2,'./training_dataset.bin', transform=transform)
testset = custom_dataloader.Custom_Dataset(224,224,3,2,'./testing_dataset.bin', transform=transform)
predset = custom_dataloader.Custom_Dataset(224,224,3,2,'./predictionimage.bin', transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=0)
predloader =  torch.utils.data.DataLoader(predset, batch_size=1,shuffle=False, num_workers=0)


classes = ('1' ,'0')
net0 = models.resnet18(pretrained=False)
for param in net0.parameters():
	param.requires_grad = False

net0.fc = nn.Identity()	 

layers = nn.Sequential(nn.Linear(512,500),nn.ReLU(),nn.Linear(500,500),nn.ReLU(),nn.Linear(500,500),nn.ReLU(),nn.Linear(500,10),nn.ReLU(),nn.Linear(10,2),nn.Softmax(dim=1))
net= nn.Sequential (net0,*layers)
net.load_state_dict(torch.load(PATH,map_location=torch.device(device)))




net.eval()
with torch.no_grad():
    for i, data in enumerate(predloader, 0):
        inputs, labels = data

        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        predection_class = preds.item()

        if predection_class == labels.item():

            # print (preds)
            # print (predection_class)
            if predection_class == 1 : 
                predection = ('Prediction : Leakage Detected')
            elif predection_class == 0 :
                predection = ('Prediction :No Leakage Detected')

            if labels.item() == 1 : 
                truth = ('Label : Leakage Detected')
            elif labels.item() == 0 :
                truth = ('Label :No Leakage Detected')


            # print(i)
            plt.imshow(image1)
            plt.title('{}\n{}'.format(predection,truth))
            plt.show()
