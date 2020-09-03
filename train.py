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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
log_file = open("log.txt","w") 

loss_excel = xlsxwriter.Workbook('epoch loss.xlsx')
loss_sheet = loss_excel.add_worksheet()
loss_sheet.write( 'A1' , 'epoch')
loss_sheet.write( 'B1' , 'training loss')
loss_sheet.write( 'C1' , 'testing loss')

No_epochs = 500
PATH = './net.pth'
for epoch in range(No_epochs):
	loss_sheet.write( epoch+1 , 0 ,epoch+1)

transform = transforms.Compose([
		transforms.ToPILImage(mode=None),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


trainset = custom_dataloader.Custom_Dataset(224,224,3,2,'./training_dataset.bin', transform=transform)
testset = custom_dataloader.Custom_Dataset(224,224,3,2,'./testing_dataset.bin', transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=False, num_workers=0)

classes = ('1' ,'0')

net0 = models.resnet18(pretrained=True)

for param in net0.parameters():
	param.requires_grad = False

net0.fc = nn.Identity()	 



layers = nn.Sequential(nn.Linear(512,500),nn.ReLU(),nn.Linear(500,500),nn.ReLU(),nn.Linear(500,500),nn.ReLU(),nn.Linear(500,10),nn.ReLU(),nn.Linear(10,2),nn.Softmax(dim=1))

net= nn.Sequential (net0,*layers)

net.to(device)


criterion = nn.CrossEntropyLoss(reduction='sum')
# optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

#----------------------------------------------------------------------------------------------------------
#training

training_loss = np.zeros(No_epochs)
testing_loss = np.zeros(No_epochs)
min_loss = 10**10
for epoch in range(No_epochs):
	loops1 = 0
	loops2 = 0
	epoch_loss1 = 0
	epoch_loss2 = 0
	No_training_samples = 0
	No_correct_training = 0
	No_testing_samples = 0
	No_correct_testing = 0
	net.train()
	for i, data in enumerate(trainloader, 0):
		# print(data[1][0].item())
		# plt.figure()
		# plt.imshow(data[0][0,:,:,:].numpy().transpose(1,2,0))
		# plt.show()
		inputs, labels = data
		inputs, labels = data[0].to(device), data[1].to(device)
		No_training_samples += inputs.shape[0]
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		max_index = outputs.max(dim = 1)[1]
		No_correct_training += (max_index == labels).sum().item()
		#calculate training_loss for each epoch
		training_loss[epoch] += loss.item()
	training_loss[epoch] /= No_training_samples
	loss_sheet.write( epoch+1 , 1 ,training_loss [epoch])

	 
	net.eval()	  
	with torch.no_grad():
		for ii,data in enumerate(testloader,0):
			# images, labels = data

			images, labels = data[0].to(device), data[1].to(device)
			No_testing_samples += images.shape[0]
			outputs = net(images)
			loss = criterion(outputs, labels)
			max_index = outputs.max(dim = 1)[1]
			No_correct_testing += (max_index == labels).sum().item()
			#calculate testing_loss for each epoch
			testing_loss[epoch] += loss.item()
	
		testing_loss[epoch] /= No_testing_samples
		loss_sheet.write( epoch+1 , 2 ,testing_loss [epoch])
			  
		print('Epoch %d training loss: %.3f testing loss: %.3f   training_correct: %.3f / %.3f  = (%.3f)   testing_correct: %.3f / %.3f  = (%.3f)' % (epoch + 1, training_loss[epoch], testing_loss[epoch],No_correct_training,No_training_samples,No_correct_training/No_training_samples,No_correct_testing,No_testing_samples,No_correct_testing/No_testing_samples))
		log_file.write(' %.3f  %.3f %.3f %.3f\n' % (training_loss[epoch], testing_loss[epoch],No_correct_training/No_training_samples,No_correct_testing/No_testing_samples))
		log_file.flush()
	if min_loss > testing_loss[epoch]:
		min_loss = testing_loss[epoch]
		torch.save(net.state_dict(), PATH)
  
print('Finished Training')
plt.plot([i+1 for i in range(No_epochs)], training_loss, label = "Training Loss")
plt.plot([i+1 for i in range(No_epochs)], testing_loss, label = "Testing Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0,1])
plt.legend()
plt.show()
loss_excel.close()
log_file.close()

#-----------------------------------------------------------------------------------------------------------------------------
