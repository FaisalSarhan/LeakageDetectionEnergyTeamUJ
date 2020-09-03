from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import struct
import torch
import numpy as np
import matplotlib.pyplot as plt

class Custom_Dataset(Dataset):

	def __init__(self, xM,xN,xZ,No_classes,file_path, transform=None):

		self.transform = transform
		self.file = open(file_path, 'rb')
		self.xM = 224
		self.xN = 224
		self.xZ = 3
		self.No_classes = No_classes
		d = self.file.read(2)
		self.length = struct.unpack('H',d)
	def __len__(self):
		return self.length[0]

	def __getitem__(self,idx):
		
		xM = self.xM
		xN = self.xN
		xZ = self.xZ
		No_classes = self.No_classes
		self.file.seek(idx*(224*224*3+1)+2)
		#start---------------------------------------------------------------------------------------------------

		image0 = self.file.read(224*224*3)                                                                   
		image1 = struct.unpack('B'*len(image0),image0)
		image2 = np.reshape(image1, [224,224,3]).astype('uint8')
		
		
		label0 = self.file.read(1)
		label1 = struct.unpack('B' , label0)
		
		
		
		
		image = np.zeros((224,224,3)).astype('uint8')
		image = image2
		label = np.zeros(1).astype('uint8')
		label = label1
		#end-----------------------------------------------------------------------------------------------------
		
		
		# plt.imshow(image.astype('uint8'))
		# print(label)
		# plt.show()
		
		

		
		if self.transform:
			image = self.transform(image)
			label = torch.Tensor(label).long()
		return image,label[0]
		
	