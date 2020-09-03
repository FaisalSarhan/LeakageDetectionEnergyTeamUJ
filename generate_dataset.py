from scipy.signal import convolve2d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import struct
import torchvision.transforms as transforms
import pandas as pd
import xlrd

transform = transforms.Resize(244, interpolation=2) # 224 is the standard image size for resnet 18

excel =  np.asarray(pd.read_excel( 'Data_shuffled_and_Blurred/labels_shuffled.xlsx', header=None )) #excel file loading
No_classes = 2
number_of_samples = len(excel)     # since number of labels will be fitched from the excel file
No_testing_samples = int(number_of_samples *0.2)
No_training_samples = int(number_of_samples - No_testing_samples)
f_traing = open('training_dataset.bin', 'wb')
f_test = open('testing_dataset.bin', 'wb')

h=struct.pack('H', No_training_samples)
f_traing.write(h)
for p in range(1 , No_training_samples+1):
	#--------------------------------------------------------------------------------------------------------
	#  to fetch training data 
	file_name = 'Data_shuffled_and_Blurred/VRayLightingAnalysis_{}.jpg'.format(p)

	image = transform(Image.open(file_name))    #load image
	# plt.figure()
	# plt.imshow(np.asarray(image))
	image = image.resize((224 ,224)) # to avoid loosing data when renet18 center crops the image
	# plt.figure()
	# plt.imshow(np.asarray(image))
	# plt.show()
	image = np.asarray(image)
	image = image.astype('uint8')
	[xM,xN] = image.transpose(2, 0, 1)[0].shape
	x = np.reshape(image, (xM*xN*3))
	label_index = int(excel[p-1]) # p-1 to match the label and image loops
	label = np.array(label_index)
	label = label.astype('uint8')

	#--------------------------------------------------------------------------------------------------------


	#start---------------------------------------------------------------------------------------------------
	# pacing data and writing it on the binary file  
	xlen = len(x)
	xpacked = struct.pack('B'*xlen, *x)
	labelPacked = struct.pack('B', label)
	f_traing.write (xpacked)
	f_traing.write(labelPacked)



	#end-----------------------------------------------------------------------------------------------------



f_traing.close()

h=struct.pack('H', No_testing_samples)
f_test.write(h)
for p in range(No_training_samples+1 , 1+number_of_samples):
	#--------------------------------------------------------------------------------------------------------
	# to fetch testing data 
	file_name = 'Data_shuffled_and_Blurred/VRayLightingAnalysis_{}.jpg'.format(p)
	image = transform(Image.open(file_name))    #load image
	image = image.resize((224 ,224))  # to avoid loosing data when renet18 center crops the image
	image = np.asarray(image)
	[xM,xN] = image.transpose(2, 0, 1)[0].shape
	x = np.reshape(image, (xM*xN*3))
	label_index = int(excel[p-1]) # p-1 to match the label and image loops
	label = np.array(label_index)
	label = label.astype('uint8')
	#--------------------------------------------------------------------------------------------------------





	#start---------------------------------------------------------------------------------------------------
	#pacing data and writing it on the binary file f_test
	xlen = len(x)
	xpacked = struct.pack('B'*xlen, *x)
	f_test.write (xpacked)
	labelPacked = struct.pack('B', label)
	f_test.write(labelPacked)


# plt.imshow(image.astype('uint8'))
# plt.show()


	#end-----------------------------------------------------------------------------------------------------


f_test.close()
