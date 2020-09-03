from scipy.signal import convolve2d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import struct
import torchvision.transforms as transforms
import pandas as pd
import xlrd



def image_to_binary(image_num):
	
	transform = transforms.Resize(244, interpolation=2) # 224 is the standard image size for resnet 18
	excel =  np.asarray(pd.read_excel( 'Data_shuffled_and_Blurred/labels_shuffled.xlsx', header=None )) #excel 
	number_of_samples = 1     

	f_pred = open('predictionimage.bin', 'wb')

	h=struct.pack('H', number_of_samples)
	f_pred.write(h)


	file_name =  'Data_shuffled_and_Blurred/VRayLightingAnalysis_{}.jpg'.format(image_num)
	image = transform(Image.open(file_name))    
	image = image.resize((224 ,224)) 
	image = np.asarray(image)
	image = image.astype('uint8')
	[xM,xN] = image.transpose(2, 0, 1)[0].shape
	x = np.reshape(image, (xM*xN*3))
	p = image_num-1
	label_index = int(excel[p]) # p-1 to match the label and image loops
	label = np.array(label_index)
	label = label.astype('uint8')


	xlen = len(x)
	xpacked = struct.pack('B'*xlen, *x)
	labelPacked = struct.pack('B', label)
	f_pred.write (xpacked)
	f_pred.write(labelPacked)



	#end-----------------------------------------------------------------------------------------------------



	f_pred.close()
