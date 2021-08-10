# -*- coding: utf-8 -*-
"""
x.liang@greenwich.ac.uk
25th March, 2020
Image Similarity using VGG16
"""
import os
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
#from scipy.spatial import distance
'''
def get_feature_vector(img):
 img1 = cv2.resize(img, (224, 224))
 feature_vector = feature_model.predict(img1.reshape(1, 224, 224, 3))
 return feature_vector
'''


    
class VGGSim():

	def __init__(self):
		# Use VGG16 model as an image feature extractor 
		self.image_input = Input(shape=(224, 224, 3))
		self.model = VGG16(input_tensor=self.image_input, include_top=True,weights='imagenet')
		layer_name = 'fc2'
		self.feature_model = Model(inputs=self.model.input,outputs=self.model.get_layer(layer_name).output)


		# fc2(Dense)output shape: (None, 4096) 
	def get_feature_vector_fromPIL(self,img):
		feature_vector = self.feature_model.predict(img)
		assert(feature_vector.shape == (1,4096))
		return feature_vector

	def calculate_similarity_cosine(self,vector1, vector2):
		#return 1- distance.cosine(vector1, vector2)
		return cosine_similarity(vector1, vector2) 

	# This distance can be in range of [0,∞]. And this distance is converted to a [0,1]
	def calculate_similarity_euclidean(self,vector1, vector2):
		#return distance.euclidean(vector1, vector2)     #distance.euclidean is slower
		return 1/(1+np.linalg.norm(vector1 - vector2))   #np.linalg.norm is faster


	def loadDir(self,dataset_path):
		# Load images in the images folder into array
		data_dir_list = os.listdir(dataset_path)

		img_data_list=[]
		for img_file in data_dir_list:
				img_path = dataset_path + '/'+ img_file
				img = image.load_img(img_path, target_size=(224, 224))
				x = image.img_to_array(img)
				img_data_list.append(x)


	
	def main(self,img_list):
		#print(img_data_list[0].shape)
		expanded_img = [np.expand_dims(x, axis=0) for x in img_list]
		#print(expanded_img[0].shape)
		processed_img = [preprocess_input(x) for x in expanded_img]
		#print(processed_img[0].shape)

		vector1 = self.get_feature_vector_fromPIL(processed_img[0])
		vector2 = self.get_feature_vector_fromPIL(processed_img[1])

		#vector_VGG16 =get_feature_vector_fromPIL(img_data_list[6])

		# Caculate cosine similarity: [-1,1], that is, [completedly different,same]
		image_similarity_cosine = self.calculate_similarity_cosine(vector1,vector2)
		# Cacluate euclidean similarity: range from [0, 1], that is, [completedly different, same]
		image_similarity_euclidean = self.calculate_similarity_euclidean(vector1,vector2)

		print('VGG16 image similarity_euclidean:',image_similarity_euclidean)
		print("VGG16 image similarity_cosine: {:.2f}%".format(image_similarity_cosine[0][0]*100))