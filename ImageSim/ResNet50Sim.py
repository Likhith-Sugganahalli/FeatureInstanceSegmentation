# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:46:51 2020
@author: x.liang@greenwich.ac.uk
Image Similarity using ResNet50
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
from ImageSim.resnet50 import ResNet50
#from keras.models import Model
from tensorflow.keras.layers import Input
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
# avg_pool (AveragePooling2D) output shape: (None, 1, 1, 2048)
# Latest Keras version causing no 'flatten_1' issue; output shape:(None,2048) 


class ResNetSim():
	def __init__(self):
		self.image_input = Input(shape=(224, 224, 3))
		self.feature_model = ResNet50(input_tensor=self.image_input, include_top=False,weights='imagenet')

	def get_feature_vector_fromPIL(self,img):
		#print('h')
		#print("PIL Vector incoming {}".format(img.shape))
		feature_vector = self.feature_model.predict(img)
		a, b, c, n = feature_vector.shape
		feature_vector= feature_vector.reshape(b,n)
		#print("PIL Vector incoming {} outgoing {}".format(type(img),feature_vector.shape))
		return feature_vector

	def calculate_similarity_cosine(self,vector1, vector2):
		#return 1 - distance.cosine(vector1, vector2)
		return cosine_similarity(vector1, vector2)

	# This distance can be in range of [0,∞]. And this distance is converted to a [0,1]
	def calculate_similarity_euclidean(self,vector1, vector2):
		return 1/(1 + np.linalg.norm(vector1- vector2))

	
	def load_from_dir(self):
		data_path ='/home/whoknows/Documents/FeatureMatching-Python/ImageSim/images'
		data_dir_list = os.listdir(data_path)

		img_data_list=[]
		for img_name in data_dir_list:
			img_path = data_path + '/'+ img_name
			img = image.load_img(img_path, target_size=(224, 224))
			x = image.img_to_array(img)
			img_data_list.append(x)
		return img_data_list



	def main(self,img_data_list):
		
		# Load images in the images folder into array
		# Load images in the images folder into array
		
		#img_data_list.append(x)

		#vector_resnet = get_feature_vector_fromPIL(img_data_list[6])

		# Load images in the images folder into array

		'''
		cwd_path = os.getcwd()
		data_path ='/home/whoknows/Documents/FeatureMatching-Python/ImageSim/images'
		data_dir_list = os.listdir(data_path)

		img_data_list=[]
		for dataset in data_dir_list:

			img_path = data_path + '/'+ dataset
			img = image.load_img(img_path, target_size=(224, 224))
			x = image.img_to_array(img)
			print('0 {}'.format(x.shape))
			x = np.expand_dims(x, axis=0)
			print('1 {}'.format(x.shape))
			x = preprocess_input(x)
			print('2 {}'.format(x.shape))
			img_data_list.append(x)
		'''
		#print(img_data_list[0].shape)
		expanded_img = [np.expand_dims(x, axis=0) for x in img_data_list]
		#print(expanded_img[0].shape)
		processed_img = [preprocess_input(x) for x in expanded_img]
		#print(processed_img[0].shape)
		
		vector1 = self.get_feature_vector_fromPIL(processed_img[0])
		vector2 = self.get_feature_vector_fromPIL(processed_img[1])



		# Cacluate euclidean similarity: range from [0, 1], that is, [completedly different, same]
		image_similarity_euclidean = self.calculate_similarity_euclidean(vector1, vector2)
		# Caculate cosine similarity: [-1,1], that is, [completedly different, same]
		image_similarity_cosine = self.calculate_similarity_cosine(vector1, vector2)

		#print('ResNet50 image similarity_euclidean: ',image_similarity_euclidean)
		#print('ResNet50 image similarity_cosine: {:.2f}%'.format(image_similarity_cosine[0][0]*100))
		return[image_similarity_euclidean,image_similarity_cosine[0][0]*100]
