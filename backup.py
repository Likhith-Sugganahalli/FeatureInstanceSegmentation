from operator import inv
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn import neighbors
from sklearn.cluster import MeanShift, estimate_bandwidth
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from colors import rgbcolor, uniquecolors
from sklearn.neighbors import NearestNeighbors
from skimage.metrics import structural_similarity as ssim


from neural_network.demo import demo as neural_network


class MyImage:
	def __init__(self, img_name):
		self.img_grey = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
		self.img = cv2.imread(img_name)
		self.__name = str(img_name)
		height, width, _ = self.img.shape
		self.img_area = int(height * width)

	def __str__(self):
		return self.__name

class CVrunner:
	def __init__(self):
		current_dir = os.path.dirname(os.path.realpath(__file__))
		self.test_location = os.path.join(current_dir,'test')
		self.template_location = os.path.join(current_dir,'templates')
		self.template_imgs = list(self.load_images_from_folder(self.template_location))
		self.test_imgs = list(self.load_images_from_folder(self.test_location))
		self.neural_network_obj = neural_network()

	def load_images_from_folder(self,folder):
		for filename in os.listdir(folder):
			img = MyImage(os.path.join(folder,filename))
			if img is not None:
				yield(img)
				#print(type(img))

	def main(self):
		for img1 in self.template_imgs:
			for img2 in self.test_imgs:
				#self.FlannMatcher(img1,img2)
				#self.bruteMatcher(img1,img2)
				#self.multipleOrb(img1.img_grey,img2.img_grey)
				#self.mean_clustering(img1,img2)
				#self.dbscan(img1,img2)
				self.imager(img1,img2)
				#self.kmean(img1,img2)
				#self.watershed(img2)
				#self.orb_features(img1)
				#self.orb_features(img2)
				#self.keypointContours(img1,img2)
				#val1 = self.unique_count_app(img1.img)
				#print("{}".format(img1),val1)

	def orb_features(self,img_object):
		#img = img_object.img
		img_grey = img_object.img_grey
		img = img_object.img
		# Initiate ORB detector
		#orb1 = cv2.ORB_create(10000)
		orb2 = cv2.ORB_create(10000, 1.2, nlevels=8, edgeThreshold = 5)
		# find the keypoints with ORB
		#kp1, des1 = orb1.detectAndCompute(img_grey, None)
		kp, des = orb2.detectAndCompute(img, None)
		#descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
		# compute the descriptors with ORB
		#kp1, des1 = orb.compute(img, kp)
		#kp1, des1 = descriptor.compute(img_grey, kp1)
		# draw only keypoints location,not size and orientation
		#img2 = cv2.drawKeypoints(img, kp1, None, color=(0,255,0), flags=0)


		img1 = cv2.drawKeypoints(img, kp, None, color=(255,0,0), flags=0)
		#img2 = cv2.drawKeypoints(img, kp2, None, color=(255,0,0), flags=0)
		#print('len of kp1:{}'.format(len(kp1)))
		print('len of kp of {}:{}'.format(img_object,len(kp)))
		#name = "{}".format(img_object)
		cv2.imshow('', img1)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		#cv2.imshow('kp2', img2)

		return kp,des
		
		#cv2.imshow('img2', img2)
		
	
		#return kp, des


	def new_orb_features(self,img):
		#img = img_object.img
		#img_grey = img_object.img_grey
		#img = img_object.img
		# Initiate ORB detector
		#orb1 = cv2.ORB_create(10000)
		orb2 = cv2.ORB_create(10000, 1.2, nlevels=8, edgeThreshold = 5)
		# find the keypoints with ORB
		#kp1, des1 = orb1.detectAndCompute(img_grey, None)
		kp, des = orb2.detectAndCompute(img, None)
		#descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
		# compute the descriptors with ORB
		#kp1, des1 = orb.compute(img, kp)
		#kp1, des1 = descriptor.compute(img_grey, kp1)
		# draw only keypoints location,not size and orientation
		#img2 = cv2.drawKeypoints(img, kp1, None, color=(0,255,0), flags=0)
		#cv2.imshow('kp2', img2)

		return kp,des
		

	def homography(self,kp1,des1,img1,kp2,des2,img2, good):
		MIN_MATCH_COUNT = 10
		if len(good)>MIN_MATCH_COUNT:
			#print('poly check')
			
			
			#print(kp1[good[0].queryIdx].pt)
			#print('{} {}  {}'.format(len(kp1),len(kp2),len(good)))
			src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,4.4)
			matchesMask = mask.ravel().tolist()
			h,w,d = img1.shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			#print(pts)
			dst = cv2.perspectiveTransform(pts,M)
			#print(dst)
			#print('#########################')
			img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		else:
			print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
			matchesMask = None

		draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
		img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
		plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB), 'gray'),plt.show()




	def test(self,kp1,des1,img1,kp2,des2,img2, matches):

		src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])
		#print(dst_pts)
		max_x_y = np.amax(dst_pts, axis=0)
		min_x_y = np.amin(dst_pts, axis=0)
		left_point = (int(min_x_y[0]-25),int(max_x_y[1]+25))
		right_point = (int(max_x_y[0]+25),int(min_x_y[1]-25))
		#print(left_point,right_point)
		#print('slicing is {}:{},{}:{}'.format(right_point[1],left_point[1],left_point[0],right_point[0]))
		#print(img2.shape)
		#cv2.rectangle(img2, left_point, right_point, (255,0,0), 2)
		#plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), 'gray'),plt.show()
		#roi = img2[right_point[1]:left_point[1],right_point[0]:left_point[0]]
		roi_img2 = img2[right_point[1]:left_point[1],left_point[0]:right_point[0]]
		h,w,d = roi_img2.shape
		resized_img1 = cv2.resize(img1, (w,h), interpolation=cv2.INTER_AREA)
		
		self.neural_network_obj.image_reader(cv2.cvtColor(resized_img1, cv2.COLOR_BGR2RGB),cv2.cvtColor(roi_img2, cv2.COLOR_BGR2RGB))
		self.neural_network_obj.main()

		
		
		'''
		extLeft = tuple(dst_pts[dst_pts[:, :, 0].argmin()][0])
		extRight = tuple(dst_pts[dst_pts[:, :, 0].argmax()][0])
		extTop = tuple(dst_pts[dst_pts[:, :, 1].argmin()][0])
		extBot = tuple(dst_pts[dst_pts[:, :, 1].argmax()][0])
		print([extLeft,extTop,extRight,extBot])
		'''
		#for point in dst_pts:
			#print(point)
			#cv2.circle(img2, (int(point[0]),int(point[1])), radius=1, color=(0, 255, 0), thickness=-1)
		


	def compare_images(self,imageA, imageB):
		# compute the mean squared error and structural similarity
		# index for the images
		h,w,d = imageB.shape
		resizedA = cv2.resize(imageA, (w,h), interpolation=cv2.INTER_AREA)
		# setup the figure
		fig = plt.figure('compare')
		plt.suptitle("MSE: %.2f, SSIM: %.2f" % (7,8))
		# show first image
		ax = fig.add_subplot(1, 2, 1)
		plt.imshow(imageA, cmap = plt.cm.gray)
		plt.axis("off")
		# show the second image
		ax = fig.add_subplot(1, 2, 2)
		plt.imshow(imageB, cmap = plt.cm.gray)
		plt.axis("off")
		# show the images
		plt.show()



	def bruteMatcher(self,kp1,des1,kp2,des2,img2,img1):
		# create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		# Match descriptors.
		#try:
		
		matches = bf.match(des1,des2)
		# Sort them in the order of their distance.
		#matches = sorted(matches, key = lambda x:x.distance)

		#print('len of matches: ',len(matches))
		#print('len of des1: ',len(des1))
		ratio = len(matches) * 100/len(des1)
		#print('ratio: ',ratio)
		if ratio > 4.0:
			self.test(kp1,des1,img1,kp2,des2,img2, matches)
			
			
		#except Exception as e:
			#print("BruteMathcer",e)
		
		#plt.imshow(img2, 'gray'),plt.show()
	

	def bruteMatcherKnn(self,kp1,des1,kp2,des2,img2,img1):

		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1,des2, k=2)

		# Apply ratio test
		
		good = []
		for m,n in matches:
			if m.distance < 0.75 * n.distance:
				good.append(m)
		
		ratio = int((len(good)/len(des1)) * 100)
		#print('len of matches: ',len(good))
		#print('len of des1: ',len(des1))
		#print('ratio: ',ratio)
		# cv2.drawMatchesKnn expects list of lists as matches.
		
		#if ratio > 1:
			#self.homography(kp1,des1,img1,kp2,des2,img2, good)
		img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
		plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB), 'gray'),plt.show()
	

	def new_bruteMatcher(self,kp1,des1,kp2,des2,img2,img1):
		# create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		# Match descriptors.
		#try:
		
		matches = bf.match(des1,des2)
		# Sort them in the order of their distance.
		#matches = sorted(matches, key = lambda x:x.distance)

		#print('len of matches: ',len(matches))
		#print('len of des1: ',len(des1))
		ratio = len(matches) * 100/len(des1)
		#print('ratio: ',ratio)
		return matches

	def FlannMatcher(self,kp1,des1,kp2,des2,img2,img1):
			FLANN_INDEX_KDTREE = 1


			FLANN_INDEX_LSH = 6
			index_params= dict(algorithm = FLANN_INDEX_LSH,
							table_number = 6, # 12
							key_size = 12,     # 20
							multi_probe_level = 1) #2
			search_params = dict(checks=50)   # or pass empty dictionary

			#print(len(des1),len(des2))
			flann = cv2.FlannBasedMatcher(index_params,search_params)
			#try:
			matches = flann.knnMatch(des1,des2,k=2)
			good = []
		
			#print(matches)
			try:
				for m,n in matches:
					if m.distance < 0.75 * n.distance:
						good.append(m)
			except Exception as e:
			#	print('###############')
				print("Flann",e)
			#	print(matches)
			if len(good) > 9:
				self.homography(kp1,des1,img1,kp2,des2,img2, good)
					
			#except Exception as e:
				#print(e)
				#pass
			
			
			#print('FLANN')
			#print(matches)
			#print("####################")
			#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=2)
			#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
			#cv2.imshow('flann', img3)
			#cv2.waitKey(0)
		
			#cv2.destroyAllWindows()


	def dbscan(self,kp):

		kp_points = []
		for point in kp:

			kp_points.append(list(point.pt))
		
		X = StandardScaler().fit_transform(kp_points)

		#db = DBSCAN(eps=0.05, min_samples=10).fit(X)
		db = DBSCAN(eps=0.09, min_samples=45).fit(X)
		core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		n_noise_ = list(labels).count(-1)

		print('Estimated number of clusters: %d' % n_clusters_)
		print('Estimated number of noise points: %d' % n_noise_)

		return core_samples_mask,labels



	def drawer(self,kp1,des1,kp_clusters,des_cluster,img1_object,img2_object):
		img1 = img1_object.img.copy()
		img2 = img2_object.img.copy()
		img1_grey = img1_object.img_grey.copy()
		img2_grey = img2_object.img_grey.copy()
		colors_list = uniquecolors(200)
		#i = 0
		#print('clusters len: ', len(kp_clusters))
		for cluster_kp, cluster_des in zip(kp_clusters,des_cluster):
			#print("color:",colors_list[i])
			
			self.bruteMatcher(kp1,des1,cluster_kp,cluster_des,img2,img1)
			#self.test(kp1,des1,img1,cluster_kp,cluster_des,img2, matches)
			'''
			if match_dst is not None:
				pass
				#img2 = cv2.polylines(img2,[np.int32(match_dst)],True,0,3, cv2.LINE_AA)
			#print('coloring with ',i)
			
			for kp_point in cluster_kp:
				point = kp_point.pt
				img2 = cv2.circle(img2, (int(point[0]),int(point[1])), radius=1, color=colors_list[i], thickness=-1)
			i +=1
			'''
		#plt.figure(2)
		#plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)),plt.show()
		#cv2.imshow('',img2)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()


	def dbscan_testing(self,kp):
		

		plt.figure(1)	
		kp_points = []
		for point in kp:

			kp_points.append(list(point.pt))
		
		X = StandardScaler().fit_transform(kp_points)
		'''
		neighbors = NearestNeighbors(n_neighbors=10)
		neighbors_fit = neighbors.fit(kp_points)
		distances, indices = neighbors_fit.kneighbors(kp_points)
		distances = np.sort(distances, axis=0)
		distances = distances[:,1]
		plt.plot(distances)
		'''

		db = DBSCAN(eps=0.09, min_samples=45).fit(X)
		#db = DBSCAN(eps=0.05, min_samples=10).fit(X)
		core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		n_noise_ = list(labels).count(-1)

		print('Estimated number of clusters: %d' % n_clusters_)
		print('Estimated number of noise points: %d' % n_noise_)
	
		# #############################################################################
		# Plot result

		# Black removed and is used for noise instead.
		unique_labels = set(labels)
		colors = [plt.cm.Spectral(each)
				for each in np.linspace(0, 1, len(unique_labels))]
		for k, col in zip(unique_labels, colors):
			if k == -1:
				# Black used for noise.
				col = [0, 0, 0, 1]

			class_member_mask = (labels == k)

			xy = X[class_member_mask & core_samples_mask]
			plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
					markeredgecolor='k', markersize=6)

			xy = X[class_member_mask & ~core_samples_mask]
			plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
					markeredgecolor='k', markersize=1)
		
		plt.gca().invert_yaxis()
		plt.title('Estimated number of clusters: %d' % n_clusters_)
		plt.show()
		
	def imager(self,img1_object,img2_object):
		img1 = img1_object.img
		img2 = img2_object.img
		img1_grey = img1_object.img_grey
		img2_grey = img2_object.img_grey
		kp1, des1 = self.orb_features(img1_object)
		kp2, des2 = self.orb_features(img2_object)
		inlied_kp1 ,inlied_des1=self.outlierKP(kp1,des1)
		inlied_kp2, inlied_des2 = self.outlierKP(kp2,des2)

		'''
		verify oulier algorithm output
		######################################################


		for kp_point in inlied_kp1:
			point = kp_point.pt
			#bubu = (point[0],point[1])
			#print(bubu)
			image_inlied_kp1 = cv2.circle(img1, (int(point[0]),int(point[1])), radius=1, color=(0, 255, 0), thickness=-1)

		
		for kp_point in inlied_kp2:
			point = kp_point.pt
			#bubu = (point[0],point[1])
			#print(bubu)
			image_inlied_kp2 = cv2.circle(img2, (int(point[0]),int(point[1])), radius=1, color=(0, 255, 0), thickness=-1)

		
		for kp_point in kp1:
			point = kp_point.pt
			#bubu = (point[0],point[1])
			#print(bubu)
			image_kp1 = cv2.circle(img1, (int(point[0]),int(point[1])), radius=1, color=(255, 0, 0), thickness=-1)

		for kp_point in kp2:
			point = kp_point.pt
			#bubu = (point[0],point[1])
			#print(bubu)
			image_kp2 = cv2.circle(img2, (int(point[0]),int(point[1])), radius=1, color=(255, 0, 0), thickness=-1)

		plt.subplot(2, 2, 1)
		plt.imshow(image_inlied_kp1)
		plt.subplot(2, 2, 2)
		plt.imshow(image_inlied_kp2)
		plt.subplot(2, 2, 3)
		plt.imshow(image_kp1)
		plt.subplot(2, 2, 4)
		plt.imshow(image_kp2)
		plt.show()


		#########################################################
		'''

		core_masks_kp2,labels_kp2 = self.dbscan(inlied_kp2)
		self.dbscan_testing(inlied_kp2)
		unique_labels2 = set(labels_kp2)

		kp_clusters = []
		des_cluster = []
		cluster_matches = []
		for k in zip(unique_labels2):
			class_member_mask = (labels_kp2 == k)


			kp_mask = [class_member_mask & core_masks_kp2]
			cluster_kp = []
			cluster_des = []
			for i in range(len(inlied_kp2)):
				if kp_mask[0][i] == True:
					cluster_kp.append(inlied_kp2[i])
					cluster_des.append(inlied_des2[i])
				else:
					pass

			kp_clusters.append(cluster_kp)
			des_cluster.append(np.array(cluster_des))


			kp_mask =[class_member_mask & ~core_masks_kp2]
			cluster_kp = []
			for i in range(len(inlied_kp2)):
				if kp_mask[0][i] == True:
					cluster_kp.append(inlied_kp2[i])
				else:
					pass	
		self.drawer(kp1,des1,kp_clusters,des_cluster,img1_object,img2_object)



	def outlierKP(self,kp,des):
		kp_points = []
		for point in kp:
			kp_points.append(list(point.pt))
		clf = LocalOutlierFactor(n_neighbors=500)
		ret = clf.fit_predict(kp_points)
		inlied_kp = []
		inlied_des = []
		for i in range(len(kp_points)):
			if ret[i] == -1:
				pass
			else:
				inlied_des.append(des[i])
				inlied_kp.append(kp[i])
		print('incoming kp len:{}  outgoing kp len: {}'.format(len(kp),len(inlied_kp)))
		return inlied_kp, inlied_des


		


	
if __name__ == '__main__':
	diff_image = CVrunner()
	diff_image.main()
