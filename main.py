import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from ImageSim.ResNet50Sim import ResNetSim
from ImageSim.VGG16Sim import VGGSim
import pytesseract


#from OS2D.neural import Os2d

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
		#OS2D is commented right now, due to it throwing up CUDA memory problems out of the blue
		#self.Os2d = Os2d()
		print('setting up VGG')
		self.VGGSim = VGGSim()
		print('setting up ResNet')
		self.ResNetSim = ResNetSim()

	def load_images_from_folder(self,folder):
		for filename in os.listdir(folder):
			img = MyImage(os.path.join(folder,filename))
			if img is not None:
				yield(img)
				#print(type(img))

	def main(self):
		for img1 in self.template_imgs:
			for img2 in self.test_imgs:
				self.runner(img1,img2)

	def orb_features(self,img_object):

		img = img_object.img

		orb2 = cv2.ORB_create(10000, 1.2, nlevels=8, edgeThreshold = 5)

		kp, des = orb2.detectAndCompute(img, None)

		return kp,des

	def homography(self,kp1,img1,kp2,img2,good):
		MIN_MATCH_COUNT = 10
		img2_copy = img2#.copy()
		if len(good)>MIN_MATCH_COUNT:

			src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,4.4)
			matchesMask = mask.ravel().tolist()
			h,w,d = img1.shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			dst = cv2.perspectiveTransform(pts,M)
			#print(, )
			#cv2.rectangle(img2_copy,, (255,0,0), 2)
			pt1 = [int(dst[0][0][0]),int(dst[0][0][1])]
			pt2 = [int(dst[1][0][0]),int(dst[1][0][1])]
			pt3 = [int(dst[2][0][0]),int(dst[2][0][1])]
			pt4 = [int(dst[3][0][0]),int(dst[3][0][1])]
			img2_copy = cv2.circle(img2_copy,(pt1[0],pt1[1]), radius=5, color=(0, 0, 255), thickness=-1)
			img2_copy = cv2.circle(img2_copy,(pt2[0],pt2[1]), radius=5, color=(0, 0, 255), thickness=-1)
			img2_copy = cv2.circle(img2_copy,(pt3[0],pt3[1]), radius=5, color=(0, 0, 255), thickness=-1)
			img2_copy = cv2.circle(img2_copy,(pt4[0],pt4[1]), radius=5, color=(0, 0, 255), thickness=-1)
			img2_copy = cv2.polylines(img2_copy,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		else:
			print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
			matchesMask = None

		draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   matchesMask = matchesMask, # draw only inliers
				   flags = 2)
		img3 = cv2.drawMatches(img1,kp1,img2_copy,kp2,good,None,**draw_params)
		plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB), 'gray'),plt.show()
	
	def watershed(self,img):
		img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#img = img_object.img
		ret, thresh = cv2.threshold(img_grey,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		# noise removal
		kernel = np.ones((3,3),np.uint8)
		opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

		#img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
		print(pytesseract.image_to_string(thresh))
		# sure background area
		sure_bg = cv2.dilate(opening,kernel,iterations=3)
		# Finding sure foreground area
		dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
		ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
		# Finding unknown region
		sure_fg = np.uint8(sure_fg)
		unknown = cv2.subtract(sure_bg,sure_fg)
		# Marker labelling
		ret, markers = cv2.connectedComponents(sure_fg)
		# Add one to all labels so that sure background is not 0, but 1
		markers = markers+1
		# Now, mark the region of unknown with zero
		markers[unknown==255] = 0
		markers = cv2.watershed(img,markers)
		img[markers == -1] = [255,0,0]
		cv2.imshow('thresh', thresh)
		cv2.imshow('opening', opening)
		cv2.imshow('sure_bg', sure_bg)
		cv2.imshow('sure_fg', sure_fg)
		cv2.imshow('markers', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def sklearn_watershed(self,roi):
		image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		# Now we want to separate the two objects in image
		# Generate the markers as local maxima of the distance to the background
		distance = ndi.distance_transform_edt(image)
		coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
		mask = np.zeros(distance.shape, dtype=bool)
		mask[tuple(coords.T)] = True
		markers, _ = ndi.label(mask)
		labels = watershed(-distance, markers, mask=image)

		fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
		ax = axes.ravel()

		ax[0].imshow(roi, cmap=plt.cm.gray)
		ax[0].set_title('Overlapping objects')
		ax[1].imshow(-distance, cmap=plt.cm.gray)
		ax[1].set_title('Distances')
		ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
		ax[2].set_title('Separated objects')

		for a in ax:
			a.set_axis_off()

		fig.tight_layout()
		plt.show()

	def test(self,kp1,des1,img1,kp2,des2,img2, matches):

		src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])
		#print(dst_pts)
		max_x_y = np.amax(dst_pts, axis=0)
		min_x_y = np.amin(dst_pts, axis=0)
		h,w,d = img2.shape

		left_point = (int(min_x_y[0]-25) if int(min_x_y[0]) > 25 else 0,int(max_x_y[1]+25) if int(max_x_y[1]+25) < h else h)
		right_point = (int(max_x_y[0]+25) if int(max_x_y[0]+25) < w else w,int(min_x_y[1]-25) if int(min_x_y[1]-25) > 25 else 0)
		#cv2.rectangle(img2, left_point, right_point, (255,0,0), 2)
		roi_img2 = img2[right_point[1]:left_point[1],left_point[0]:right_point[0]]

		#h,w,d = roi_img2.shape
		#resized_img1 = cv2.resize(img1, (h,w), interpolation=cv2.INTER_AREA)
		#print(roi_img2.shape)
		return roi_img2

	def bruteMatcher(self,kp1,des1,kp2,des2):
		# create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		try:
			matches = bf.match(des1,des2)

			return matches
		except:
			pass
	
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
	
	def runner(self,img1_object,img2_object):

		kp1, des1 = self.orb_features(img1_object)
		kp2, des2 = self.orb_features(img2_object)

		inlied_kp1 ,inlied_des1=self.outlierKP(kp1,des1)
		inlied_kp2, inlied_des2 = self.outlierKP(kp2,des2)

		kp2_clusters,des2_clusters = self.grouper(inlied_kp2,inlied_des2)
		self.refiner(kp1,des1,kp2_clusters,des2_clusters,img1_object,img2_object)
	
	def refiner(self,kp1,des1,kp2_clusters,des2_clusters,img1_object,img2_object):
		img1 = img1_object.img
		img2 = img2_object.img
		rois = []
		for cluster_kp, cluster_des in zip(kp2_clusters,des2_clusters):
			matches = self.bruteMatcher(kp1,des1,cluster_kp,cluster_des)
			if matches is None:
				pass
			else:
				ratio = len(matches) * 100/len(des1)
				if ratio > 4.0:

					#Uncomment for CNNs
					roi = self.test(kp1,des1,img1,cluster_kp,cluster_des,img2, matches)
					rois.append(roi)

					#uncomment for bounding boxes
					self.homography(kp1,img1,cluster_kp,img2,matches)

					#self.watershed(roi)

					
		
		if len(rois) > 0:
			resized_img1 = cv2.resize(img1, (224,224))
			fig = plt.figure(constrained_layout = True)
			rows = int(len(rois)/2) + 1
			cols = 2
			for i,roi in enumerate(rois):
				#uncomment to view rois individually
				#plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)),plt.show()


				resized_roi = cv2.resize(roi, (224,224))

				#self.Os2d.image_reader(img1,roi)
				#self.Os2d.main()

				#Using VGG16
				#ret = self.VGGSim.main([cv2.cvtColor(resized_img1, cv2.COLOR_BGR2RGB),cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)])

				#Using ResNet50
				#ret = self.ResNetSim.main([cv2.cvtColor(resized_img1, cv2.COLOR_BGR2RGB),cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)])

				#ret = [round(x,2) for x in ret]

				##########################
				#roi plot for CNNs
				##########################
				'''
				fig.add_subplot(rows, cols, i+1)
				plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
				plt.axis('off')
				plt.title('sim_ecli:{} sim_cosine:{}'.format(ret[0],ret[1]))
				'''
			#uncomment for roi plots
			#plt.show(block=True)

	def grouper(self,inlied_kp,inlied_des):

		core_masks_kp2,labels_kp2 = self.dbscan(inlied_kp)
		unique_labels2 = set(labels_kp2)

		kp_clusters = []
		des_clusters = []

		for k in zip(unique_labels2):
			class_member_mask = (labels_kp2 == k)

			kp_mask = [class_member_mask & core_masks_kp2]
			cluster_kp = []
			cluster_des = []
			for i in range(len(inlied_kp)):
				if kp_mask[0][i] == True:
					cluster_kp.append(inlied_kp[i])
					cluster_des.append(inlied_des[i])
				else:
					pass

			kp_clusters.append(cluster_kp)
			des_clusters.append(np.array(cluster_des))


			kp_mask =[class_member_mask & ~core_masks_kp2]
			cluster_kp = []
			for i in range(len(inlied_kp)):
				if kp_mask[0][i] == True:
					cluster_kp.append(inlied_kp[i])
				else:
					pass
		return kp_clusters,des_clusters

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
