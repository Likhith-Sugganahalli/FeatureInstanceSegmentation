	def SIFT(self,img_object):
		# Initiate SIFT detector
		img = img_object.img_grey
		sift = cv2.SIFT()

		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(img,None)
		return kp1 , des1

	def kmean(self,img1_object,img2_object):
		img1 = img1_object.img
		img2 = img2_object.img
		img1_grey = img1_object.img_grey
		img2_grey = img2_object.img_grey
		

		# find the keypoints and descriptors with ORB
		kp2, des2 = self.orb_features(img2_object)
		x = np.array([kp2[0].pt])

		for i in range(len(kp2)):
			x = np.append(x, [kp2[i].pt], axis=0)

		x = x[1:len(x)]
		# define criteria and apply kmeans()
		print(x)
		#samples = cv2.fromarray(x)
		des = np.float32(x)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		compactness, labels, centers=cv2.kmeans(des,20,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
		# Now separate the data, Note the flatten()
		for point in x:
			#bubu = (point[0],point[1])
			#print(bubu)
			image = cv2.circle(img2, (int(point[0]),int(point[1])), radius=1, color=(0, 255, 0), thickness=-1)
		for point in centers:
			image1 = cv2.circle(image, (int(point[0]),int(point[1])), radius=5, color=(255, 0, 0), thickness=-1)
		cv2.imshow('flann', image1)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	def watershed(self,img_object):
		img_grey = img_object.img_grey
		img = img_object.img
		ret, thresh = cv2.threshold(img_grey,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		# noise removal
		kernel = np.ones((4,4),np.uint8)
		opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
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
		cv2.imshow('img', img)
		#contours


	def FlannMatcher(self,kp1,des1,kp2,des2,img2,img1):
		FLANN_INDEX_KDTREE = 1


		FLANN_INDEX_LSH = 6
		index_params= dict(algorithm = FLANN_INDEX_LSH,
						table_number = 6, # 12
						key_size = 12,     # 20
						multi_probe_level = 1) #2
		search_params = dict(checks=50)   # or pass empty dictionary


		flann = cv2.FlannBasedMatcher(index_params,search_params)
		matches = flann.knnMatch(des1,des2,k=2)
		print('FLANN')
		print(matches)
		print("####################")
		img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=2)
		#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		cv2.imshow('flann', img3)
		cv2.waitKey(0)
	
		cv2.destroyAllWindows()
    
	def bruteMatcherKnn(self,kp1,des1,kp2,des2,img2,img1):

		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1,des2, k=2)

		# Apply ratio test
		good = []
		for m,n in matches:
			if m.distance < 0.75*n.distance:
				good.append([m])
		print('len of good: ',len(good))
		print('len of des2: ',len(des1))
		print('ratio: ',len(good)/len(des1))
		# cv2.drawMatchesKnn expects list of lists as matches.
		img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)


def keypointContours(self,img1,img2):
		img1_grey = img1.img_grey	
		img2_grey = img2.img_grey	
		img1_img = img1.img
		img2_img = img2.img
		# Find Canny edges
		approx_number_of_objects = int(15 * 1.5)
		kernel = np.ones((3,3),np.uint8)

		opening2 = cv2.morphologyEx(img2_grey, cv2.MORPH_OPEN, kernel)


		edged2 = cv2.Canny(img2_grey, 50, 200)
		edged1 = cv2.Canny(img1_grey, 50, 200)

		

		gradient1 = cv2.morphologyEx(edged1, cv2.MORPH_GRADIENT, kernel)
		#erosion1 = cv2.erode(gradient1,kernel,iterations = 1)

		gradient2 = cv2.morphologyEx(edged2, cv2.MORPH_GRADIENT, kernel)
		erosion2 = cv2.erode(gradient2,kernel,iterations = 1)
		
		#print(type(edged2))

		#self.bruteMatch(gradient1,gradient2)

		'''
		cv2.imshow('gradient1', gradient1)
		cv2.imshow('gradient2', gradient2)

		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
		th1 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

		#th2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

		blur = cv2.GaussianBlur(img2,(5,5),0)
		ret3,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		_,otsu_th2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		_,otsu_th1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


		self.bruteMatch(otsu_th1,otsu_th2)
		'''

		#ret, thresh = cv2.threshold(img2, 127, 255, 0)

		# Finding Contours
		# Use a copy of the image e.g. edged.copy()
		# since findContours alters the image
		contours1, hierarchy1 = cv2.findContours(gradient1, 
			cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contours2, hierarchy2 = cv2.findContours(erosion2, 
			cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		cv2.imshow('erosion2', erosion2)
		cv2.imshow('opening2', opening2)
		cv2.drawContours(img2_img, contours2, -1, (0, 255, 0), 3)
		cv2.imshow('contours', img2_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
				
		'''
		# Draw all contours
		# -1 signifies drawing all contours
		contours2 = sorted(contours2,key=cv2.contourArea,reverse=True)

		contours1 = sorted(contours1,key=cv2.contourArea,reverse=True)

		best_of_contours2 = contours2[:approx_number_of_objects]
		shaped_contours_2 = []
		areaed_contours_2 = []
		
		lower_range = 400#cv2.contourArea(contours1[0]) * 2
		higher_range = 231787#cv2.contourArea(contours1[0]) * 0.1

		print("{} {} {}".format(cv2.contourArea(contours1[0]),lower_range,higher_range))
		print(len(best_of_contours2))
		for cnt in best_of_contours2:
			print(cv2.contourArea(cnt))

			
			epsilon = 0.01*cv2.arcLength(cnt,True)
			cont = cv2.approxPolyDP(cnt,epsilon,True)
			img2_copy = img2.copy()
			#shaped_contours_2.append(cv2.approxPolyDP(cnt,epsilon,True))
			cv2.drawContours(img2_copy, [cnt], -1, (255, 0, 0), 3)
			cv2.drawContours(img2_copy, [cont], -1, (0, 255, 0), 3)
			cv2.imshow('img2', img2_copy)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			

		


		

		if lower_range< cv2.contourArea(cnt) <higher_range:
				areaed_contours_2.append(cnt)

		cv2.drawContours(img2, shaped_contours_2, -1, (0, 255, 0), 3)

		cv2.drawContours(img1, contours1, -1, (0, 255, 0), 3)

		#cv2.imshow('gradient2', erosion2)
		#print(cv2.contourArea(contours1[0]))
		
		cv2.imshow('img2', img2)
		cv2.imshow('img1', img1)
		print("Number of contours2 found = " + str(len(best_of_contours2)))
		print("Number of contours1 found = " + str(len(contours1)))
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		#plt.imshow(img3),plt.show()
		'''


	def newcontours(self,markers,img_object):
		img = img_object.img
		kernel = np.ones((4,4),np.uint8)
		img2 = img.copy()
		markers1 = markers.astype(np.uint8)
		ret, m2 = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
		cv2.imshow('m2', m2)
		gradient1 = cv2.morphologyEx(m2, cv2.MORPH_GRADIENT, kernel)
		cv2.imshow('gradient1', gradient1)
		contours, hierarchy = cv2.findContours(m2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   
		contourAreas = [] 
		print(img_object.img_area)
		for c in contours:
			
			print('##############')
			area = cv2.contourArea(c)
			if area not in contourAreas:
				contourAreas.append(area)
				threshold = img_object.img_area * 0.5
				print(area)
				if area > threshold:
					print('big')
				else:
					temp = img2.copy()
					cv2.drawContours(temp, c, -1, (255, 0, 0), 2)
					cv2.imshow('contours', temp)
					cv2.waitKey(0)
					cv2.destroyAllWindows()
			else:
				pass


def multipleOrb(self,img1,img2):
		MIN_MATCH_COUNT = 10
		orb = cv2.ORB_create(10000, 1.2, nlevels=8, edgeThreshold = 5)

		# find the keypoints and descriptors with ORB
		kp1, des1 = orb.detectAndCompute(img1, None)
		kp2, des2 = orb.detectAndCompute(img2, None)
		

		x = np.array([kp2[0].pt])

		for i in range(len(kp2)):
			x = np.append(x, [kp2[i].pt], axis=0)

		x = x[1:len(x)]

		bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)

		ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
		ms.fit(x)
		labels = ms.labels_
		cluster_centers = ms.cluster_centers_

		labels_unique = np.unique(labels)
		n_clusters_ = len(labels_unique)
		print("number of estimated clusters : %d" % n_clusters_)

		s = [None] * n_clusters_
		for i in range(n_clusters_):
			l = ms.labels_
			d, = np.where(l == i)
			print(d.__len__())
			s[i] = list(kp2[xx] for xx in d)

		des2_ = des2

		for i in range(n_clusters_):

			kp2 = s[i]
			l = ms.labels_
			d, = np.where(l == i)
			des2 = des2_[d, ]

			FLANN_INDEX_KDTREE = 0
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
			search_params = dict(checks = 50)

			flann = cv2.FlannBasedMatcher(index_params, search_params)

			des1 = np.float32(des1)
			des2 = np.float32(des2)

			matches = flann.knnMatch(des1, des2, 2)

			# store all the good matches as per Lowe's ratio test.
			good = []
			for m,n in matches:
				if m.distance < 0.7*n.distance:
					good.append(m)

			if len(good)>3:
				src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
				dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

				M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)

				if M is None:
					print ("No Homography")
				else:
					matchesMask = mask.ravel().tolist()

					h,w = img1.shape
					pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
					dst = cv2.perspectiveTransform(pts,M)

					img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

					draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
									singlePointColor=None,
									matchesMask=matchesMask,  # draw only inliers
									flags=2)

					img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

					plt.imshow(img3, 'gray'), plt.show()

			else:
				print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
				matchesMask = None


def mean_clustering(self,img1_object,img2_object):
		img1 = img1_object.img
		img2 = img2_object.img
		img1_grey = img1_object.img_grey
		img2_grey = img2_object.img_grey
		

		# find the keypoints and descriptors with ORB
		kp2, des2 = self.orb_features(img2_object)

		#kp1, des1 = orb.detectAndCompute(img1_grey, None)
		#image_kp = cv2.circle(img2, kp2, radius=1, color=(255, 0, 0), thickness=-1)
		x = self.outlierKP(kp2)
		bandwidth = estimate_bandwidth(x, quantile=0.1)

		ms = MeanShift(bandwidth=bandwidth)
		ms.fit(x)
		labels = ms.labels_
		cluster_centers = ms.cluster_centers_
		#print(x)
		print('############')
		print(cluster_centers)
		for point in x:
			#bubu = (point[0],point[1])
			#print(bubu)
			image = cv2.circle(img2, (int(point[0]),int(point[1])), radius=1, color=(0, 255, 0), thickness=-1)

		for point in cluster_centers:
			image1 = cv2.circle(image, (int(point[0]),int(point[1])), radius=10, color=(255, 0, 0), thickness=-1)
		labels_unique = np.unique(labels)
		n_clusters_ = len(labels_unique)
		print("number of estimated clusters : %d" % n_clusters_)
		cv2.imshow('flann', image1)
		cv2.waitKey(0)
		cv2.destroyAllWindows()



	def unique_count_app(self,a):
		a2D = a.reshape(-1,a.shape[-1])
		col_range = (256, 256, 256) # generically : a2D.max(0)+1
		a1D = np.ravel_multi_index(a2D.T, col_range)
		return np.unravel_index(np.bincount(a1D).argmax(), col_range)
    

