Links:

google docs:https://docs.google.com/document/d/1B3WY0HRutYEf5ACodvgU9-S1XU5fsKQrKeeJFAOtFWU/edit?usp=sharing

https://github.com/XingLiangLondon/Image-Similarity-in-Percentage
https://github.com/aosokin/os2d

Feature Matching:

Input to the program
    The template image
    The test images


Output of the program
    images with as many instances of the object ( of interest) as possible marked by a bounding box


Methods investigated:
As the problem is an ongoing research topic, there was no one main concept to identify and apply,
so i choose to initial commit to 4 main ideas to proceed with,

Contour Matching and all such applications which worked to redefine the test image in terms of the input image

Feature Matching, this generally involves cases where the object is know to be present as feature matching only helps in figuring out the orientation and the       position of the object.But, i utilized feature matching to define feature rich patches in the test image, allowing for object discovery, more specifically        interesting feature patches.

Neural Networks, this is a broad category, becasuse i tested various neural networks that would be useful in different stages of the process.
    1) Object discovery
    2) Similarity Matching
    3) One Shot Object detection

All three uses are very promising and require more in-depth research, all 3 applications are included in the git repo, ResNet50 and VGG16 were used to obtain feature vector of the test image while OS2D is a One shot Object detection Neural Network specifically  designed to work with groceries and similar products,

ResNet50 and VGG16 can be run directly from the main.py while the OS2D is not completely included in the repo, only the python code I modified to work with main.py

Instructions to separately setup the OS2D repo are included.

A brief list of concepts utilized, considered in the process of creating this repo:

    Feature Matchers (BF, BfKnn, and Flann)
    Feature Descriptors (SIFT and ORB)
    Mean-Shift clustering as an initial clustering algorithm for keypoints, and Kmeans to verify clustering potential
    Structural Similarity Index
    DBScan for clustering keypoints
    Local Outlier Factor Algorithm to reduce noise and improve clustering efficiency 

    Canny Edge Algorithm
    Morphological Transformations
    Watershed Algorithm
    Image Thresholding
    Image Filtering
    Contour Detection


Instructions

To run ROI Similarity using cosine similarity and euclidean similarity

Uncomment Lines 243 and 244

With ResNet:
Uncomment the following on Line 272

#ret = self.ResNetSim.main([cv2.cvtColor(resized_img1, cv2.COLOR_BGR2RGB),cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)])

With VGG16:
Uncomment the following on Line 269

#ret = self.VGGSim.main([cv2.cvtColor(resized_img1, cv2.COLOR_BGR2RGB),cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)])

then uncomment Lines 274,286 and section 279-284

Homography can be run with, or without improve_homography, which is an attempt to redraw the perspectiveTransform into a more accurate sqaure, the function is still a work in progress, and is not as accurate as needed,
