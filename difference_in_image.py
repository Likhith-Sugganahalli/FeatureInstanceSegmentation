# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import argparse
import imutils
import cv2
import os
from matplotlib import pyplot as plt
import os
import itertools

class MyImage:
	def __init__(self, img_name):
		self.img = cv2.imread(img_name)
		self.__name = img_name

	def __str__(self):
		return self.__name


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b) 


class Diff_image():
	"""
	docstring for Diff_image
	"""
	def __init__(self):
		self.images = []
		self.working_dir = os.path.dirname(os.path.realpath(__file__))


	def load_images_from_folder(self,folder):
		for filename in os.listdir(folder):
			img = MyImage(os.path.join(folder,filename))
			if img is not None:
				self.images.append(img)
				#print(type(img))


	def main(self):
		photos_folder = os.path.join(self.working_dir,'photos')
		self.load_images_from_folder(photos_folder)
		"""
		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-f", "--first", required=True,help="first input image")
		# load the two input images
		imageA = cv2.imread(args["first"])
		"""
		



		#imageB = cv2.imread()


		iteratable_images = pairwise(self.images)
		for first_image,second_image in iteratable_images:
			print(first_image)
			print(second_image)


			imageA = cv2.imread(str(first_image))

			fit_a = cv2.resize(imageA, (640, 360),  interpolation = cv2.INTER_NEAREST)
			blur_a = cv2.blur(fit_a,(5,5)) 
			grayA = cv2.cvtColor(blur_a, cv2.COLOR_BGR2GRAY)


			#print("gbubb {}".format(type(second_image)))
			fit_b = cv2.resize(second_image.img, (640, 360),  interpolation = cv2.INTER_NEAREST)
			blur_b = cv2.blur(fit_b,(5,5))
			# convert the images to grayscale
			grayB = cv2.cvtColor(blur_b, cv2.COLOR_BGR2GRAY)
			# compute the Structural Similarity Index (SSIM) between the two
			# images, ensuring that the difference image is returned
			(score, diff) = ssim(grayA, grayB, full=True)
			diff = (diff * 255).astype("uint8")
			print("SSIM: {} for {}".format(score, str(second_image)))

			# threshold the difference image, followed by finding contours to
			# obtain the regions of the two input images that differ
			thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)
		
			# loop over the contours
			for c in cnts:
				# compute the bounding box of the contour and then draw the
				# bounding box on both input images to represent where the two
				# images differ
				(x, y, w, h) = cv2.boundingRect(c)
				cv2.rectangle(fit_a, (x, y), (x + w, y + h), (0, 0, 255), 2)
				cv2.rectangle(fit_b, (x, y), (x + w, y + h), (0, 0, 255), 2)
		# show the output images
		
		cv2.imshow("Original_a", fit_a)
		cv2.imshow("Original_b", fit_b)

		cv2.imshow("blur_a", blur_a)

		cv2.imshow("blur_b", blur_b)

		cv2.imshow("Diff", diff)
		cv2.imshow("Thresh", thresh)
		cv2.waitKey(0)
		

if __name__ == '__main__':
	diff_image = Diff_image()
	diff_image.main()