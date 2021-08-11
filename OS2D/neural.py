
import os
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from OS2D.os2d.modeling.model import build_os2d_from_config
from OS2D.os2d.config import cfg
import OS2D.os2d.utils.visualization as visualizer
from OS2D.os2d.structures.feature_map import FeatureMapSize
from OS2D.os2d.utils import setup_logger, read_image, get_image_size_after_resize_preserving_aspect_ratio,read_image_from_array

class Os2d():

	def __init__(self):
		self.logger = setup_logger("OS2D")
		cfg.is_cuda = torch.cuda.is_available()
		cfg.init.model = "/home/whoknows/Documents/FeatureMatching-Python/OS2D/models/os2d_v2-train.pth"
		self.net, self.box_coder, self.criterion, self.img_normalization, self.optimizer_state = build_os2d_from_config(cfg)
		
	
	def imager_loader(self, class_image_location_list, test_image_location):
		self.class_ids = []
		self.class_images = []
		print('hi {}'.format(test_image_location))
		self.input_image = read_image(test_image_location)
		#print('here')
		
		for i,image in enumerate(class_image_location_list):
			#print(image)
			self.class_images.append(read_image(image))
			self.class_ids.append(i)


	def image_reader(self,template_img,roi):
		self.class_ids = []
		self.class_images = []
		self.class_images.append(read_image_from_array(template_img))
		self.class_ids = [0]
		#for i,roi in enumerate(rois_list):
		self.input_image = read_image_from_array(roi)


	def main(self,resize_test=1500):
		#torch.cuda.clear_memory_allocated()

		transform_image = transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize(self.img_normalization["mean"], self.img_normalization["std"])
						])

		h, w = get_image_size_after_resize_preserving_aspect_ratio(h=self.input_image.size[1],
																	w=self.input_image.size[0],
																	target_size=resize_test)
		self.input_image = self.input_image.resize((w, h))

		input_image_th = transform_image(self.input_image)
		input_image_th = input_image_th.unsqueeze(0)
		if cfg.is_cuda:
			input_image_th = input_image_th.cuda()


		class_images_th = []
		for class_image in self.class_images:
			h, w = get_image_size_after_resize_preserving_aspect_ratio(h=class_image.size[1],
																	w=class_image.size[0],
																	target_size=cfg.model.class_image_size)
			class_image = class_image.resize((w, h))

			class_image_th = transform_image(class_image)
			if cfg.is_cuda:
				class_image_th = class_image_th.cuda()

			class_images_th.append(class_image_th)

		with torch.no_grad():
			loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = self.net(images=input_image_th, class_images=class_images_th)



		# with torch.no_grad():
		#     feature_map = net.net_feature_maps(input_image_th)

		#     class_feature_maps = net.net_label_features(class_images_th)
		#     class_head = net.os2d_head_creator.create_os2d_head(class_feature_maps)

		#     loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(class_head=class_head,
		# 

		image_loc_scores_pyramid = [loc_prediction_batch[0]]
		image_class_scores_pyramid = [class_prediction_batch[0]]
		img_size_pyramid = [FeatureMapSize(img=input_image_th)]
		transform_corners_pyramid = [transform_corners_batch[0]]

		boxes = self.box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
												img_size_pyramid, self.class_ids,
												nms_iou_threshold=cfg.eval.nms_iou_threshold,
												nms_score_threshold=0.10,
												transform_corners_pyramid=transform_corners_pyramid)

		# remove some fields to lighten visualization                                       
		boxes.remove_field("default_boxes")

		# Note that the system outputs the correaltions that lie in the [-1, 1] segment as the detection scores (the higher the better the detection).
		scores = boxes.get_field("scores")

		figsize = (8, 8)
		fig=plt.figure(figsize=figsize)
		columns = len(self.class_images)
		for i, class_image in enumerate(self.class_images):
			fig.add_subplot(1, columns, i + 1)
			plt.imshow(class_image)
			plt.axis('off')


		plt.rcParams["figure.figsize"] = figsize

		cfg.visualization.eval.max_detections = 8
		cfg.visualization.eval.score_threshold = float("-inf")
		visualizer.show_detections(boxes, self.input_image,
								cfg.visualization.eval)

if __name__ == '__main__':
	diff_image = Os2d()
	diff_image.imager_loader(["/home/whoknows/Documents/FeatureMatching-Python/templates/choco_pie_top.png"],
					"/home/whoknows/Documents/FeatureMatching-Python/test/test_image1.png")
	diff_image.main()
