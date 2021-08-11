#from ImageSim.ResNet50_similarity_Xing import neural
import os
from OS2D.neural import Os2d
#diff_image = neural()
#imgs = diff_image.load_from_dir()
#diff_image.main(imgs)

path = os.path.dirname(os.path.realpath(__file__))
template_location = os.path.join(path,'templates/')
test_location = os.path.join(path,'test/')
#print(template_location)
#print(test_location)
def load_images_from_folder(folder):
		for filename in os.listdir(folder):
			img = (os.path.join(folder,filename))
			yield(img)
				#print(type(img)




template_imgs = list(load_images_from_folder(template_location))
test_imgs = list(load_images_from_folder(test_location))
print(test_imgs[0])
print(template_imgs[0])
obj = Os2d()
obj.imager_loader([template_imgs[0]], test_imgs[0])          
obj.main(1500)
