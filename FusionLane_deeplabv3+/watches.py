from PIL import Image
from PIL import ImageEnhance
from pylab import *
pic_dir='./datasets/pascal_voc_seg/exp/train_on_train_set/vis/segmentation_results/000001_prediction.png'

img = Image.open(pic_dir)
arr = array(img)
print(arr)
enh_con = ImageEnhance.Contrast(img)
contrast = 3
img_contrasted = enh_con.enhance(contrast)

img_contrasted.show()