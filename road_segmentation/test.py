import numpy as np
from imgaug import augmenters as iaa
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
'''
path = '/home/felipesulser/Desktop/ex9/road_segmentation/test_set_images/test_1/test_1.png'
img = [mpimg.imread(path)]

seq = iaa.Sequential([
    
    iaa.Affine(
            rotate=(45, 45), # rotate by -45 to +45 degrees
        ), 
])
images_aug = seq.augment_images(img)
print(img)
print(images_aug)
#plt.imshow(img[0])
plt.imshow(img[0])
plt.show()
'''
'''
def max_out(inputs, num_units, axis=None):
        
        shape = inputs.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
        shape[axis] = num_units
        shape += [num_channels // num_units]
        outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
        return outputs

inputs = tf.constant([[[1,2,4,5],[3,4,3,9]],[[4,4,5,6],[1,0,5,4]]])
re = max_out(inputs,int(inputs.get_shape().as_list()[-1]/2))
sess = tf.Session()
print(inputs.get_shape().as_list())
print(re.get_shape().as_list())
print(sess.run(re))
'''
Mat_1 = np.matrix([[1,2],[3,4]])
Mat_2 = np.matrix([[1,2],[3,4]])
Mat_3 = np.matrix([[1,2],[3,4]])
Mat_4 = np.matrix([[1,2],[3,4]])
Mat_5 = np.matrix([[1,2],[3,4]])
Mat_6 = np.matrix([[1,2],[3,4]])
Mat_7 = np.matrix([[1,2],[3,4]])
Mat_8 = np.matrix([[1,2],[3,4]])
Mat_9 = np.matrix([[1,2],[3,4]])
print(str(int(0/625)))
print(str(int(-25/625)))
mats = np.vstack([np.hstack([Mat_1, Mat_2,Mat_3]), np.hstack([Mat_4, Mat_4,Mat_6]),np.hstack([Mat_7, Mat_8,Mat_9])])
print(mats)