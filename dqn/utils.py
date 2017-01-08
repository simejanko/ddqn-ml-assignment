import matplotlib.pyplot as plt
from scipy.misc import imresize
import numpy as np

def rgb2gray(rgb):
    """
    Converts rgb image to grayscale using standard formula.
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def preprocess_input(image , cut_u=0, cut_d=0, h=-1):
    """
    Preprocesses multiple images by converting them to grayscale,
    cutting their top/bottom height, rescaling them and putting them in array,
    ready for use in keras model.
    """
    #output = []
    #for image in images:
    if cut_d>0:
        image = rgb2gray(image[cut_u:-cut_d, :, :]).astype(int)
    else:
        image = rgb2gray(image[cut_u:, :, :]).astype(int)
    aspect = image.shape[1]/image.shape[0]
    if h != -1:
        #TODO: naslednji run nearest
        image = imresize(image, (h, int(h*aspect)), interp='bilinear')
    #    output.append(image)
    #print(np.array_equal(output[0],output[1]))
    #return np.array(output)
    #plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    #plt.show()
    return image.astype(np.uint8)