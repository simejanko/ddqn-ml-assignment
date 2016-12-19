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
    return image


class ObservationSequenceStore:
    """
    Class for storing sequence of observations.
    """
    def __init__(self, size):
        self.store = []
        self.size = size

    def add_observation(self, frame):
        self.store.append(frame)
        if len(self.store) > self.size+1:
            del self.store[0]

    def is_full(self):
        return self.size+1 == len(self.store)

    def get_first_seq(self):
        return np.array(self.store[:-1])

    def get_second_seq(self):
        return np.array(self.store[1:])


class ImageObservationStore(ObservationSequenceStore):
    """
    Clas for storing sequence of image observations. Includes preprocessing option.
    """
    def __init__(self, size, cut_u=0, cut_d=0, h=-1):
        """
        cut_u, cut_d and h are preprocessing parameters of preprocess_input function.
        """
        self.cut_u = cut_u
        self.cut_d = cut_d
        self.h = h
        super(ImageObservationStore, self).__init__(size)

    def add_observation(self, frame):
        super(ImageObservationStore, self).add_observation(preprocess_input(frame, self.cut_u, self.cut_d, self.h))