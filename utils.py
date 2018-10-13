import random
from itertools import islice


class RandomCropper(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, image):
        return image[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw)]
 

class CenterCropper(object):
    def __init__(self, image_sample, crop_size):
        self.h, self.w = image_size
        self.th, self.tw = crop_size

    def __call__(self, image):
        return image[(self.h-self.th)//2:(self.h+self.th)//2,
                     (self.w-self.tw)//2:(self.w+self.tw)//2]

def window(seq, n = 2):
    """ Returns a sliding window over data from the iterable """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for el in it:
        result = result[1:] + (el,)
        yield result

def to_rgb(image):
    if image.ndim == 2:
        return np.stack([image]*3, axis = -1)
    else:
        return image
