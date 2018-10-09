import tensorflow as tf

class StaticCenterCrop(object):
    def __init__(self, crop_size):
        self.th, self.tw = crop_size
