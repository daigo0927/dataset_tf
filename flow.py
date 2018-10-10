import numpy as np
import tensorflow as tf
from pathlib import Path
from imageio import imread
from functools import partial
from abc import abstractmethod, ABCMeta

import pdb

import utils

def load_flow(uri):
    """
    Function to load optical flow data
    Args: str uri: target flow path
    Returns: np.ndarray: extracted optical flow
    """
    with open(uri, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
        return None


def _parse(filenames):
# def _parse(image_0_path, image_1_path, flow_path):
    """
    Parser for optical flow datasets
    Args:
    - tuple<str> filenames: tuple containing image and flow paths (image_0, image_1, flow)
    Returns:
    - tf.Tensor images: paired continuous images
    - tf.Tensor flow: ground truth optical flow between paired images
    """
    return tf.py_func(_read_py_func, [filenames], [tf.uint8, tf.uint8, tf.float32])

def _read_py_func(filenames):
    """ decode binary string to UTF-8 """
    image_0_path, image_1_path, flow_path = filenames
    image_0 = imread(image_0_path.decode())
    image_1 = imread(image_1_path.decode())
    flow = load_flow(flow_path.decode())
    return image_0, image_1, flow


def _preprocess(image_0, image_1, flow,
                crop_type = 'random', crop_shape = None):
    """
    Preprocess the raw data
    Args:
    - tf.Tensor images: raw images
    - tf.Tensor flow: raw flow
    - str crop_type: crop type for raw images and flow ['random', 'center', None]
    - tuple<int> crop_shape: shape of crop
    Returns:
    - tf.Tensor images: processed images
    - tf.Tensor flow: processed flow
    """

    if crop_type is not None:
        image_0 = tf.py_func(_crop_py_func, [image_0, crop_type, crop_shape], tf.uint8)
        image_1 = tf.py_func(_crop_py_func, [image_1, crop_type, crop_shape], tf.uint8)
        flow = tf.py_func(_crop_py_func, [flow, crop_type, crop_shape], tf.float32)

    images = tf.stack([image_0, image_1], axis = 0)
    images = tf.cast(images, tf.float32)
    images = images/255.

    return images, flow

def _crop_py_func(x, crop_type, crop_shape):
    cropper = utils.RandomCropper(x.shape[:2], crop_shape) if crop_type == 'random'\
      else utils.CenterCropper(x.shape[:2], crop_shape)
    x = cropper(x)
    return x


class BaseDataset(metaclass = ABCMeta):
    """ Wrapper class of flexibly utilize tf.data pipeline """
    def __init__(self, dataset_dir, train_or_val,
                 crop_type = 'random', crop_shape = None,
                 shuffle = False, batch_size = 1, num_parallel_calls = 0):
        """ 
        Args:
        - str dataset_dir: target dataset directory
        - str train_or_val: flag indicates train or validation
        - str crop_type: 
        """
        self.dataset_dir = dataset_dir
        if not train_or_val in ['train', 'val']:
            raise ValueError('train_or_val is either train or val')
        self.train_or_val = train_or_val
        
        if not crop_type in ['random', 'center', None]:
            raise ValueError('crop_type should be in random/center/None')
        if crop_type == 'random':
            self.cropper = partial(utils.RandomCropper, crop_shape = crop_shape)
        elif crop_type == 'center':
            self.cropper = partial(utils.CenterCropper, crop_shape = crop_shape)
        else:
            self.cropper = None
        self.crop_type = crop_type
        self.crop_shape = crop_shape

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls

        p = Path(dataset_dir) / (train_or_val+'.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()
        self._build()

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val+'.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                img_0_path, img_1_path, flow_path = i.split(',')
                flow_path = flow_path.strip()
                self.samples.append((img_0_path, img_1_path, flow_path))

    @abstractmethod
    def has_no_txt(self):
        pass

    def parse(self, filenames):
        """
        Tensorflow file parser using native python function
        Args: tf.Tensor<tf.string> filenames: indicates target images and flow files
        Returns:
        - tf.Tensor<tf.uint8> image_0, image_1: target 0,1-th image
        - tf.Tensor<tf.float32> flow: target optical flow
        """
        return tf.py_func(self._read_py, [filenames], [tf.uint8, tf.uint8, tf.float32])

    def _read_py(self, filenames):
        """ python function for read image and flow data """
        img_0_path, img_1_path, flow_path = filenames
        image_0 = imread(image_0_path.decode())
        image_1 = imread(image_1_path.decode())
        flow = self.load_flow(flow_path.decode())
        return image_0, image_1, flow

    def load_flow(self, flow_path):
        """ Function to read optical flow (normally .flo) data """
        return load_flow(flow_path)

    def preprocess(self, image_0, image_1, flow):
        """ Function to preprocess raw images and optical flow """
        if self.cropper is not None:
            image_0 = tf.py_func(self.cropper, [image_0], tf.uint8)
            image_1 = tf.py_func(self.cropper, [image_1], tf.uint8)
            flow = tf.py_func(self.cropper, [flow], tf.float32)

        images = tf.stack([image_0, image_1], axis = 0)
        images = tf.cast(images, tf.float32)
        images = images/255.

        return images, flow

    def _build(self):
        self.dataset = tf.data.Dataset.from_tensor_slices(self.samples)
        if self.shuffle:
            self.dataset = self.dataset.shuffle(len(self.samples))

        self.dataset = (self.dataset.map(self.parse, self.num_parallel_calls)
                        .map(self.preprocess, self.num_parallel_calls)
                        .batch(self.batch_size)
                        .repeat()
                        .prefetch(1))
        return

    def make_one_shot_iterator(self):
        return self.dataset.make_one_shot_iterator()


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess = tf.Session()

    samples = []
    with open('/Users/Daigo/Data/MPI-Sintel-complete/val.txt', 'r') as f:
        for i  in f.readlines():
            image_0_path, image_1_path, flow_path = i.split(',')
            flow_path = flow_path.strip()
            samples.append((image_0_path, image_1_path, flow_path))

    dataset = (tf.data.Dataset.from_tensor_slices(samples)
               .shuffle(len(samples))
               .map(_parse)
               .map(partial(_preprocess, crop_type = 'random', crop_shape = (384, 448)))
               .batch(4)
               .prefetch(1))

    iterator = dataset.make_one_shot_iterator()
    next_el = iterator.get_next()
    images, flow = sess.run(next_el)
    print(f'image shape {images.shape}, flow shape {flow.shape}')
