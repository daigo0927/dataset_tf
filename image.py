import random
import numpy as np
import tensorflow as tf
import cv2
import warnings
from pathlib import Path
from imageio import imread
from functools import partial
from itertools import groupby
from abc import abstractmethod, ABCMeta

from . import utils


class BaseDataset(metaclass = ABCMeta):
    """ Abstract class to flexibly utilize tf.data pipeline """
    def __init__(self, dataset_dir, train_or_val, origin_size = None,
                 crop_type = 'random', crop_shape = None, resize_shape = None,
                 use_label = True, one_hot = True,
                 batch_size = 1, num_parallel_calls = 1):
        """ 
        Args:
        - dataset_dir str: target dataset directory
        - train_or_val str: flag indicates train or validation
        - origin_size tuple<int>: original size of target images
        - crop_type str: crop type either of [random, center, None]
        - crop_shape tuple<int>: crop shape
        - resize_shape tuple<int>: resize shape
        - use_label bool: if use label or not
        - one_hot bool: if encode label one-hot or not
        - batch_size int: batch size
        - num_parallel_calls int: number of parallel process
        """
        self.dataset_dir = dataset_dir
        if not train_or_val in ['train', 'val']:
            raise ValueError('train_or_val is either train or val')
        self.train_or_val = train_or_val

        self.image_size = utils.get_size(origin_size, crop_shape, resize_shape)
        self.crop_type = crop_type
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape

        self.use_label = use_label
        self.one_hot = one_hot

        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls

        self.get_classes()
        p = Path(dataset_dir) / (train_or_val+'.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()

        self._build()

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val + '.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                img_path, label = i.split(',')
                label = label.strip()
                self.samples.append(img_path, label)

    @abstractmethod
    def has_no_txt(self):
        pass

    def split(self, samples):
        p = Path(self.dataset_dir)
        val_ratio = 0.1
        random.shuffle(samples)
        idx = int(len(samples) * (1 - val_ratio))
        train_samples = samples[:idx]
        val_samples = samples[idx:]

        with open(p / 'train.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in train_samples))
        with open(p / 'val.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in val_samples))

        self.samples = train_samples if self.train_or_val == 'train' else val_samples

    def parse(self, file_and_label):
        """
        Tensorflow file parser 
        Args:
        - img_path tf.string: image path
        - label tf.uint8: label
        Returns:
        - tf.Tensor<tf.uint8> image_0, image_1: target 0,1-th image
        - tf.Tensor<tf.uint8> flow: target optical flow
        """
        # image, label = tf.py_func(lambda f: self._read_py(f), [file_and_label],
        #                           [tf.uint8, tf.int32])
        img_path, label = file_and_label[0], file_and_label[1]
        # pdb.set_trace()
        image = tf.py_func(lambda p: imread(p.decode()), [img_path], tf.uint8)
        label = tf.string_to_number(label, tf.int32)
        return image, label

    def _read_py(self, file_and_label):
        """ Native python function for read image and label """
        img_path, label = file_and_label
        image = imread(img_path.decode())
        label = int(label.decode())
        return image, label

    def preprocess(self, image, label):
        """ Function to preprocess raw image """
        # Force all images to be RGB
        image = tf.py_func(utils.to_rgb, [image], tf.uint8)

        if self.crop_shape is not None:
            image = tf.py_func(self._crop_py, [image], tf.uint8)

        if self.resize_shape is not None:
            image = tf.py_func(lambda x: cv2.resize(x, dsize = tuple(self.resize_shape[::-1])),
                               [image], tf.uint8)

        if self.one_hot:
            label = tf.one_hot(label, self.num_classes)

        image = tf.cast(image, tf.float32)

        return images, labels

    def _crop_py(self, image):
        """ Native python function for cropping """
        image_size = image.shape[0:2]
        if self.crop_type == 'random':
            cropper = utils.RandomCropper(image_size, self.crop_shape)
        elif self.crop_type == 'center':
            cropper = utils.CenterCropper(image_size, self.crop_shape)
        else:
            raise ValueError('invalid cropping argument has found')
        image = cropper(image)
        return image

    @abstractmethod
    def get_classes(self):
        pass 
            
    def _build(self):
        self._dataset = tf.data.Dataset.from_tensor_slices(self.samples)

        if self.train_or_val == 'train'
            self._dataset = self._dataset.shuffle(len(self.samples)).repeat()

        self._dataset = (self._dataset.map(self.parse, self.num_parallel_calls)
                         .map(self.preprocess, self.num_parallel_calls)
                         .batch_size(self.batch_size)
                         .prefetch(1))
        return

    def get_element(self):
        """ Get data samples
        Returns:
        - images: tf.Tensor: minibatch of images
        - (if required) labels: tf.Tensor: minibatch of labels
        
        if self.train_or_val == 'val' (i.e., validation mode), additionally returns
        - initializer: tf.data.Iterator.initializer: iterator initializer
        """
        if self.train_or_val == 'train':
            iterator = self._dataset.make_one_shot_iterator()
            images, labels = iterator.get_next()
            images.set_shape((self.batch_size, *self.image_size, 3))
            if self.use_label:
                return images, labels
            else:
                return images
        else:
            iterator = self._dataset.make_initializable_iterator()
            images, labels = iterator.get_next()
            images.set_shape((self.batch_size, *self.image_size, 3))
            if self.use_label:
                return images, labels, iterator.initializer
            else:
                return images, iterator.initializer


class Food101(BaseDataset):
    """ Food-101 dataset pipeline """
    def __init__(self, dataset_dir, train_or_val, origin_size = None,
                 crop_type = 'random', crop_shape = None, resize_shape = None,
                 use_label = True, one_hot = True,
                 batch_size = 1, num_parallel_calls = 1):
        """ 
        Args:
        - dataset_dir str: target dataset directory
        - train_or_val str: flag indicates train or validation
        - origin_size tuple<int>: original size of target images
        - crop_type str: crop type either of [random, center, None]
        - crop_shape tuple<int>: crop shape
        - resize_shape tuple<int>: resize shape
        - use_label bool: if use label or not
        - one_hot bool: if encode label one-hot or not
        - shuffle bool: if shuffle or not
        - batch_size int: batch size
        - num_parallel_calls int: number of parallel process
        """
        super().__init__(dataset_dir, train_or_val, origin_size
                         crop_type, crop_shape, resize_shape,
                         use_label, one_hot, batch_size, num_parallel_calls)
        warnings.filterwarnings('ignore', category = UserWarning)

    def has_no_txt(self):
        p = Path(self.dataset_dir)
        train_or_test = 'train' if self.train_or_val == 'train' else 'test'
        p_set = p / 'meta' / (train_or_test+'.txt')
        p_img = p / 'images'
        self.samples = []

        with open(p_set, 'r') as f:
            for i in f.readlines():
                i = i.strip()
                class_ = i.split('/')[0]
                sample = (str(p_img/(i+'.jpg')), str(self.classes.index(class_)))
                self.samples.append(sample)

        with open(p/(self.train_or_val+'.txt'), 'w') as f:
            f.writelines((','.join(i) + '\n' for i in self.samples))

    def get_classes(self):
        p = Path(self.dataset_dir)
        p_class = p / 'meta/classes.txt'
        with open(p_class, 'r') as f:
            self.classes = f.read().split('\n')[:-1]
        self.num_classes = len(self.classes)
            

def get_dataset(dataset_name):
    """
    Get specified dataset 
    Args: dataset_name str: target dataset name
    Returns: dataset tf.data.Dataset: target dataset class

    Available dataset pipelines
    - Food101
    """
    datasets = {"Food101":Food101}
    return datasets[dataset_name]
