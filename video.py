import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from imageio import imread
from functools import partial
from itertools import groupby
from abc import abstractmethod, ABCMeta

import pdb

import utils


class BaseDataset(metaclass = ABCMeta):
    """ Wrapper class to flexibly utilize tf.data pipeline """
    def __init__(self, dataset_dir, train_or_val,
                 strides = 3, stretchable = False,
                 crop_type = 'random', crop_shape = None,
                 shuffle = False, batch_size = 1, num_parallel_calls = 0):
        """ 
        Args:
        - str dataset_dir: target dataset directory
        - str train_or_val: flag indicates train or validation
        - int strides: target temporal range of triplet images
        - bool stretchable: enabling shift of start and end index of triplet images
        - str crop_type: crop type either of [random, center, None]
        - int batch_size: batch size
        - int num_parallel_calls: number of parallel process
        """
        self.dataset_dir = dataset_dir
        if not train_or_val in ['train', 'val']:
            raise ValueError('train_or_val is either train or val')
        self.train_or_val = train_or_val

        self.strides = strides
        self.stretchable = stretchable

        self.crop_type = crop_type
        self.crop_shape = crop_shape

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls

        p = Path(dataset_dir) / (train_or_val+'_{self.strides}frames.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()
        self._build()

    def has_txt(self):
        # p = Path(self.dataset_dir) / (self.train_or_val + f'_{self.strides}frames.txt')
        # self.samples = []
        # with open(p, 'r') as f:
        #     for i in f.readlines():
        #         self.samples.append(i.strip().split(','))
        # TODO: adapt different type of dataset (e.g. resolution, data quality, ...)

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

        with open(p / f'train_{self.strides}frames.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in train_samples))
        with open(p / f'val_{self.strides}frames.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in val_samples))

        self.samples = train_samples if self.train_or_val == 'train' else val_samples

    def parse(self, filenames):
        """
        Tensorflow file parser using native python function
        Args: tf.Tensor<tf.string> filenames: indicates target images and flow files
        Returns:
        - tf.Tensor<tf.uint8> image_0, image_1: target 0,1-th image
        - tf.Tensor<tf.float32> flow: target optical flow
        """
        return tf.py_func(self._read_py, [filenames], [tf.uint8, tf.float32])

    def _read_py(self, filenames):
        """ Native python function for read image and flow data """
        if self.stretchable:
            f, f_end = sorted(random.samples(range(1, self.strides), 2))
        else:
            f = random.randint(1, self.strides-2)
            f_end = self.strides-1
        t = f/f_end
        
        image_0_path, image_t_path, image_1_path = filenames[0], filenames[f], filenames[f_end]
        image_0, image_t, image_1 = map(lambda x: imread(x.decode()),
                                        [image_0_path, image_t_path, image_1_path])
        return np.stack([image_0, image_t, image_1], axis = 0), t

    def preprocess(self, images, t):
        """ Function to preprocess raw images and optical flow """
        if self.crop_shape is not None:
            images = tf.py_func(self._crop_py, [images], tf.uint8)

        images = tf.cast(images, tf.float32)
        images = images/255.

        return images, t

    def _crop_py(self, images):
        """ Native python function for cropping """
        image_size = images.shape[1:3]
        if self.crop_type == 'random':
            cropper = utils.RandomCropper(image_size, self.crop_shape)
        elif self.crop_type == 'center':
            cropper = utils.CenterCropper(image_size, self.crop_shape)
        else:
            raise ValueError('invalid cropping argument has found')
        return cropper(images)
            
    def _build(self):
        self._dataset = tf.data.Dataset.from_tensor_slices(self.samples)
        if self.shuffle:
            self._dataset = self._dataset.shuffle(len(self.samples))

        self._dataset = (self._dataset.map(self.parse, self.num_parallel_calls)
                        .map(self.preprocess, self.num_parallel_calls)
                        .batch(self.batch_size)
                        .repeat()
                        .prefetch(1))
        return

    def make_one_shot_iterator(self):
        return self._dataset.make_one_shot_iterator()


class DAVIS(BaseDataset):
    def __init__(self, dataset_dir, train_or_val, resolution = '480p',
                 strides = 3, stretchable = False,
                 crop_type = 'random', crop_shape = None,
                 shuffle = False, batch_size = 1, num_parallel_calls = 0):
        if not resolution in ['480p', 'Full-Resolution']:
            raise ValueError('Invalid argument for target resolution')
        self.resolution = resolution
        super().__init__(dataset_dir, train_or_val, strides, stretchable,
                         crop_type, crop_shape, shuffle, batch_size, num_parallel_calls)

    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_set = p / 'ImageSets/2017' / (self.train_or_val+'.txt')
        p_img = p / 'JPEGImage' / self.resolution

        self.samples = []
        with open(p_set, 'r') as f:
            for i in f.readlines():
                p_img_categ = p_img / i.strip()
                collection = sorted(map(str, p_img_categ.glob('*.jpg')))
                self.samples += [i for i in window(collection, self.strides)]

        

class Sintel(BaseDataset):
    def __init__(self, dataset_dir, train_or_val, mode = 'clean',
                 crop_type = 'random', crop_shape = None,
                 shuffle = False, batch_size = 1, num_parallel_calls = 0):
        self.mode = mode
        super().__init__(dataset_dir, train_or_val, crop_type, crop_shape,
                         shuffle, batch_size, num_parallel_calls)

    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_img = p / 'training' / self.mode
        p_flow = p / 'training/flow'
        
        collections_of_scenes = sorted(map(str, p_img.glob('**/*.png')))
        collections = [list(g) for k, g in groupby(collections_of_scenes, lambda x: x.split('/'[-2]))]
        samples = [(*i, i[0].replace(self.mode, 'flow').replace('.png', '.flo'))\
                    for collection in collections for i in utils.window(collection, 2)]
        self.split(samples)


class SintelClean(Sintel):
    def __init__(self, dataset_dir, train_or_val, crop_type, crop_shape,
                 shuffle, batch_size, num_parallel_calls):
        super().__init__(dataset_dir, train_or_val, 'clean', crop_type, crop_shape,
                         shuffle, batch_size, num_parallel_calls)

class SintelFinal(Sintel):
    def __init__(self, dataset_dir, train_or_val, crop_type, crop_shape,
                 shuffle, batch_size, num_parallel_calls):
        super().__init__(dataset_dir, train_or_val, 'final', crop_type, crop_shape,
                         shuffle, batch_size, num_parallel_calls)


