import random
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from imageio import imread
from functools import partial
from itertools import groupby
from abc import abstractmethod, ABCMeta

from . import utils


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

def resize_flow(flow, resize_shape):
    if flow.ndim != 3:
        raise ValueError(f'Flow dimension should be 3, but found {flow.ndim} dimension')
    h, w = flow.shape[:2]
    th, tw = resize_shape # target size
    scale = np.array([tw/w, th/h]).reshape((1, 1, 2))
    flow = cv2.resize(flow, dsize = (tw, th))*scale
    flow = np.float32(flow)
    return flow


class BaseDataset(metaclass = ABCMeta):
    """ Abstract class to flexibly utilize tf.data pipeline """
    def __init__(self, dataset_dir, train_or_val, origin_size = None,
                 crop_type = 'random', crop_shape = None, resize_shape = None,
                 batch_size = 1, num_parallel_calls = 1):
        """ 
        Args:
        - dataset_dir str: target dataset directory
        - train_or_val str: flag indicates train or validation
        - origin_size tuple<int>: original size of target images
        - crop_type str: crop type, 'random', 'center', or None
        - crop_shape tuple<int>: crop shape
        - resize_shape tuple<int>: resize shape
        - batch_size int: batch size
        - num_parallel_calls int: # of parallel calls
        """
        self.dataset_dir = dataset_dir
        if not train_or_val in ['train', 'val']:
            raise ValueError('train_or_val is either train or val')
        self.train_or_val = train_or_val

        self.image_size = utils.get_size(origin_size, crop_shape, resize_shape)
        self.crop_type = crop_type
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape

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
        """ Native python function for read image and flow data """
        image_0_path, image_1_path, flow_path = filenames
        image_0 = imread(image_0_path.decode())
        image_1 = imread(image_1_path.decode())
        flow = self.load_flow(flow_path.decode())
        return image_0, image_1, flow

    def load_flow(self, flow_path):
        """ Function to read optical flow (normally .flo) data """
        return load_flow(flow_path)

    def preprocess(self, image_0, image_1, flow):
        """ Function to preprocess raw images and optical flow """
        if self.crop_shape is not None:
            image_0, image_1, flow = tf.py_func(self._crop_py, [image_0, image_1, flow],
                                                [tf.uint8, tf.uint8, tf.float32])
        if self.resize_shape is not None:
            image_0, image_1, flow = tf.py_func(self._resize_py, [image_0, image_1, flow],
                                                [tf.uint8, tf.uint8, tf.float32])

        images = tf.stack([image_0, image_1], axis = 0)
        images = tf.cast(images, tf.float32)

        return images, flow

    def _crop_py(self, image_0, image_1, flow):
        """ Native python function for cropping """
        image_size = image_0.shape[:2]
        if self.crop_type == 'random':
            cropper = utils.RandomCropper(image_size, self.crop_shape)
        elif self.crop_type == 'center':
            cropper = utils.CenterCropper(image_size, self.crop_shape)
        else:
            raise ValueError('invalid cropping argument has found')
        image_0, image_1, flow = map(cropper, [image_0, image_1, flow])
        return image_0, image_1, flow

    def _resize_py(self, image_0, image_1, flow):
        """ Native python function for resizing """
        image_0 = cv2.resize(image_0, dsize = tuple(self.resize_shape[::-1]))
        image_1 = cv2.resize(image_1, dsize = tuple(self.resize_shape[::-1]))
        flow = resize_flow(flow, self.resize_shape)
        return image_0, image_1, flow
            
    def _build(self):
        self._dataset = tf.data.Dataset.from_tensor_slices(self.samples)

        if self.train_or_val == 'train':
            self._dataset = self._dataset.shuffle(len(self.samples)).repeat()

        self._dataset = (self._dataset.map(self.parse, self.num_parallel_calls)
                         .map(self.preprocess, self.num_parallel_calls)
                         .batch(self.batch_size)
                         .prefetch(1))
        return 

    def get_element(self):
        """ Get data samples
        Returns:
        - images: tf.Tensor: minibatch of pairwise images
        - flows: tf.Tensor: minibatch of 
        
        if self.train_or_val == 'val' (i.e., validation mode), additionally returns
        - initializer: tf.data.Iterator.initializer: iterator initializer
        """
        if self.train_or_val == 'train':
            iterator = self._dataset.make_one_shot_iterator()
            images, flows = iterator.get_next()
            images.set_shape((self.batch_size, 2, *self.image_size, 3))
            flows.set_shape((self.batch_size, *self.image_size, 2))
            return images, flows
        else:
            iterator = self._dataset.make_initializable_iterator()
            images, flows = iterator.get_next()
            images.set_shape((self.batch_size, 2, *self.image_size, 3))
            flows.set_shape((self.batch_size, *self.image_size, 2))
            return images, flows, iterator.initializer
        

class FlyingChairs(BaseDataset):
    """ FlyingChairs dataset pipeline """
    def __init__(self, dataset_dir, train_or_val, origin_size = None,
                 crop_type = 'random', crop_shape = None, resize_shape = None,
                 batch_size = 1, num_parallel_calls = 1):
        super().__init__(dataset_dir, train_or_val, origin_size,
                         crop_type, crop_shape, resize_shape,
                         batch_size, num_parallel_calls)

    def has_no_txt(self):
        p = Path(self.dataset_dir) / 'data'
        imgs = sorted(p.glob('*.ppm'))
        samples = [(str(i[0]), str(i[1]), str(i[0]).replace('img1', 'flow').replace('.ppm', '.flo'))\
                   for i in zip(imgs[::2], imgs[1::2])]
        self.split(samples)

class FlyingThings3D(BaseDataset):
    def __init__(self, dataset_dir, train_or_val, origin_size = None,
                 crop_type = 'random', crop_shape = None, resize_shape = None,
                 batch_size = 1, num_parallel_calls = 1):
        super().__init__(dataset_dir, train_or_val, origin_size,
                         crop_type, crop_shape, resize_shape,
                         batch_size, num_parallel_calls)

    def has_no_txt(self):
        # TODO
        pass


class Sintel(BaseDataset):
    """ MPI-Sintel-complete dataset pipeline """
    def __init__(self, dataset_dir, train_or_val, mode = 'clean', origin_size = None,
                 crop_type = 'random', crop_shape = None, resize_shape = None,
                 batch_size = 1, num_parallel_calls = 1):
        self.mode = mode
        super().__init__(dataset_dir, train_or_val, origin_size,
                         crop_type, crop_shape, resize_shape,
                         batch_size, num_parallel_calls)

    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_img = p / 'training' / self.mode
        p_flow = p / 'training/flow'
        
        collections_of_scenes = sorted(map(str, p_img.glob('**/*.png')))
        collections = [list(g) for k, g in groupby(collections_of_scenes, lambda x: x.split('/')[-2])]
        samples = [(*i, i[0].replace(self.mode, 'flow').replace('.png', '.flo'))\
                    for collection in collections for i in utils.window(collection, 2)]
        self.split(samples)


class SintelClean(Sintel):
    """ MPI-Sintel-complete dataset (clean path) pipeline """
    def __init__(self, dataset_dir, train_or_val, origin_size = None,
                 crop_type = 'random', crop_shape = None, resize_shape = None,
                 batch_size = 1, num_parallel_calls = 1):
        super().__init__(dataset_dir, train_or_val, 'clean', origin_size,
                         crop_type, crop_shape, resize_shape,
                         batch_size, num_parallel_calls)

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val+'.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                img_0_path, img_1_path, flow_path = i.split(',')
                img_0_path, img_1_path = map(lambda p: p.replace('final', 'clean'),
                                             (img_0_path, img_1_path))
                flow_path = flow_path.strip()
                self.samples.append((img_0_path, img_1_path, flow_path))

class SintelFinal(Sintel):
    """ MPI-Sintel-complete dataset (final path) pipeline """
    def __init__(self, dataset_dir, train_or_val, origin_size = None,
                 crop_type = 'random', crop_shape = None, resize_shape = None,
                 batch_size = 1, num_parallel_calls = 1):
        super().__init__(dataset_dir, train_or_val, 'final', origin_size,
                         crop_type, crop_shape, resize_shape,
                         batch_size, num_parallel_calls)

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val+'.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                img_0_path, img_1_path, flow_path = i.split(',')
                img_0_path, img_1_path = map(lambda p: p.replace('clean', 'final'),
                                             (img_0_path, img_1_path))
                flow_path = flow_path.strip()
                self.samples.append((img_0_path, img_1_path, flow_path))


def get_dataset(dataset_name):
    """
    Get specified dataset 
    Args: dataset_name str: target dataset name
    Returns: dataset tf.data.Dataset: target dataset class

    Available dataset pipelines
    - FlyingChairs
    - Sintel, SintelClean, SintelFinal
    """
    datasets = {"FlyingChairs":FlyingChairs,
                "Sintel":Sintel,
                "SintelClean":SintelClean,
                "SintelFinal":SintelFinal}
    return datasets[dataset_name]
