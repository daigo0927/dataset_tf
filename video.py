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


class BaseDataset(metaclass = ABCMeta):
    """ Abstract class to flexibly utilize tf.data pipeline """
    def __init__(self, dataset_dir, train_or_val,
                 strides = 3, stretchable = False, origin_size = None,
                 crop_type = 'random', crop_shape = None, resize_shape = None,
                 batch_size = 1, num_parallel_calls = 1):
        """ 
        Args:
        - dataset_dir str: target dataset directory
        - train_or_val str: flag indicates train or validation
        - strides int: target temporal range of triplet images
        - stretchable bool: enabling shift of start and end index of triplet images
        - origin_size tuple<int>: original size of target images
        - crop_type str: crop type either of [random, center, None]
        - crop_shape tuple<int>: crop shape
        - resize_shape tuple<int>: resize shape
        - batch_size int: batch size
        - num_parallel_calls int: number of parallel process
        """
        self.dataset_dir = dataset_dir
        if not train_or_val in ['train', 'val']:
            raise ValueError('train_or_val is either train or val')
        self.train_or_val = train_or_val

        self.strides = strides
        self.stretchable = stretchable

        self.image_size = utils.get_size(origin_size, crop_shape, resize_shape)
        self.crop_type = crop_type
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape

        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls

        p = Path(dataset_dir) / (train_or_val+'_{self.strides}frames.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()
        self._build()

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val + f'_{self.strides}frames.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                self.samples.append(i.strip().split(','))

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
        return tf.py_func(self._read_py, [filenames], [tf.uint8, tf.uint8, tf.uint8, tf.float32])

    def _read_py(self, filenames):
        """ Native python function for read image and flow data """
        if self.stretchable:
            f, f_end = sorted(np.random.choice(range(1, self.strides), 2, replace = False))
        else:
            f_end = self.strides-1
            f = np.random.randint(1, f_end)

        t = np.array(f/f_end, dtype = np.float32)
        
        image_0_path, image_t_path, image_1_path = filenames[0], filenames[f], filenames[f_end]
        image_0, image_t, image_1 = map(lambda x: imread(x.decode()),
                                        [image_0_path, image_t_path, image_1_path])
        return image_0, image_t, image_1, t

    def preprocess(self, image_0, image_t, image_1, t):
        """ Function to preprocess raw images and optical flow """
        if self.crop_shape is not None:
            image_0, image_t, image_1 = tf.py_func(self._crop_py, [image_0, image_t, image_1],
                                                   [tf.uint8, tf.uint8, tf.uint8])

        if self.resize_shape is not None:
            image_0, image_t, image_1 = tf.py_func(self._resize_py, [image_0, image_t, image_1],
                                                   [tf.uint8, tf.uint8, tf.uint8])

        images = tf.stack([image_0, image_t, image_1], axis = 0)
        images = tf.cast(images, tf.float32)
        images = images/255.

        return images, t

    def _crop_py(self, image_0, image_t, image_1):
        """ Native python function for cropping """
        image_size = image_0.shape[0:2]
        if self.crop_type == 'random':
            cropper = utils.RandomCropper(image_size, self.crop_shape)
        elif self.crop_type == 'center':
            cropper = utils.CenterCropper(image_size, self.crop_shape)
        else:
            raise ValueError('invalid cropping argument has found')
        image_0, image_t, image_1 = map(cropper, [image_0, image_t, image_1])
        return image_0, image_t, image_1

    def _resize_py(self, image_0, image_t, image_1):
        """ Native python function for resizing """
        resizer = partial(cv2.resize, dsize = tuple(self.resize_shape[::-1]))
        image_0, image_t, image_1 = map(resizer, [image_0, image_t, image_1])
        return image_0, image_t, image_1
            
    def _build(self):
        self._dataset = tf.data.Dataset.from_tensor_slices(self.samples)

        if self.train_or_val == 'train':
            self._dataset = self._dataset.shuffle(len(self.samples)).repeat()

        self._dataset = (self._dataset.map(self.parse, self.num_parallel_calls)
                         .map(self.preprocess, self.num_parallel_calls)
                         .batch(self.batch_size)
                         .prefetch(1))
        return

    def get_element(self)
        """ Get data samples
        Returns:
        - images: tf.Tensor: minibatch of pairwise images
        - t: tf.Tensor: positions of middle frame between start and end frames
        
        if self.train_or_val == 'val' (i.e., validation mode), additionally returns
        - initializer: tf.data.Iterator.initializer: iterator initializer
        """
        if self.train_or_val == 'train':
            iterator = self._dataset.make_one_shot_iterator
            images, t = iterator.get_next()
            images.set_shape((self.batch_size, 3, *self.image_size, 3))
            return images, t
        else:
            iterator = self._dataset.make_initializable_iterator()
            images, t = iterator.get_next()
            images.set_shape((self.batch_size, 3, *self.image_size, 3))
            return images, t, iterator.initializer
    


class DAVIS(BaseDataset):
    """ DAVIS dataset pipeline """
    def __init__(self, dataset_dir, train_or_val, resolution = '480p',
                 strides = 3, stretchable = False, origin_size = None,
                 crop_type = 'random', crop_shape = None, resize_shape = None,
                 batch_size = 1, num_parallel_calls = 1):
        """ 
        Args:
        - dataset_dir str: target dataset directory
        - train_or_val str: flag indicates train or validation
        - resolution str: either 480p or Full-Resolution for target resolution
        - strides int: target temporal range of triplet images
        - stretchable bool: enabling shift of start and end index of triplet images
        - origin_size tuple<int>: original size of target images
        - crop_type str: crop type either of [random, center, None]
        - crop_shape tuple<int>: crop shape of target images
        - resize_shape tuple<int>: resize shape
        - batch_size int: batch size
        - num_parallel_calls int: number of parallel process
        """
        if not resolution in ['480p', 'Full-Resolution']:
            raise ValueError('Invalid argument for target resolution')
        self.resolution = resolution
        super().__init__(dataset_dir, train_or_val, strides, stretchable,
                         origin_size, crop_type, crop_shape, resize_shape,
                         batch_size, num_parallel_calls)

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val + f'_{self.strides}frames.txt')
        res_other = 'Full-Resolution' if self.resolution == '480p' else '480p'
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                self.samples.append(i.replace(res_other, self.resolution).strip().split(','))

    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_set = p / 'ImageSets/2017' / (self.train_or_val+'.txt')
        p_img = p / 'JPEGImages' / self.resolution

        self.samples = []
        with open(p_set, 'r') as f:
            for i in f.readlines():
                p_img_categ = p_img / i.strip()
                collection = sorted(map(str, p_img_categ.glob('*.jpg')))
                self.samples += [i for i in utils.window(collection, self.strides)]

        with open(p / (self.train_or_val+f'_{self.strides}frames.txt'), 'w') as f:
            f.writelines((','.join(i) + '\n' for i in self.samples))
        

class Sintel(BaseDataset):
    """ MPI-Sintel-complete dataset pipeline """
    def __init__(self, dataset_dir, train_or_val, mode = 'clean',
                 strides = 3, stretchable = False, origin_size = None,
                 crop_type = 'random', crop_shape = None, resize_shape = None,
                 batch_size = 1, num_parallel_calls = 1):
        """ 
        Args:
        - dataset_dir str: target dataset directory
        - train_or_val str: flag indicates train or validation
        - mode str: either clean or final to specify data path
        - strides int: target temporal range of triplet images
        - stretchable bool: enabling shift of start and end index of triplet images
        - origin_size tuple<int>: original size of target images
        - crop_type str: crop type either of [random, center, None]
        - crop_shape tuple<int>: crop shape of target images
        - resize_shape tuple<int>: resize shape
        - batch_size int: batch size
        - num_parallel_calls int: number of parallel process
        """
        self.mode = mode
        super().__init__(dataset_dir, train_or_val, strides, stretchable,
                         origin_size, crop_type, crop_shape, resize_shape,
                         batch_size, num_parallel_calls)

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
    """ MPI-Sintel-complete dataset (clean path) pipeline """
    def __init__(self, dataset_dir, train_or_val,
                 strides, stretchable, origin_size,
                 crop_type, crop_shape, resize_shape,
                 batch_size, num_parallel_calls):
        super().__init__(dataset_dir, train_or_val, 'clean',
                         strides, stretchable, origin_size,
                         crop_type, crop_shape, resize_shape,
                         batch_size, num_parallel_calls)

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val+f'_{self.strides}.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                self.samples.append(i.replace('final', 'clean').strip().split(','))
                
class SintelFinal(Sintel):
    """ MPI-Sintel-complete dataset (final path) pipeline """
    def __init__(self, dataset_dir, train_or_val,
                 strides, stretchable, origin_size,
                 crop_type, crop_shape, resize_shape,
                 batch_size, num_parallel_calls):
        super().__init__(dataset_dir, train_or_val, 'final',
                         strides, stretchable, origin_size,
                         crop_type, crop_shape, resize_shape,
                         batch_size, num_parallel_calls)

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val+f'_{self.strides}.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                self.samples.append(i.replace('clean', 'final').strip().split(','))

                
def get_dataset(dataset_name):
    """
    Get specified dataset 
    Args: dataset_name str: target dataset name
    Returns: dataset tf.data.Dataset: target dataset class

    Available dataset pipelines
    - DAVIS
    - Sintel, SintelClean, SintelFinal
    """
    datasets = {"DAVIS":DAVIS,
                "Sintel":Sintel,
                "SintelClean":SintelClean,
                "SintelFinal":SintelFinal}
    return datasets[dataset_name]
