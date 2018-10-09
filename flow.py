import numpy as np
import cv2
import tensorflow as tf
from imageio import imread
from functools import partial

import pdb

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
    image_0_path, image_1_path, flow_path = filenames
    # image_0, image_1 = map(imread, (image_0_path, image_1_path))
    image_0, image_1 = tf.py_func(imread, [image_0_path, image_1_path], tf.float32)
    flow = tf.py_func(load_flow, [flow_path], tf.float32)
    # flow = load_flow(flow_path)

    # convert = partial(tf.convert_to_tensor, dtype = tf.float32)
    # image_0, image_1, flow = map(convert, (image_0, image_1, flow))
    # image_0.set_shape((384, 448, 3))
    # image_1.set_shape((384, 448, 3))
    # flow.set_shape((384, 448, 3))

    return image_0, image_1, flow

def _read_py(filenames):
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
    # image_0.set_shape([None, None, None])
    # image_1.set_shape([None, None, None])
    # flow.set_shape([None, None, None])

    if crop_shape is not None:
        th, tw = crop_shape
        cropper = partial(tf.random_crop, size = (th, tw, 3)) if crop_type == 'random'\
            else partial(tf.image.resize_image_with_crop_or_pad, target_height = th, target_width = tw)
        image_0, image_1, flow = map(cropper, (image_1, image_0, flow))

    images = tf.stack([image_0, image_1], axis = 0)
    images = tf.cast(images, tf.float32)
    images = images/255.

    return images, flow


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess = tf.Session()

    # sess = tf.Session()
    samples = []
    # with open('/Users/Daigo/Data/DAVIS/val_480p_3frames.txt', 'r') as f:
    with open('/Users/Daigo/Data/MPI-Sintel-complete/val.txt', 'r') as f:
        for i  in f.readlines():
            image_0_path, image_1_path, flow_path = i.split(',')
            flow_path = flow_path.strip()
            samples.append((image_0_path, image_1_path, flow_path))

    # parse_fn = lambda i0_path, i1_path, f_path: _parse(i0_path, i1_path, f_path)
    # train_fn = lambda image_0, image_1, flow: _preprocess(image_0, image_1, flow, 'random', (192, 224))
    # parse_fn = lambda filenames: tuple(_parse(filenames))
    parse_fn = lambda fname: tf.py_func(_read_py, [fname], [tf.uint8, tf.uint8, tf.float32])
    train_fn = lambda image_0, image_1, flow: _preprocess(image_0, image_1, flow, 'center', (192, 224))
   
    # dataset = (tf.data.Dataset.from_tensor_slices(samples)
    #            .map(parse_fn)
    #            .map(train_fn)
    #            .batch(4)
    #            .prefetch(1))
    # dataset = (tf.data.Dataset.from_tensor_slices((image_0_paths, image_1_paths, flow_paths))
    #            .shuffle(len(image_0_paths))
    #            .map(parse_fn)
    #            .map(train_fn)
    #            .batch(4)
    #            .prefetch(1))
    dataset = tf.data.Dataset.from_tensor_slices(samples)
    dataset = dataset.map(parse_fn, num_parallel_calls = 1)
    dataset = dataset.map(train_fn, num_parallel_calls = 1)
    dataset = dataset.batch(4)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_one_shot_iterator()
    next_el = iterator.get_next()
    images, flow = sess.run(next_el)
    print(f'image shape {images.shape}, flow shape {flow.shape}')
