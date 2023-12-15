import tensorflow as tf
import numpy as np
#import keras
#from keras import layers
def as_frame(events):
    coords = events[:, :2].numpy()
    frame = np.zeros((128,128,3), dtype=np.unit8)
    frame[coords[:, 0], coords[:, 1], :] = [255, 255, 255]
    return frame

def as_voxelgrid(events):
    voxelgrid, edges = np.histogramdd(
        events.numpy()[:, :3],
        bins=(32, 32, 32),
        range=((0, 128), (0, 128), (0, 100))
    )
    voxelgrid = voxelgrid.astype(np.float32) / voxelgrid.max()
    return voxelgrid
def prepare(ds, shuffle=False):
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.Resizing(32,32),
        tf.keras.layers.Rescaling(1./255)
    ])
    ds = ds.map(lambda x,y: (resize_and_rescale(x),y),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(100)
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)