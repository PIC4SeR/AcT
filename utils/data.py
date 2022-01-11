# Copyright 2021 Simone Angarano. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
from mpose import MPOSE

def load_mpose(dataset, split, verbose=False):
    
    dataset = MPOSE(pose_extractor=dataset, 
                    split=split, 
                    preprocess=None, 
                    velocities=True, 
                    remove_zip=False,
                    verbose=verbose)
    dataset.reduce_keypoints()
    dataset.scale_and_center()
    dataset.remove_confidence()
    dataset.flatten_features()
    
    return dataset.get_data()

def random_flip(x, y):
    time_steps = x.shape[0]
    n_features = x.shape[1]
    x = tf.reshape(x, (time_steps, n_features//2, 2))
    
    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    if choice >= 0.5:
        x = tf.math.multiply(x, [-1.0,1.0])
        
    x = tf.reshape(x, (time_steps,-1))
    return x, y

def random_noise(x, y):
    time_steps = tf.shape(x)[0]
    n_features = tf.shape(x)[1]
    noise = tf.random.normal((time_steps, n_features), mean=0.0, stddev=0.05, dtype=tf.float64)
    x = x + noise
    return x, y

def one_hot(x, y, n_classes):
    return x, tf.one_hot(y, n_classes)