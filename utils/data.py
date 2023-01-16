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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def load_kinetics(config, fold=0):
    
    X_train = np.load('/media/Datasets/kinetics/train_data_joint.npy') #[...,0] # get first pose
    X_train = np.moveaxis(X_train,1,3)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_train = X_train[:,::config['SUBSAMPLE'],:]
    y_train = np.load('/media/Datasets/kinetics/train_label.pkl', allow_pickle=True)
    y_train = np.transpose(np.array(y_train))[:,1].astype('float32') # get class only
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=config['VAL_SIZE'],
                                                      random_state=config['SEEDS'][fold],
                                                      stratify=y_train)
    
    X_test = np.load('/media/Datasets/kinetics/val_data_joint.npy') #[...,0] # get first pose
    X_test = np.moveaxis(X_test,1,3)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    X_test = X_test[:,::config['SUBSAMPLE'],:]
    y_test = np.load('/media/Datasets/kinetics/val_label.pkl', allow_pickle=True)
    y_test = np.transpose(np.array(y_test))[:,1].astype('float32') # get class only
    
    train_gen = callable_gen(kinetics_generator(X_train, y_train, config['BATCH_SIZE']))
    val_gen = callable_gen(kinetics_generator(X_val, y_val, config['BATCH_SIZE']))
    test_gen = callable_gen(kinetics_generator(X_test, y_test, config['BATCH_SIZE']))
    
    return train_gen, val_gen, test_gen, len(y_train), len(y_test)

def load_mpose(dataset, split, verbose=False):
    
    dataset = MPOSE(pose_extractor=dataset, 
                    split=split, 
                    preprocess=None, 
                    velocities=False, 
                    remove_zip=False,
                    verbose=verbose)
    dataset.reduce_keypoints()
    dataset.scale_and_center()
    #dataset.remove_confidence()
    dataset.flatten_features()
    #dataset.reduce_labels()
    
    return dataset.get_data()

def random_flip(x, y):
    time_steps = x.shape[0]
    n_features = x.shape[1]
    if not n_features % 2:
        x = tf.reshape(x, (time_steps, n_features//2, 2))

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if choice >= 0.5:
            x = tf.math.multiply(x, [-1.0,1.0])
    else:
        x = tf.reshape(x, (time_steps, n_features//3, 3))

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if choice >= 0.5:
            x = tf.math.multiply(x, [-1.0,1.0,1.0])
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

def kinetics_generator(X, y, batch_size):
    while True:
        ind_list = [i for i in range(X.shape[0])]
        shuffle(ind_list)
        X  = X[ind_list,...]
        y = y[ind_list]
        for count in range(len(y)):
            yield (X[count], y[count])
        
def callable_gen(_gen):
        def gen():
            for x,y in _gen:
                 yield x,y
        return gen