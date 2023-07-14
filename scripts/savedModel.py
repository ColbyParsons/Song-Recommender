import csv
import random
import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr
import collections
import math
import json
import random
import csv

import matplotlib.pyplot as plt

import numpy as np


model = "2-model"

loaded = tf.saved_model.load(model)

sliceFile = 'slicesRemix/slice42'

data_format = {
  'playlistName': tf.TensorSpec(shape=(), dtype=tf.string), 
  'play_artist_name': tf.TensorSpec(shape=(50,), dtype=tf.string), 
  'play_danceability': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'play_energy': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'play_key': tf.TensorSpec(shape=(50,), dtype=tf.int32), 
  'play_loudness': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'play_mode': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'play_speechiness': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'play_acousticness': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'play_instrumentalness': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'play_liveness': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'play_valence': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'play_tempo': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'play_duration_ms': tf.TensorSpec(shape=(50,), dtype=tf.int32), 
  'play_time_signature': tf.TensorSpec(shape=(50,), dtype=tf.int32), 
  'play_release_date': tf.TensorSpec(shape=(50,), dtype=tf.int32), 
  'play_popularity': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'artist_name': tf.TensorSpec(shape=(100,), dtype=tf.string), 
  'danceability': tf.TensorSpec(shape=(100,), dtype=tf.float32), 
  'energy': tf.TensorSpec(shape=(100,), dtype=tf.float32), 
  'key': tf.TensorSpec(shape=(100,), dtype=tf.int32), 
  'loudness': tf.TensorSpec(shape=(100,), dtype=tf.float32), 
  'mode': tf.TensorSpec(shape=(100,), dtype=tf.float32), 
  'speechiness': tf.TensorSpec(shape=(100,), dtype=tf.float32), 
  'acousticness': tf.TensorSpec(shape=(100,), dtype=tf.float32), 
  'instrumentalness': tf.TensorSpec(shape=(100,), dtype=tf.float32), 
  'liveness': tf.TensorSpec(shape=(100,), dtype=tf.float32), 
  'valence': tf.TensorSpec(shape=(100,), dtype=tf.float32), 
  'tempo': tf.TensorSpec(shape=(100,), dtype=tf.float32), 
  'duration_ms': tf.TensorSpec(shape=(100,), dtype=tf.int32), 
  'time_signature': tf.TensorSpec(shape=(100,), dtype=tf.int32), 
  'release_date': tf.TensorSpec(shape=(100,), dtype=tf.int32), 
  'popularity': tf.TensorSpec(shape=(100,), dtype=tf.float32)
}

# def generateData():
with open(sliceFile) as currFile:
    currJSON = json.load(currFile)
    data = {}
    for key in currJSON.keys():
        data[key] = currJSON[key][0]
    print(data['ratings'])
    data.pop('ratings')
    for key in data.keys():
        if key == 'playlistName':
            data[key] = tf.reshape(tf.convert_to_tensor(data[key]), (1,))
            continue
        data[key] = tf.reshape(tf.convert_to_tensor(data[key]), (1, -1))
    # print(data)
# dataSlice = tf.data.Dataset.from_tensors(data)
# for datum in dataSlice:
#     yield datum

# dataset = tf.data.Dataset.from_generator(generateData, output_signature=data_format)
    # print(len(currJSON))
    # print(obj1)
    # print(obj2)
    print(loaded(data))