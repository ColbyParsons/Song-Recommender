import csv
import random
import os
import tensorflow as tf
import collections
import json
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr

import numpy as np


playlistSize = 50
num_samples = 100

validateRun = 'valid-5'
print(validateRun)
def generateTestData():
  for i in range(0,200):
    fd = open('slicesRemix/slice'+str(i))
    currSlices = json.load(fd)
    currData = tf.data.Dataset.from_tensor_slices(currSlices)
    for datum in currData:
      yield datum
    fd.close()

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
  'popularity': tf.TensorSpec(shape=(100,), dtype=tf.float32), 
  'ratings': tf.TensorSpec(shape=(100,), dtype=tf.float32)
}

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

listwise_model = tf.saved_model.load(validateRun)

test_dataset = tf.data.Dataset.from_generator(generateTestData, output_signature=data_format).batch(256).cache()

listwise_model_result = listwise_model.evaluate(test_dataset, return_dict=True)
print("NDCG of the ListMLE model: {:.4f}".format(listwise_model_result["ndcg_metric"]))

