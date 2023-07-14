import csv
import os
import tensorflow as tf

import numpy as np

#['0.458',
# ' 0.591', 
# ' 5', 
# ' -5.621', 
# ' 1', 
# ' 0.0326', 
# ' 0.568', 
# ' 0.0000154', 
# ' 0.286', 
# ' 0.654', 
# ' 184.913', 
# ' 0000uJA4xCdxThagdLkkLR', 
# ' 161187', 
# ' 3', 
# ' Cherryholmes', 
# ' Cherryholmes', 
# ' 2005-01-01', 
# ' 161186', 
# ' Heart As Cold As Stone', 
# ' 0']

def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def songDataFeatures(row):
    features = {
        'danceability':     _float_feature(float(row[0].strip())),
        'energy':           _float_feature(float(row[1].strip())),
        'key':              _int64_feature(int(row[2].strip())),
        'loudness':         _float_feature(float(row[3].strip())),
        'mode':             _int64_feature(int(row[4].strip())),
        'speechiness':      _float_feature(float(row[5].strip())),
        'acousticness':     _float_feature(float(row[6].strip())), 
        'instrumentalness': _float_feature(float(row[7].strip())),
        'liveness':         _float_feature(float(row[8].strip())),
        'valence':          _float_feature(float(row[9].strip())),
        'tempo':            _float_feature(float(row[10].strip())),
        'song_id':          _bytes_feature(row[11].strip().encode('utf-8')), 
        'duration_ms':      _int64_feature(int(row[12].strip())),
        'time_signature':   _int64_feature(int(row[13].strip())),
        'artist_name':      _bytes_feature(row[14].strip().encode('utf-8')),
        'album_name':       _bytes_feature(row[15].strip().encode('utf-8')),
        'release_date':     _int64_feature(int(row[16].strip()[:4])),
        'duration_ms':      _int64_feature(int(row[17].strip())), 
        'song_name':        _bytes_feature(row[18].strip().encode('utf-8')),
        'popularity':       _int64_feature(int(row[19].strip())) 
    }

with open('songData/sorted.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    writer = csv.writer(writeFile, delimiter=',')
    for idx, row in enumerate(reader):
        
        break