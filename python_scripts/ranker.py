import csv
import random
import os
# import tensorflow as tf
import collections
import math
import json
# import tensorflow_recommenders as tfrs
# import tensorflow_ranking as tfr
import random
import csv

import matplotlib.pyplot as plt

import numpy as np

###############################################################
#### Data Formatting
###############################################################

def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# def songDataFeatures(row):
#     try:
#         features = {
#             'danceability':     _float_feature(float(row[0].strip())),
#             'energy':           _float_feature(float(row[1].strip())),
#             'key':              _int64_feature(int(row[2].strip())),
#             'loudness':         _float_feature(float(row[3].strip())),
#             'mode':             _int64_feature(int(row[4].strip())),
#             'speechiness':      _float_feature(float(row[5].strip())),
#             'acousticness':     _float_feature(float(row[6].strip())), 
#             'instrumentalness': _float_feature(float(row[7].strip())),
#             'liveness':         _float_feature(float(row[8].strip())),
#             'valence':          _float_feature(float(row[9].strip())),
#             'tempo':            _float_feature(float(row[10].strip())),
#             'song_id':          _bytes_feature(row[11].strip().encode('utf-8')), 
#             'duration_ms':      _int64_feature(int(row[12].strip())),
#             'time_signature':   _int64_feature(int(row[13].strip())),
#             'artist_name':      _bytes_feature(row[14].strip().encode('utf-8')),
#             'album_name':       _bytes_feature(row[15].strip().encode('utf-8')),
#             'release_date':     _int64_feature(int(row[16].strip()[:4])),
#             'duration_ms':      _int64_feature(int(row[17].strip())), 
#             'song_name':        _bytes_feature(row[18].strip().encode('utf-8')),
#             'popularity':       _int64_feature(int(row[19].strip())) 
#         }
#         example_proto = tf.train.Example(features=tf.train.Features(feature=features))
#         return example_proto.SerializeToString()
#     except:
#         print(row)
#         quit()

# def generator():
#     with open('songData/fixed2.csv') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',')
#         for idx, row in enumerate(reader):
#             if idx % 10000 == 0:
#                 print(idx)
#             yield songDataFeatures(row)

# serialized_features_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())

# filename = 'songData/test.tfrecord'
# writer = tf.data.experimental.TFRecordWriter(filename)
# writer.write(serialized_features_dataset)


DEBUG_FLAG = False

#taken from geeks for geeks and modified (I'm lazy)
def isInArr(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0
 
    while low <= high:
 
        mid = (high + low) // 2
 
        # If x is greater, ignore left half
        if arr[mid]['song_id'] < x:
            low = mid + 1
 
        # If x is smaller, ignore right half
        elif arr[mid]['song_id'] > x:
            high = mid - 1
 
        # means x is present at mid
        else:
            return True
 
    # If we reach here, then the element was not present
    return False

def getFeaturesAsDict(row):
    features = {
        'danceability':     float(row[0].strip()),
        'energy':           float(row[1].strip()),
        'key':              int(row[2].strip()),
        'loudness':         float(row[3].strip()),
        'mode':             int(row[4].strip()),
        'speechiness':      float(row[5].strip()),
        'acousticness':     float(row[6].strip()), 
        'instrumentalness': float(row[7].strip()),
        'liveness':         float(row[8].strip()),
        'valence':          float(row[9].strip()),
        'tempo':            float(row[10].strip()),
        'song_id':          row[11].strip(), 
        'duration_ms':      int(row[12].strip()),
        'time_signature':   int(row[13].strip()),
        'artist_name':      row[14].strip(),
        'album_name':       row[15].strip(),
        'release_date':     int(row[16].strip()[:4]),
        'song_name':        row[18].strip(),
        'popularity':       int(row[19].strip()) 
    }
    return features


def getRandomSong():
  while True:
    f = open("songData/shuffled.csv")
    while True:
      line = f.readline()
      if not line:
        break
      yield getFeaturesAsDict(list(csv.reader([line]))[0])
    f.close()

randSongGen = getRandomSong()

def getNNegativeSongs(counter, playlistSongs):
  #songIds = map(lambda a : a["song_id"] ,playlistSongs) map for when we add more features
  nSongs = []
  playlistSongs.sort(key=lambda x : x['song_id'])
  for i in range(0, counter):
    songDict = next(randSongGen)
    while isInArr(playlistSongs, songDict['song_id']):
      songDict = next(randSongGen)
    nSongs.append(songDict)
  return nSongs


#danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, 
#liveness, valence, tempo, song_id, duration_ms, time_signature, artist_name, album_name, 
#release_date, duration_ms, song_name, popularity

#todo change tensor_slices to include other features later
#tensor_slices = {"playlistObj": {}, "songSamples":[], "ratings":[]}
tensor_slices = {
  "playlistName": [],
  "play_artist_name":[],
  "play_danceability":[],
  "play_energy":[],
  "play_key":[],
  "play_loudness":[],
  "play_mode":[],
  "play_speechiness":[],
  "play_acousticness":[],
  "play_instrumentalness":[],
  "play_liveness":[],
  "play_valence":[],
  "play_tempo":[],
  "play_duration_ms":[],
  "play_time_signature":[],
  "play_release_date":[],
  "play_popularity":[],
  # Song data portion
  "artist_name":[],
  "danceability":[],
  "energy":[],
  "key":[],
  "loudness":[],
  "mode":[],
  "speechiness":[],
  "acousticness":[],
  "instrumentalness":[],
  "liveness":[],
  "valence":[],
  "tempo":[],
  "duration_ms":[],
  "time_signature":[],
  "release_date":[],
  "popularity":[],
  # ratings portion
  "ratings":[]
}

# tensor_slices = {
#   "playlistName": [],
#   # Song data portion
#   "songIds":[],
#   # ratings portion
#   "ratings":[]
# }
playlistSize = 50
num_samples = 50

next(randSongGen) #prime coroutine


tensor_temp_slices = {
  "artist_name":[],
  "danceability":[],
  "energy":[],
  "key":[],
  "loudness":[],
  "mode":[],
  "speechiness":[],
  "acousticness":[],
  "instrumentalness":[],
  "liveness":[],
  "valence":[],
  "tempo":[],
  "duration_ms":[],
  "time_signature":[],
  "release_date":[],
  "popularity":[],
}
def addSongDataToSlice(songObj):
  tensor_temp_slices["artist_name"].append(songObj["artist_name"])
  tensor_temp_slices["danceability"].append(float(songObj["danceability"]))
  tensor_temp_slices["energy"].append(float(songObj["energy"]))
  tensor_temp_slices["key"].append(int(songObj["key"]))
  tensor_temp_slices["loudness"].append(float(songObj["loudness"]))
  tensor_temp_slices["mode"].append(float(songObj["mode"]))
  tensor_temp_slices["speechiness"].append(float(songObj["speechiness"]))
  tensor_temp_slices["acousticness"].append(float(songObj["acousticness"]))
  tensor_temp_slices["instrumentalness"].append(float(songObj["instrumentalness"]))
  tensor_temp_slices["liveness"].append(float(songObj["liveness"]))
  tensor_temp_slices["valence"].append(float(songObj["valence"]))
  tensor_temp_slices["tempo"].append(float(songObj["tempo"]))
  try:
    tensor_temp_slices["duration_ms"].append(int(songObj["duration_ms"]))
  except:
    tensor_temp_slices["duration_ms"].append(int(186533))
    print("WTF")
  tensor_temp_slices["time_signature"].append(int(songObj["time_signature"]))
  tensor_temp_slices["release_date"].append(int(songObj["release_date"]))
  tensor_temp_slices["popularity"].append(float(songObj["popularity"]))
  
def flushTempSlice():
  tensor_slices["artist_name"].append(tensor_temp_slices["artist_name"])
  tensor_slices["danceability"].append(tensor_temp_slices["danceability"])
  tensor_slices["energy"].append(tensor_temp_slices["energy"])
  tensor_slices["key"].append(tensor_temp_slices["key"])
  tensor_slices["loudness"].append(tensor_temp_slices["loudness"])
  tensor_slices["mode"].append(tensor_temp_slices["mode"])
  tensor_slices["speechiness"].append(tensor_temp_slices["speechiness"])
  tensor_slices["acousticness"].append(tensor_temp_slices["acousticness"])
  tensor_slices["instrumentalness"].append(tensor_temp_slices["instrumentalness"])
  tensor_slices["liveness"].append(tensor_temp_slices["liveness"])
  tensor_slices["valence"].append(tensor_temp_slices["valence"])
  tensor_slices["tempo"].append(tensor_temp_slices["tempo"])
  tensor_slices["duration_ms"].append(tensor_temp_slices["duration_ms"])
  tensor_slices["time_signature"].append(tensor_temp_slices["time_signature"])
  tensor_slices["release_date"].append(tensor_temp_slices["release_date"])
  tensor_slices["popularity"].append(tensor_temp_slices["popularity"])
  for k in tensor_temp_slices.keys():
    tensor_temp_slices[k] = []

def flushPlaylistSlice():
  tensor_slices["play_artist_name"].append(tensor_temp_slices["artist_name"])
  tensor_slices["play_danceability"].append(tensor_temp_slices["danceability"])
  tensor_slices["play_energy"].append(tensor_temp_slices["energy"])
  tensor_slices["play_key"].append(tensor_temp_slices["key"])
  tensor_slices["play_loudness"].append(tensor_temp_slices["loudness"])
  tensor_slices["play_mode"].append(tensor_temp_slices["mode"])
  tensor_slices["play_speechiness"].append(tensor_temp_slices["speechiness"])
  tensor_slices["play_acousticness"].append(tensor_temp_slices["acousticness"])
  tensor_slices["play_instrumentalness"].append(tensor_temp_slices["instrumentalness"])
  tensor_slices["play_liveness"].append(tensor_temp_slices["liveness"])
  tensor_slices["play_valence"].append(tensor_temp_slices["valence"])
  tensor_slices["play_tempo"].append(tensor_temp_slices["tempo"])
  tensor_slices["play_duration_ms"].append(tensor_temp_slices["duration_ms"])
  tensor_slices["play_time_signature"].append(tensor_temp_slices["time_signature"])
  tensor_slices["play_release_date"].append(tensor_temp_slices["release_date"])
  tensor_slices["play_popularity"].append(tensor_temp_slices["popularity"])
  for k in tensor_temp_slices.keys():
    tensor_temp_slices[k] = []
  

# take half the songs and make them part of the playlist (pad to playlistSize by bag of words method)
# take other half and add positive ratings.
# pad ratings to num_samples using negative examples from songs
def addTensorSlices(playlistName, songsToSplit):
  if len(songsToSplit) <= 1:
    return
  tensor_slices["playlistName"].append(playlistName)
  random.shuffle(songsToSplit)
  num_to_take = int(len(songsToSplit)/2)
  playlistSongs = songsToSplit[:num_to_take]
  ratingSongs = songsToSplit[num_to_take:]
  ratings = []

  ratingSongCount = 0
  negativeSongCount = 0
  negativeSongs = getNNegativeSongs(num_samples - len(ratingSongs), songsToSplit)
  while ratingSongCount < len(ratingSongs) or negativeSongCount < len(negativeSongs):
      if ratingSongCount >= len(ratingSongs):
        addSongDataToSlice(negativeSongs[negativeSongCount])
        ratings.append(0.)
        negativeSongCount += 1
        continue
      elif negativeSongCount >= len(negativeSongs):
        addSongDataToSlice(ratingSongs[ratingSongCount])
        ratings.append(1.)
        ratingSongCount += 1
        continue
      currRand = random.randint(0, num_samples)
      if currRand > len(ratingSongs):
        addSongDataToSlice(negativeSongs[negativeSongCount])
        ratings.append(0.)
        negativeSongCount += 1
      else:
        addSongDataToSlice(ratingSongs[ratingSongCount])
        ratings.append(1.)
        ratingSongCount += 1
  tensor_slices["ratings"].append(ratings)
  flushTempSlice()
  playlistSongsAdded = 0
  while playlistSongsAdded < playlistSize:
    random.shuffle(playlistSongs)
    for song in playlistSongs:
      addSongDataToSlice(song)
      playlistSongsAdded += 1
      if playlistSongsAdded >= playlistSize:
        break
  flushPlaylistSlice()

  

# # make list-wise training data
chunkSize = 70

for counter, currFilename in enumerate(os.listdir('playlistSongData')):
  fd = open(os.path.join('playlistSongData', currFilename))
  currfile = json.load(fd)
  currWrite = open('smallSlices/slice'+str(counter), 'w')
  example_list = []
  for idx, playlist in enumerate(currfile["playlists"]):
    currSongIds = []
    for track in playlist["tracks"]:
      currSongIds.append(track)
    
    # split songs into chunks if playlist longer than 100
    if len(currSongIds) > chunkSize:
      numSplits = int(math.ceil(len(currSongIds)/float(chunkSize)))
      splitSize = int(len(currSongIds)/numSplits)
      for chunkIndex in range(0, len(currSongIds), splitSize):
        addTensorSlices(playlist["name"] ,currSongIds[chunkIndex:chunkIndex + splitSize])
    else:
      addTensorSlices(playlist["name"], currSongIds)

  json.dump(tensor_slices, currWrite)
  fd.close()
  currWrite.close()
  for k in tensor_slices.keys():
    tensor_slices[k] = []

  if (counter+1) % 10 == 0:
    print(counter)
  

quit()

# fd = open(currfilename)
# currfile = json.load(fd)
# example_list = []
# for idx, playlist in enumerate(currfile["playlists"]):
#   currSongIds = []
#   for track in playlist["tracks"]:
#     currSongIds.append(track)

#   # split songs into chunks if playlist longer than 100
#   if len(currSongIds) > 100:
#     numSplits = int(math.ceil(len(currSongIds)/100.0))
#     splitSize = int(len(currSongIds)/numSplits)
#     for chunkIndex in range(0, len(currSongIds), splitSize):
#       addTensorSlices(playlist["name"] ,currSongIds[chunkIndex:chunkIndex + splitSize])
#   else:
#     addTensorSlices(playlist["name"], currSongIds)

# fd.close()


###############################################################
#### Feature Preprocessing
###############################################################

# feature_description = {
#     'danceability':     tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#     'energy':           tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#     'key':              tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'loudness':         tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#     'mode':             tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'speechiness':      tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#     'acousticness':     tf.io.FixedLenFeature([], tf.float32, default_value=0.0), 
#     'instrumentalness': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#     'liveness':         tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#     'valence':          tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#     'tempo':            tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#     'song_id':          tf.io.FixedLenFeature([], tf.string, default_value=''), 
#     'duration_ms':      tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'time_signature':   tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'artist_name':      tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'album_name':       tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'release_date':     tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'duration_ms':      tf.io.FixedLenFeature([], tf.int64, default_value=0), 
#     'song_name':        tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'popularity':       tf.io.FixedLenFeature([], tf.int64, default_value=0)
# }

# def _parse_function(example_proto):
#   # Parse the input `tf.train.Example` proto using the dictionary above.
#   return tf.io.parse_single_example(example_proto, feature_description)

# song_dataset = tf.data.TFRecordDataset("songData/test.tfrecord")
# songs_decoded = song_dataset.map(_parse_function)

dance_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=0.5504335813296364,
    variance=0.03405148474678806
)
energy_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=0.5849082174284876,
    variance=0.0707619008806274
)
key_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=5.261574032109226,
    variance=12.666169201737365
)
loudness_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=-9.659818722024411,
    variance=31.632226537794434
)
mode_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=0.6549471987223165,
    variance=0.22599136561602995
)
speechiness_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=0.08923561047456818,
    variance=0.013273295737293848
)
acousticness_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=0.3547138903422462,
    variance=0.12557698581758756
)
instrumentalness_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=0.22083003656308406,
    variance=0.12208968452189842
)
liveness_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=0.2091079115120563,
    variance=0.0360289506491073
)
valence_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=0.47568901929398116,
    variance=0.07288552800177106
)
tempo_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=119.99103445094083,
    variance=895.1464628482091
)
duration_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=247437.86104705842,
    variance=24369437559.779995
)
signature_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=3.879958064956869,
    variance=0.222650271393
)
date_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=2006.3176734275353,
    variance=2625.4987410055155
)
popularity_normalization = tf.keras.layers.Normalization(
    axis=None,
    mean=11.03368422169683,
    variance=206.8401484846528
)


#We set up a large number of bins to reduce the chance of hash collisions.
num_hashing_bins = 10_000_000

#until the test model works we will use this for song ids
artist_name_hashing = tf.keras.layers.Hashing(
    num_bins=num_hashing_bins
)

#we will remove this later once the test model works
playlist_name_hashing = tf.keras.layers.Hashing(
    num_bins=num_hashing_bins
)

###############################################################
#### Models
###############################################################

embedding_dimension = 32

class QueryModel():

  def call(self, features, artist_embedding_func):
    # these are a [batch_size, playlistSize] tensor
    dance_embedding = dance_normalization(features["play_danceability"])
    energy_embedding = energy_normalization(features["play_energy"])
    key_embedding = key_normalization(features["play_key"])
    loudness_embedding = loudness_normalization(features["play_loudness"])
    mode_embedding = mode_normalization(features["play_mode"])
    speechiness_embedding = speechiness_normalization(features["play_speechiness"])
    acousticness_embedding = acousticness_normalization(features["play_acousticness"])
    instrumentalness_embedding = instrumentalness_normalization(features["play_instrumentalness"])
    liveness_embedding = liveness_normalization(features["play_liveness"])
    valence_embedding = valence_normalization(features["play_valence"])
    tempo_embedding = tempo_normalization(features["play_tempo"])
    duration_embedding = duration_normalization(features["play_duration_ms"])
    signature_embedding = signature_normalization(features["play_time_signature"])
    date_embedding = date_normalization(features["play_release_date"])
    popularity_embedding = popularity_normalization(features["play_popularity"])

    # artist embeddings are a [batch_size, playlistSize, embedding_dim]
    # tensor.
    artist_embeddings = artist_embedding_func(features["play_artist_name"])

    reshaped_artist_embed = tf.reshape(artist_embeddings, [-1, playlistSize * embedding_dimension])
    
    return tf.concat([
        reshaped_artist_embed,
        dance_embedding,
        energy_embedding,
        key_embedding,
        loudness_embedding,
        mode_embedding,
        speechiness_embedding,
        acousticness_embedding,
        instrumentalness_embedding,
        liveness_embedding,
        valence_embedding,
        tempo_embedding,
        duration_embedding,
        signature_embedding,
        date_embedding,
        popularity_embedding
        ], 1)

class RankingModel(tfrs.Model):

  def __init__(self, loss):
    super().__init__()
    
    self.playlist_model = QueryModel()

    # Compute embeddings for users.
    self.playlist_embeddings = tf.keras.Sequential([
      playlist_name_hashing,
      tf.keras.layers.Embedding(num_hashing_bins, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.artist_embeddings = tf.keras.Sequential([
      artist_name_hashing,
      tf.keras.layers.Embedding(num_hashing_bins, embedding_dimension)
    ])

    # Compute predictions.
    self.score_model = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
    ])

    self.task = tfrs.tasks.Ranking(
      loss=loss,
      metrics=[
        tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
        tf.keras.metrics.RootMeanSquaredError()
      ]
    )

# tensor_slices = {"playlistName": [], "songIds":[], "ratings":[]}
  def call(self, features):
    # We first convert the id features into embeddings.
    # playlist_embeddings are a [batch_size, embedding_dim] tensor.
    playlist_embeddings = self.playlist_embeddings(features["playlistName"])
    
    # playlist_features are a [batch_size, playlistSize * num_features] tensor
    playlist_features = self.playlist_model.call(features, self.artist_embeddings)


    # full_playlist_embeddings are a [batch_size, playlistSize * num_features + embedding_dim] tensor
    full_playlist_embeddings = tf.concat([playlist_embeddings,playlist_features], 1)

    # artist embeddings are a [batch_size, sample_size, embedding_dim]
    # tensor.
    artist_embeddings = self.artist_embeddings(features["artist_name"])

    # these are a [batch_size, sample_size, 1] tensor
    dance_embedding = tf.expand_dims( dance_normalization(features["danceability"]), -1)
    energy_embedding = tf.expand_dims( energy_normalization(features["energy"]), -1)
    key_embedding = tf.expand_dims( key_normalization(features["key"]), -1)
    loudness_embedding = tf.expand_dims( loudness_normalization(features["loudness"]), -1)
    mode_embedding = tf.expand_dims( mode_normalization(features["mode"]), -1)
    speechiness_embedding = tf.expand_dims( speechiness_normalization(features["speechiness"]), -1)
    acousticness_embedding = tf.expand_dims( acousticness_normalization(features["acousticness"]), -1)
    instrumentalness_embedding = tf.expand_dims( instrumentalness_normalization(features["instrumentalness"]), -1)
    liveness_embedding = tf.expand_dims( liveness_normalization(features["liveness"]), -1)
    valence_embedding = tf.expand_dims( valence_normalization(features["valence"]), -1)
    tempo_embedding = tf.expand_dims( tempo_normalization(features["tempo"]), -1)
    duration_embedding = tf.expand_dims( duration_normalization(features["duration_ms"]), -1)
    signature_embedding = tf.expand_dims( signature_normalization(features["time_signature"]), -1)
    date_embedding = tf.expand_dims( date_normalization(features["release_date"]), -1)
    popularity_embedding = tf.expand_dims( popularity_normalization(features["popularity"]), -1)

    # We want to concatenate user embeddings with movie emebeddings to pass
    # them into the ranking model. To do so, we need to reshape the user
    # embeddings to match the shape of movie embeddings.
    list_length = features["artist_name"].shape[1]
    user_embedding_repeated = tf.repeat(
        tf.expand_dims(full_playlist_embeddings, 1), [list_length], axis=1)

    # Once reshaped, we concatenate and pass into the dense layers to generate
    # predictions.
    concatenated_embeddings = tf.concat(
        [user_embedding_repeated,
        artist_embeddings,
        dance_embedding,
        energy_embedding,
        key_embedding,
        loudness_embedding,
        mode_embedding,
        speechiness_embedding,
        acousticness_embedding,
        instrumentalness_embedding,
        liveness_embedding,
        valence_embedding,
        tempo_embedding,
        duration_embedding,
        signature_embedding,
        date_embedding,
        popularity_embedding
        ], 2)

    return self.score_model(concatenated_embeddings)

  def compute_loss(self, features, training=False):
    labels = features.pop("ratings")

    scores = self(features)

    return self.task(
        labels=labels,
        predictions=tf.squeeze(scores, axis=-1),
    )

epochs = 30

# ratings = tf.data.Dataset.from_tensor_slices(tensor_slices)

# tf.random.set_seed(42)

# shuffled = ratings.shuffle(1000)

# train = shuffled.take(1111)
# test = shuffled.skip(1111).take(277)

# cached_train = train.shuffle(1000).batch(8192).cache()
# cached_test = test.batch(4096).cache()

def generateTestData():
  for i in range(1,2):
    fd = open('slicesRemix/slice'+str(i))
    currSlices = json.load(fd)
    currData = tf.data.Dataset.from_tensor_slices(currSlices)
    for datum in currData:
      yield datum
    fd.close()

def generateTrainData():
  for i in range(0,1):
    fd = open('slicesRemix/slice'+str(i))
    currSlices = json.load(fd)
    currData = tf.data.Dataset.from_tensor_slices(currSlices)
    for datum in currData:
      yield datum
    fd.close()


# # ratings = 

# fileDataset = tf.data.Dataset.list_files("slices/*").map(parseData)
# print(tensor_type)

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

data_format2 = {
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
  'artist_name': tf.TensorSpec(shape=(50,), dtype=tf.string), 
  'danceability': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'energy': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'key': tf.TensorSpec(shape=(50,), dtype=tf.int32), 
  'loudness': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'mode': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'speechiness': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'acousticness': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'instrumentalness': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'liveness': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'valence': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'tempo': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'duration_ms': tf.TensorSpec(shape=(50,), dtype=tf.int32), 
  'time_signature': tf.TensorSpec(shape=(50,), dtype=tf.int32), 
  'release_date': tf.TensorSpec(shape=(50,), dtype=tf.int32), 
  'popularity': tf.TensorSpec(shape=(50,), dtype=tf.float32), 
  'ratings': tf.TensorSpec(shape=(50,), dtype=tf.float32)
}

train_dataset = tf.data.Dataset.from_generator(generateTrainData, output_signature=data_format).batch(1024).cache()

# Sometimes one or the other loss calc does better
listwise_model = RankingModel(tfr.keras.losses.ListMLELoss())
# listwise_model = RankingModel(tfr.keras.losses.PairwiseHingeLoss())

listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

history = listwise_model.fit(train_dataset, epochs=epochs, verbose=False)

test_dataset = tf.data.Dataset.from_generator(generateTestData, output_signature=data_format).batch(1024).cache()

listwise_model_result = listwise_model.evaluate(test_dataset, return_dict=True)
print("NDCG of the ListMLE model: {:.4f}".format(listwise_model_result["ndcg_metric"]))


# plotEpochs= [x + 1 for x in range(epochs)]
# plt.plot(plotEpochs, history.history["ndcg_metric"])
# plt.title("NDCG vs epoch")
# plt.xlabel("epoch")
# plt.ylabel("NDCG")
# plt.savefig('NDCG.png')

# tf.saved_model.save(listwise_model, "test-model")
