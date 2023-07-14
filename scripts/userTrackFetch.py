import requests
import base64
import json
import random
from datetime import date
import tensorflow as tf
import sys

def handleRequest(params, endpoint, state, name):
    num429s = 0
    max429s = 3
    r = {}
    while num429s < max429s:
        headers = { 'Authorization': 'Bearer ' + state.getBearer(), 'Content-Type' : 'application/json' }
        r = requests.get(endpoint, headers=headers, params=params)
        if r.status_code == 200:
            return r.json()
        else:
            num429s += 1
    print(params)
    print(r.status_code)
    print(name)
    print(bearerStr)
    print("SOMETHING WENT VERY WRONG")
    quit()


class httpState:
    def __init__(self):
        self.authTokens = [
            'b4a2d661229f4f198c6660f74c61e3dd:bb42d2f062864534a4523cda15a8ee7d',
            '5ec85080826944ac938df2b872c71518:d255ba858f254be2912bc060a79044fe',
            'dd022171b2d944dcb0102589f0947b4f:34f249915a1642e886490d9311d97daf'
        ]
        self.currentTokenIndex = 0
        self.currentBearer = self.getNewBearer()

    def setBearer(self, bearer):
        self.currentBearer = bearer

    def getNewBearer(self):
        messageBytes = self.authTokens[self.currentTokenIndex].encode('ascii')
        bearerUrl = "https://accounts.spotify.com/api/token"
        bearerHeaders = { 'Authorization': 'Basic ' + base64.urlsafe_b64encode(messageBytes).decode('ascii') }
        data = { 'grant_type' : 'client_credentials' }
        r = requests.post(bearerUrl, headers=bearerHeaders, data=data )
        return r.json()['access_token']

    def updateBearer(self):
        self.currentBearer = self.getNewBearer()

    def getBearer(self):
        return self.currentBearer

    def switchAuthToken(self):
        self.currentTokenIndex = (self.currentTokenIndex + 1) % len(self.authTokens)

    def getNumAuthTokens(self):
        return len(self.authTokens)

featureBatchSize = 50
tracksBatchSize = 50

class dataState:
    def __init__(self):
        self.songData = []
        self.uris = []
        self.cleanup = []
        self.filled = 0

    def addItem(self, uri, artistName, relDate):
        self.songData.append({ # empty fields get populated later in addFeaturesToData
            "id": uri, # populate on init
            "artist_name": artistName, # populate on init
            "danceability": None,
            "energy": None,
            "key": None,
            "loudness": None,
            "mode": None,
            "speechiness": None,
            "acousticness": None,
            "instrumentalness": None,
            "liveness": None,
            "valence": None,
            "tempo": None,
            "duration_ms": None,
            "time_signature": None,
            "release_date": relDate[:4], # populate on init
            "popularity": None # populate from song data
        })
        self.uris.append(uri)

    def getTracksAudioFeatures(self, startIndex, state):
        endpoint = 'https://api.spotify.com/v1/audio-features'
        params = {
            'ids' : ','.join(self.uris[startIndex:startIndex+featureBatchSize])
        }
        return handleRequest(params, endpoint, state, "getTracksAudioFeatures")
        
    
    def getTracks(self, startIndex, state):
        endpoint = 'https://api.spotify.com/v1/tracks'
        params = {
            'ids' : ','.join(self.uris[startIndex:startIndex+tracksBatchSize])
        }
        return handleRequest(params, endpoint, state, "getTracks")

    def addFeaturesToData(self, req_state):
        for uriIndex in range(self.filled, len(self.uris), tracksBatchSize):
            featureJSON = self.getTracksAudioFeatures(uriIndex, req_state)
            for songIndex, songFeatures in enumerate(featureJSON['audio_features']):
                if not songFeatures:
                    self.cleanup.append(uriIndex + songIndex)
                    continue
                currFeatures = self.songData[uriIndex + songIndex]
                currFeatures['danceability'] = songFeatures['danceability']
                currFeatures['energy'] = songFeatures['energy']
                currFeatures['key'] = songFeatures['key']
                currFeatures['loudness'] = songFeatures['loudness']
                currFeatures['mode'] = songFeatures['mode']
                currFeatures['speechiness'] = songFeatures['speechiness']
                currFeatures['acousticness'] = songFeatures['acousticness']
                currFeatures['instrumentalness'] = songFeatures['instrumentalness']
                currFeatures['liveness'] = songFeatures['liveness']
                currFeatures['valence'] = songFeatures['valence']
                currFeatures['tempo'] = songFeatures['tempo']
                currFeatures['duration_ms'] = songFeatures['duration_ms']
                currFeatures['time_signature'] = songFeatures['time_signature']
        for toClean in reversed(self.cleanup):
            self.songData.pop(toClean)
            self.uris.pop(toClean)
        self.cleanup = []

    def addPopularityToData(self, req_state):
        for uriIndex in range(self.filled, len(self.uris), featureBatchSize):
            trackJSON = self.getTracks(uriIndex, req_state)
            for songIndex, songFeatures in enumerate(trackJSON['tracks']):
                if not songFeatures:
                    self.cleanup.append(uriIndex + songIndex)
                    continue
                self.songData[uriIndex + songIndex]['popularity'] = songFeatures['popularity']
        for toClean in reversed(self.cleanup):
            self.songData.pop(toClean)
            self.uris.pop(toClean)
        self.cleanup = []
        if len(self.uris) > 0:
            self.filled = len(self.uris) - 1

    def writeSongData(self, filename):
        writeFile = open(filename, 'w')
        json.dump(self.songData, writeFile)
        writeFile.close()

    def print(self):
        print(self.songData)
        print(len(self.songData))
        print("URI LEN: "+ str(len(self.uris)))

    def getNumTracks(self):
        return len(self.songData)

    def fixLength(self, length):
        while len(self.songData) < length:
            index = random.randint(0,len(self.songData) - 1)
            self.songData.append(self.songData[index])
            self.uris.append(self.uris[index])
    
    def fixLengthModulo(self, length):
        while (len(self.songData) % length) != 0:
            index = random.randint(0,len(self.songData) - 1)
            self.songData.append(self.songData[index])
            self.uris.append(self.uris[index])
    
    def getSongData(self):
        return self.songData


# 0 < limit <= 50
# 0 <= offset <= 1000
# limit + offset <= 1000s
def getNewSongs(limit, offset, state):
    endpoint = 'https://api.spotify.com/v1/search'
    params = {
        'q' : 'tag:new',
        'type' : 'album',
        'limit' : str(limit),
        'offset' : str(offset),
        'market' : 'CA' #TODO fix market for live application
    }
    return handleRequest(params, endpoint, state, "getNewSongs")



# gets song uris from /search api album type return
def getSongUris(albumJson, state, song_state):
    for item in albumJson['albums']['items']:
        getSongUrisFromAlbum(item['id'], item['artists'][0]['name'], item['release_date'], state, song_state)


def getLikedSongs(state, song_state, offset, limit):
    endpoint = 'https://api.spotify.com/v1/me/tracks'
    params = {
        'limit' : str(limit),
        'offset' : str(offset)
    }
    for item in handleRequest(params, endpoint, state, "getLikedSongs")['items']:
        song_state.addItem(item["track"]['id'], item["track"]["artists"][0]["name"], item["track"]["album"]["release_date"])


sample_tracks = []
def getTopTracks(state, song_state, offset, limit):
    endpoint = 'https://api.spotify.com/v1/me/top/tracks'
    params = {
        'limit' : str(limit),
        'offset' : str(offset),
        'time_range' : 'medium_term'
    }
    counter = 0
    for item in handleRequest(params, endpoint, state, "getTopTracks")['items']:
        song_state.addItem(item['id'], item["artists"][0]["name"], item["album"]["release_date"])
        if counter < 10:
            sample_tracks.append(item['id'])
            counter += 1
    params = {
        'limit' : str(limit),
        'offset' : str(offset),
        'time_range' : 'short_term'
    }
    counter = 0
    for item in handleRequest(params, endpoint, state, "getTopTracks")['items']:
        song_state.addItem(item['id'], item["artists"][0]["name"], item["album"]["release_date"])
        if counter < 10:
            sample_tracks.append(item['id'])
            counter += 1
    params = {
        'limit' : str(5),
        'offset' : str(0),
        'time_range' : 'short_term'
    }
    for item in handleRequest(params, endpoint, state, "getTopTracks")['items']:
        sample_tracks.append(item['id'])

def getRecTracks(state, song_state):
    for i in range(0, 25, 5):
        endpoint = 'https://api.spotify.com/v1/recommendations'
        params = {
            'limit' : "100",
            "seed_tracks" : ','.join(sample_tracks[i:i+5])
        }
        for item in handleRequest(params, endpoint, state, "getRecTracks")['tracks']:
            song_state.addItem(item['id'], item["artists"][0]["name"], item["album"]["release_date"])

yearTrackLimit = 50
def getYearTracks(state, song_state, offset):
    endpoint = 'https://api.spotify.com/v1/search'
    params = {
        'limit' : str(yearTrackLimit),
        'offset' : str(offset),
        'type' : 'track',
        'q' : 'year:' + str(date.today().year)
    }
    for item in handleRequest(params, endpoint, state, "getYearTracks")['tracks']['items']:
        song_state.addItem(item['id'], item["artists"][0]["name"], item["album"]["release_date"])

numLikedSongs = 10
numTopTracks = 20

http_state = httpState()
playlist_data_state = dataState()
candidate_track_state = dataState()

bearerStr = str(sys.argv[1])
http_state.setBearer(bearerStr)

# populate playlist query for model
getLikedSongs(http_state, playlist_data_state, 0, numLikedSongs)
getTopTracks(http_state, playlist_data_state, 0, numTopTracks)

playlist_data_state.addFeaturesToData(http_state)
playlist_data_state.addPopularityToData(http_state)

# in case too many songs without features we grab more top tracks
fixBatchSize = 50
counter = 0
while(playlist_data_state.getNumTracks() < 50 and counter < 3):
    getTopTracks(http_state, playlist_data_state, numTopTracks + counter * fixBatchSize, fixBatchSize)
    playlist_data_state.addFeaturesToData(http_state)
    playlist_data_state.addPopularityToData(http_state)
    counter += 1
playlist_data_state.fixLength(50)


# populate candidate track set for model

getRecTracks(http_state, candidate_track_state)
# numYearTracks = 100
# for i in range(0, numYearTracks, yearTrackLimit):
#     getYearTracks(http_state, candidate_track_state, i)

candidate_track_state.addFeaturesToData(http_state)
candidate_track_state.addPopularityToData(http_state)
candidate_track_state.fixLength(100)
candidate_track_state.fixLengthModulo(100)

# turn playlist and candidate data into tensor slices

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
  "id":[]
}

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
  "id":[]
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
    tensor_temp_slices["duration_ms"].append(int(songObj["duration_ms"]))
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
    tensor_slices["id"].append(tensor_temp_slices["id"])
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

for idx, song in enumerate(playlist_data_state.getSongData()):
    if idx == 50:
        break
    addSongDataToSlice(song)
flushPlaylistSlice()

for idx, song in enumerate(candidate_track_state.getSongData()):
    addSongDataToSlice(song)
    tensor_temp_slices["id"].append(song["id"])
    if (idx + 1) % 100 == 0:
        flushTempSlice()

curr_data = {
  "playlistName": tf.reshape(tf.convert_to_tensor(""), (1,)),
  "play_artist_name": tf.reshape(tf.convert_to_tensor(tensor_slices["play_artist_name"][0] ), (1, -1)),
  "play_danceability": tf.reshape(tf.convert_to_tensor( tensor_slices["play_danceability"][0] ), (1, -1)),
  "play_energy": tf.reshape(tf.convert_to_tensor( tensor_slices["play_energy"][0] ), (1, -1)),
  "play_key": tf.reshape(tf.convert_to_tensor( tensor_slices["play_key"][0] ), (1, -1)),
  "play_loudness": tf.reshape(tf.convert_to_tensor( tensor_slices["play_loudness"][0] ), (1, -1)),
  "play_mode": tf.reshape(tf.convert_to_tensor( tensor_slices["play_mode"][0] ), (1, -1)),
  "play_speechiness": tf.reshape(tf.convert_to_tensor( tensor_slices["play_speechiness"][0] ), (1, -1)),
  "play_acousticness": tf.reshape(tf.convert_to_tensor( tensor_slices["play_acousticness"][0] ), (1, -1)),
  "play_instrumentalness": tf.reshape(tf.convert_to_tensor( tensor_slices["play_instrumentalness"][0] ), (1, -1)),
  "play_liveness": tf.reshape(tf.convert_to_tensor( tensor_slices["play_liveness"][0] ), (1, -1)),
  "play_valence": tf.reshape(tf.convert_to_tensor( tensor_slices["play_valence"][0] ), (1, -1)),
  "play_tempo": tf.reshape(tf.convert_to_tensor( tensor_slices["play_tempo"][0] ), (1, -1)),
  "play_duration_ms": tf.reshape(tf.convert_to_tensor( tensor_slices["play_duration_ms"][0] ), (1, -1)),
  "play_time_signature": tf.reshape(tf.convert_to_tensor( tensor_slices["play_time_signature"][0] ), (1, -1)),
  "play_release_date": tf.reshape(tf.convert_to_tensor( tensor_slices["play_release_date"][0] ), (1, -1)),
  "play_popularity": tf.reshape(tf.convert_to_tensor( tensor_slices["play_popularity"][0] ), (1, -1)),
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
  "popularity":[]
}

# run model

model = "../full-model"

loaded = tf.saved_model.load(model)

results = []
for runNum in range(0,len(tensor_slices["popularity"])):
    for key in tensor_temp_slices:
        if key == "id":
            continue
        curr_data[key] = tf.reshape(tf.convert_to_tensor( tensor_slices[key][runNum] ), (1, -1))
    results.append(loaded(curr_data).numpy().flatten().tolist())

# sort and merge model output

idResults = []
for runNum in range(0,len(tensor_slices["popularity"])):
    currResults = []
    for idx, score in enumerate(results[runNum]):
        currResults.append({'id' : tensor_slices['id'][runNum][idx], 'score' : score })
    currResults.sort(key=lambda x : x['score'], reverse=True)
    idResults.append(currResults)

returnList = []
for num in range(0, len(idResults) * 100):
    currMaxindex = -1
    for currListIndex, sortedList in enumerate(idResults):
        if len(sortedList) == 0:
            continue
        if currMaxindex == -1 or sortedList[0]["score"] > idResults[currMaxindex][0]["score"]:
            currMaxindex = currListIndex
    poppedId = idResults[currMaxindex].pop(0)['id']
    if len(returnList) == 0 or poppedId != returnList[-1]:
        returnList.append(poppedId)

# remove songs already saved by the user
songsNotSaved = []
for i in range(0, len(returnList), 50):
    endpoint = 'https://api.spotify.com/v1/me/tracks/contains'
    params = {
        'ids' : ','.join(returnList[i:i+50])
    }
    for index, inLib in enumerate(handleRequest(params, endpoint, http_state, "getContains")):
        if not inLib:
            songsNotSaved.append(returnList[i + index])
    if len(songsNotSaved) > 50:
        break


# return top N songs to browser
print(songsNotSaved[:50])

