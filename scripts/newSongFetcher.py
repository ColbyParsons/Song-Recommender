import requests
import base64
import json

class httpState:
    def __init__(self):
        self.authTokens = [] # Auth tokens removed
        self.currentTokenIndex = 0
        self.currentBearer = self.getNewBearer()

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

featureBatchSize = 75
tracksBatchSize = 50

class dataState:
    def __init__(self):
        self.songData = []
        self.uris = []
        self.cleanup = []

    def addItem(self, uri, artistName, relDate):
        uriCleaned = uri[14:]
        self.songData.append({ # empty fields get populated later in addFeaturesToData
            "id": uriCleaned,
            "artist_name": artistName, # populate from album
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
            "release_date": relDate, # populate from album
            "popularity": None, # populate from song data
            "rating" : None
        })
        self.uris.append(uriCleaned)

    def getTracksAudioFeatures(self, startIndex, state):
        endpoint = 'https://api.spotify.com/v1/audio-features'
        params = {
            'ids' : ','.join(self.uris[startIndex:startIndex+featureBatchSize])
        }
        num429s = 0
        max429s = state.getNumAuthTokens() + 1
        while num429s < max429s:
            headers = { 'Authorization': 'Bearer ' + state.getBearer(), 'Content-Type' : 'application/json' }
            r = requests.get(endpoint, headers=headers, params=params)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 401:
                state.updateBearer()
            elif r.status_code == 429:
                state.switchAuthToken(self)
                state.updateBearer()
                num429s += 1
            else:
                print(params)
                print(r.status_code)
                print("SOMETHING WENT VERY WRONG: getTracksAudioFeatures")
                quit()
        print("ALL AUTHS TO MANY REQUESTS")
        quit()
    
    def getTracks(self, startIndex, state):
        endpoint = 'https://api.spotify.com/v1/tracks'
        params = {
            'ids' : ','.join(self.uris[startIndex:startIndex+tracksBatchSize])
        }
        num429s = 0
        max429s = state.getNumAuthTokens() + 1
        while num429s < max429s:
            headers = { 'Authorization': 'Bearer ' + state.getBearer(), 'Content-Type' : 'application/json' }
            r = requests.get(endpoint, headers=headers, params=params)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 401:
                state.updateBearer()
            elif r.status_code == 429:
                state.switchAuthToken(self)
                state.updateBearer()
                num429s += 1
            else:
                print(params)
                print(r.status_code)
                print("SOMETHING WENT VERY WRONG: getTracks")
                quit()
        print("ALL AUTHS TO MANY REQUESTS")
        quit()

    def addFeaturesToData(self, req_state):
        for uriIndex in range(0, len(self.uris), tracksBatchSize):
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

    def addPopularityToData(self, req_state):
        for uriIndex in range(0, len(self.uris), featureBatchSize):
            trackJSON = self.getTracks(uriIndex, req_state)
            for songIndex, songFeatures in enumerate(trackJSON['tracks']):
                self.songData[uriIndex + songIndex]['popularity'] = songFeatures['popularity']

    def writeSongData(self, filename):
        for toClean in reversed(self.cleanup):
            self.songData.pop(toClean)
            self.uris.pop(toClean)
        self.cleanup = []
        writeFile = open(filename, 'w')
        json.dump(self.songData, writeFile)
        writeFile.close()

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
    num429s = 0
    max429s = state.getNumAuthTokens() + 1
    while num429s < max429s:
        headers = { 'Authorization': 'Bearer ' + state.getBearer(), 'Content-Type' : 'application/json' }
        r = requests.get(endpoint, headers=headers, params=params)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 401:
            state.updateBearer()
        elif r.status_code == 429:
            state.switchAuthToken(self)
            state.updateBearer()
            num429s += 1
        else:
            print(params)
            print(r.status_code)
            print("SOMETHING WENT VERY WRONG: getNewSongs")
            quit()
    print("ALL AUTHS TO MANY REQUESTS")
    quit()


# adds to list of song data with songs from album
def getSongUrisFromAlbum(albumId, artistName, relDate, state, song_state):
    endpoint = 'https://api.spotify.com/v1/albums/' + str(albumId) + '/tracks'
    params = {
        'limit' : '50',
        'offset' : '0'
    }
    num429s = 0
    max429s = state.getNumAuthTokens() + 1
    while num429s < max429s:
        headers = { 'Authorization': 'Bearer ' + state.getBearer(), 'Content-Type' : 'application/json' }
        r = requests.get(endpoint, headers=headers, params=params)
        if r.status_code == 200:
            for item in r.json()['items']:
                song_state.addItem(item['uri'], artistName, relDate)
            return
        elif r.status_code == 401:
            state.updateBearer()
        elif r.status_code == 429:
            state.switchAuthToken(self)
            state.updateBearer()
            num429s += 1
        else:
            print(params)
            print(r.status_code)
            print("SOMETHING WENT VERY WRONG: getSongUrisFromAlbum")
            quit()
    print("ALL AUTHS TO MANY REQUESTS")
    quit()

# gets song uris from /search api album type return
def getSongUris(albumJson, state, song_state):
    for item in albumJson['albums']['items']:
        getSongUrisFromAlbum(item['id'], item['artists'][0]['name'], item['release_date'], state, song_state)


http_state = httpState()
data_state = dataState()

searchBatchSize = 50

for i in range(0, 1000, searchBatchSize):
    print(i)
    getSongUris(getNewSongs(searchBatchSize, i, http_state), http_state, data_state)

data_state.addFeaturesToData(http_state)
data_state.addPopularityToData(http_state)
data_state.writeSongData('website/last_two_weeks')
