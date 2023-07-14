import json
import os
import math
import csv
import threading

directory = 'playlistData'

def songDataJson(row, track_name):
    features = {
        'danceability':     row[0].strip(),
        'energy':           row[1].strip(),
        'key':              row[2].strip(),
        'loudness':         row[3].strip(),
        'mode':             row[4].strip(),
        'speechiness':      row[5].strip(),
        'acousticness':     row[6].strip(), 
        'instrumentalness': row[7].strip(),
        'liveness':         row[8].strip(),
        'valence':          row[9].strip(),
        'tempo':            row[10].strip(),
        'song_id':          row[11].strip(), 
        'duration_ms':      row[12].strip(),
        'time_signature':   row[13].strip(),
        'artist_name':      row[14].strip(),
        'album_name':       row[15].strip(),
        'release_date':     row[16].strip()[:4],
        'duration_ms':      row[17].strip(), 
        'song_name':        track_name,
        'popularity':       row[19].strip() 
    }
    return features

def getRow(songid, track_name):
    #strip id here
    line = songid.strip()[14:]
    d1 = line[0] + "_" if line[0].isupper() else line[0]
    d2 = line[1] + "_" if line[1].isupper() else line[1]
    d3 = line[2] + "_" if line[2].isupper() else line[2]
        
    path = os.path.join('songData', 'songSplit', d1, d2, d3)
    f = open(path)
    reader = csv.reader(f, delimiter=',')
    # writer = csv.writer(writeFile, delimiter=',')
    for row in reader:
        if row[11].strip() == line:
            f.close()
            return songDataJson(row, track_name)
    print(line)
    print("SHOULDNT REACH THIS")
    quit()

dirs = os.listdir(directory)
# set1 = dirs[0:200]
# set2 = dirs[200:400]
# set3 = dirs[400:600]
# set4 = dirs[600:800]
# set5 = dirs[800:1000]

# class myThread (threading.Thread):
#    def __init__(self, filesToProcess, identifier):
#         threading.Thread.__init__(self)
#         self.identifier = identifier
#         self.filesToProcess = filesToProcess
#    def run(self):
#         print(self.identifier)
#         i = 0
#         for filename in self.filesToProcess:
#             currFilename = os.path.join(directory, filename)
#             writeFile = open(os.path.join('playlistSongData', filename), 'w')
#             if os.path.isfile(currFilename):
#                 currFile = open(currFilename)
#                 dictionary = json.load(currFile)
#                 for playlist in dictionary["playlists"]:
#                     songs = []
#                     for idx, track in enumerate(playlist["tracks"]):
#                         songs.append(getRow(track["track_uri"],track["track_name"]))
#                     playlist["tracks"] = songs
#                 json.dump(dictionary, writeFile)
#                 currFile.close()
#             writeFile.close()
#             i += 1
#             if i % 10 == 0:
#                 print(str(self.identifier)+", "+str(i))


    

# thread1 = myThread(set1, 1)
# thread2 = myThread(set2, 2)
# thread3 = myThread(set3, 3)
# thread4 = myThread(set4, 4)
# thread5 = myThread(set5, 5)

# thread1.start()
# thread2.start()
# thread3.start()
# thread4.start()
# thread5.start()
# thread1.join()
# thread2.join()
# thread3.join()
# thread4.join()
# thread5.join()

i = 0
for filename in dirs:
    currFilename = os.path.join(directory, filename)
    writeFile = open(os.path.join('playlistSongData', filename), 'w')
    if os.path.isfile(currFilename):
        currFile = open(currFilename)
        dictionary = json.load(currFile)
        for playlist in dictionary["playlists"]:
            songs = []
            for idx, track in enumerate(playlist["tracks"]):
                songs.append(getRow(track["track_uri"],track["track_name"]))
            playlist["tracks"] = songs
        json.dump(dictionary, writeFile)
        currFile.close()
    writeFile.close()
    i += 1
    if i % 10 == 0:
        print(i)