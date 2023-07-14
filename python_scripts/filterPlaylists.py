import json
import os
import math

directory = 'spotify_million_playlist_dataset/data'
noAnalysisFile = 'songData/missingAnalysis2'


flag = False
i = 0

# currFile = open("playlistData/mpd.slice.109000-109999.json")
# dictionary = json.load(currFile)
# for playlist in dictionary["playlists"]:
#     # print(str(i) + ", " + str(len(playlist["tracks"])))
#     for track in playlist["tracks"]:
#         toDelete = open(noAnalysisFile)
#         for uri in toDelete:
#             if (track["track_uri"] == ("spotify:track:"+uri.strip())):
#                 print("match")
#                 print(track["track_uri"])
#                 print("spotify:track:"+uri.strip())
#     i = i+1

#MAKE DIR WITHOUT MISSING ANALYSIS PLAYLISTS
for filename in os.listdir(directory):
    currFilename = os.path.join(directory, filename)
    writeFile = open(os.path.join('playlistData', filename), 'w')
    if os.path.isfile(currFilename):
        currFile = open(currFilename)
        dictionary = json.load(currFile)
        for playlist in dictionary["playlists"]:
            toRemove = []
            for idx, track in enumerate(playlist["tracks"]):
                toDelete = open(noAnalysisFile)
                for uri in toDelete:
                    if (track["track_uri"] == ("spotify:track:"+uri.strip())):
                        toRemove.append(idx)
            for index in reversed(toRemove):
                playlist["tracks"].pop(index)
        json.dump(dictionary, writeFile)
        currFile.close()
    writeFile.close()
    i += 1
    if i % 10 == 0:
        print(i)


# cehck if not MISSING ANALYSIS PLAYLISTS
# for filename in os.listdir('playlistData2'):
#     currFilename = os.path.join('playlistData2', filename)
# currFilename = "playlistData2/mpd.slice.105000-105999.json"
# if os.path.isfile(currFilename):
#     currFile = open(currFilename)
#     dictionary = json.load(currFile)
#     for playlist in dictionary["playlists"]:
#         for idx, track in enumerate(playlist["tracks"]):
#             toDelete = open(noAnalysisFile)
#             for uri in toDelete:
#                 if (track["track_uri"] == ("spotify:track:"+uri.strip())):
#                     print(track["track_uri"])
#                     quit()
    # i += 1
    # print(i)

# numplaylists = 0
# for filename in os.listdir(directory):
#     currFilename = os.path.join(directory, filename)
#     if os.path.isfile(currFilename):
#         currFile = open(currFilename)
#         dictionary = json.load(currFile)
#         numplaylists += len(dictionary["playlists"])
#     i += 1
#     print(i)

# print(numplaylists) # 999844 left after no-analysis removed