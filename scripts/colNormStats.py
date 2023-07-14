import csv
import os

# 'danceability':     _float_feature(float(row[0].strip())),
# 'energy':           _float_feature(float(row[1].strip())),
# 'key':              _int64_feature(int(row[2].strip())),
# 'loudness':         _float_feature(float(row[3].strip())),
# 'mode':             _int64_feature(int(row[4].strip())),
# 'speechiness':      _float_feature(float(row[5].strip())),
# 'acousticness':     _float_feature(float(row[6].strip())), 
# 'instrumentalness': _float_feature(float(row[7].strip())),
# 'liveness':         _float_feature(float(row[8].strip())),
# 'valence':          _float_feature(float(row[9].strip())),
# 'tempo':            _float_feature(float(row[10].strip())),
# 'song_id':          _bytes_feature(row[11].strip().encode('utf-8')), 
# 'duration_ms':      _int64_feature(int(row[12].strip())),
# 'time_signature':   _int64_feature(int(row[13].strip())),
# 'artist_name':      _bytes_feature(row[14].strip().encode('utf-8')),
# 'album_name':       _bytes_feature(row[15].strip().encode('utf-8')),
# 'release_date':     _int64_feature(int(row[16].strip()[:4])),
# 'duration_ms':      _int64_feature(int(row[17].strip())), 
# 'song_name':        _bytes_feature(row[18].strip().encode('utf-8')),
# 'popularity':       _int64_feature(int(row[19].strip()))


numRecords = 2261593

danceMean = 0
energyMean = 0
keyMean = 0
loudnessMean = 0
modeMean = 0
speechinessMean = 0
acousticnessMean = 0
livenessMean = 0
valenceMean = 0
tempoMean = 0
durationMean = 0
signatureMean = 0
dateMean = 0
instrumentalnessMean = 0
popMean = 0

danceVariance = 0
energyVariance = 0
keyVariance = 0
loudnessVariance = 0
modeVariance = 0
speechinessVariance = 0
acousticnessVariance = 0
livenessVariance = 0
valenceVariance = 0
tempoVariance = 0
durationVariance = 0
signatureVariance = 0
dateVariance = 0
instrumentalnessVariance = 0
popVariance = 0


with open('songData/fixed.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for idx, row in enumerate(reader):
        danceMean += float(row[0].strip())
        energyMean += float(row[1].strip())
        keyMean += int(row[2].strip())
        loudnessMean += float(row[3].strip())
        modeMean += int(row[4].strip())
        speechinessMean += float(row[5].strip())
        acousticnessMean += float(row[6].strip())
        livenessMean += float(row[8].strip())
        valenceMean += float(row[9].strip())
        tempoMean += float(row[10].strip())
        durationMean += int(row[12].strip())
        signatureMean += int(row[13].strip())
        dateMean += int(row[16].strip()[:4])
        instrumentalnessMean += float(row[7].strip())
        popMean += int(row[19].strip())

    
    danceMean = float(danceMean / numRecords)
    energyMean = float(energyMean / numRecords)
    keyMean = float(keyMean / numRecords)
    loudnessMean = float(loudnessMean / numRecords)
    modeMean = float(modeMean / numRecords)
    speechinessMean = float(speechinessMean / numRecords)
    acousticnessMean = float(acousticnessMean / numRecords)
    livenessMean = float(livenessMean / numRecords)
    valenceMean = float(valenceMean / numRecords)
    tempoMean = float(tempoMean / numRecords)
    durationMean = float(durationMean / numRecords)
    signatureMean = float(signatureMean / numRecords)
    dateMean = float(dateMean / numRecords)
    instrumentalnessMean = float(instrumentalnessMean / numRecords)
    popMean = float(popMean / numRecords)

    print("danceMean: " + str(danceMean))
    print("energyMean: " + str(energyMean))
    print("keyMean: " + str(keyMean))
    print("loudnessMean: " + str(loudnessMean))
    print("modeMean: " + str(modeMean))
    print("speechinessMean: " + str(speechinessMean))
    print("acousticnessMean: " + str(acousticnessMean))
    print("livenessMean: " + str(livenessMean))
    print("valenceMean: " + str(valenceMean))
    print("tempoMean: " + str(tempoMean))
    print("durationMean: " + str(durationMean))
    print("signatureMean: " + str(signatureMean))
    print("dateMean: " + str(dateMean))
    print("instrumentalnessMean: " + str(instrumentalnessMean))
    print("popMean: " + str(popMean))

with open('songData/fixed.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for idx, row in enumerate(reader):
        danceVariance += (float(row[0].strip()) - danceMean) * (float(row[0].strip()) - danceMean)
        energyVariance += (float(row[1].strip()) - energyMean) * (float(row[1].strip()) - energyMean)
        keyVariance += (int(row[2].strip()) - keyMean) * (int(row[2].strip()) - keyMean)
        loudnessVariance += (float(row[3].strip()) - loudnessMean) * (float(row[3].strip()) - loudnessMean)
        modeVariance += (int(row[4].strip()) - modeMean) * (int(row[4].strip()) - modeMean)
        speechinessVariance += (float(row[5].strip()) - speechinessMean) * (float(row[5].strip()) - speechinessMean)
        acousticnessVariance += (float(row[6].strip()) - acousticnessMean) * (float(row[6].strip()) - acousticnessMean)
        livenessVariance += (float(row[8].strip()) - livenessMean) * (float(row[8].strip()) - livenessMean)
        valenceVariance += (float(row[9].strip()) - valenceMean) * (float(row[9].strip()) - valenceMean)
        tempoVariance += (float(row[10].strip()) - tempoMean) * (float(row[10].strip()) - tempoMean)
        durationVariance += (int(row[12].strip()) - durationMean) * (int(row[12].strip()) - durationMean)
        signatureVariance += (int(row[13].strip()) - signatureMean) * (int(row[13].strip()) - signatureMean)
        dateVariance += (int(row[16].strip()[:4]) - dateMean) * (int(row[16].strip()[:4]) - dateMean)
        instrumentalnessVariance += (float(row[7].strip()) - instrumentalnessMean) * (float(row[7].strip()) - instrumentalnessMean)
        popVariance += (int(row[19].strip()) - popMean) * (int(row[19].strip()) - popMean)

    danceVariance = float(danceVariance / numRecords)
    energyVariance = float(energyVariance / numRecords)
    keyVariance = float(keyVariance / numRecords)
    loudnessVariance = float(loudnessVariance / numRecords)
    modeVariance = float(modeVariance / numRecords)
    speechinessVariance = float(speechinessVariance / numRecords)
    acousticnessVariance = float(acousticnessVariance / numRecords)
    livenessVariance = float(livenessVariance / numRecords)
    valenceVariance = float(valenceVariance / numRecords)
    tempoVariance = float(tempoVariance / numRecords)
    durationVariance = float(durationVariance / numRecords)
    signatureVariance = float(signatureVariance / numRecords)
    dateVariance = float(dateVariance / numRecords)
    instrumentalnessVariance = float(instrumentalnessVariance / numRecords)
    popVariance = float(popVariance / numRecords)

    print("danceVariance: " + str(danceVariance))
    print("energyVariance: " + str(energyVariance))
    print("keyVariance: " + str(keyVariance))
    print("loudnessVariance: " + str(loudnessVariance))
    print("modeVariance: " + str(modeVariance))
    print("speechinessVariance: " + str(speechinessVariance))
    print("acousticnessVariance: " + str(acousticnessVariance))
    print("livenessVariance: " + str(livenessVariance))
    print("valenceVariance: " + str(valenceVariance))
    print("tempoVariance: " + str(tempoVariance))
    print("durationVariance: " + str(durationVariance))
    print("signatureVariance: " + str(signatureVariance))
    print("dateVariance: " + str(dateVariance))
    print("instrumentalnessVariance: " + str(instrumentalnessVariance))
    print("popVariance: " + str(popVariance))