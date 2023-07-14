import json
import os
import math

directory = 'playlistSongData'
histogramData = [0, 0, 0, 0, 4, 5410, 6234, 7174, 8186, 9010, 10660, 10765, 12014, 12321, 13042, 14184, 13853, 13684, 13631, 13535, 15053, 13874, 13604, 13154, 13247, 13026, 12839, 12504, 12512, 12329, 13076, 11899, 11788, 11625, 11373, 11175, 10920, 10553, 10454, 10272, 10980, 9885, 10060, 9783, 9555, 9274, 9011, 8843, 8694, 8505, 8891, 8300, 8059, 7968, 7852, 7672, 7442, 7276, 7117, 7191, 7013, 6804, 6735, 6593, 6448, 6344, 6184, 6006, 5807, 5838, 5781, 5709, 5540, 5287, 5307, 5258, 5012, 5004, 4885, 4734, 4833, 4847, 4624, 4582, 4539, 4384, 4311, 4206, 3997, 3932, 4047, 3932, 3815, 3847, 3723, 3779, 3560, 3471, 3504, 3505, 4470, 3616, 3503, 3402, 3261, 3271, 3163, 2996, 3122, 2987, 2969, 2932, 2863, 2930, 2643, 2600, 2705, 2668, 2614, 2619, 2481, 2574, 2478, 2469, 2395, 2328, 2349, 2374, 2222, 2111, 2229, 2183, 2117, 2057, 2060, 2058, 2082, 1986, 1997, 1879, 1936, 1907, 1896, 1802, 1765, 1766, 1705, 1689, 1737, 1720, 1734, 1707, 1582, 1567, 1627, 1586, 1570, 1503, 1471, 1473, 1515, 1438, 1418, 1395, 1396, 1412, 1328, 1362, 1357, 1292, 1343, 1307, 1258, 1247, 1237, 1237, 1179, 1189, 1206, 1206, 1134, 1168, 1131, 1134, 1095, 1064, 1036, 1009, 1018, 988, 992, 979, 1032, 969, 944, 898, 938, 907, 839, 940, 978, 917, 917, 932, 869, 849, 874, 865, 836, 854, 859, 839, 794, 790, 804, 784, 727, 790, 715, 701, 734, 702, 730, 714, 681, 657, 616, 712, 685, 667, 672, 639, 637, 624, 596, 606, 610, 581, 600, 566, 533, 584, 502, 513, 452, 482, 457, 446, 446, 380, 369, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
# i = 0
# histlen = 600
# histogramData = [0] * histlen
# for filename in os.listdir(directory):
#     currFilename = os.path.join(directory, filename)
#     if os.path.isfile(currFilename):
#         currFile = open(currFilename)
#         dictionary = json.load(currFile)
#         for playlist in dictionary["playlists"]:
#             histogramData[len(playlist["tracks"])] += 1
#     i += 1
#     if i % 100 == 0:
#         print(i)

lastindex = 0
for ide, val in enumerate(reversed(histogramData)):
    if val > 0:
        lastindex = len(histogramData) - ide
        break

i = 0
numLessThan100 = 0
arrSum = 0
weightedSum = 0
maxVal = 0
maxIndex = 0
for val in histogramData:
    if i <= 100:
        numLessThan100 += val
    arrSum += val
    weightedSum += val * i
    if val > maxVal:
        maxVal = val
        maxIndex = i
    i = i + 1

print(numLessThan100)

i = 0
runningSum = 0
median = 0
for val in histogramData:
    runningSum += val
    i = i + 1
    if runningSum >= (arrSum/2):
        median = i
        break

mean = weightedSum/arrSum
i = 0
runningSum = 0
for val in histogramData:
    runningSum += (i - mean) * (i - mean) * val
    i = i + 1

print(histogramData)
print("Max: "+str(lastindex-1))
print("Min: 5")
print("Mean: "+str(mean))
print("Median: "+str(median))
print("Mode: "+str(maxIndex))
print("stddev: "+str(math.sqrt(runningSum/arrSum)))
print("sum of playlist length: "+str(weightedSum))