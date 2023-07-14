import csv
import os

parentDir = "songData"

firstchars = {}
secondchars = {}
thirdchars = {}
fourthchars = {}
with open('songData/sortedUris') as readFile:
    lines = readFile.readlines()
    for idx, line in enumerate(lines):
        if len(line) < 4:
            print("THIS SHOULDNT HAPPEN")
            quit()
        if line[0] not in firstchars:
            firstchars[line[0]] = 1
        else:
            firstchars[line[0]] += 1

        if line[1] not in secondchars:
            secondchars[line[1]] = 1
        else:
            secondchars[line[1]] += 1

        if line[2] not in thirdchars:
            thirdchars[line[2]] = 1
        else:
            thirdchars[line[2]] += 1

        if line[3] not in fourthchars:
            fourthchars[line[3]] = 1
        else:
            fourthchars[line[3]] += 1

        # if idx % 1000 == 0:
        #     print(idx)

# print("first")
# print(len(firstchars.keys()))
# print(firstchars)
# print("secondchars")
# print(len(secondchars.keys()))
# print(secondchars)
# print("thirdchars")
# print(len(thirdchars.keys()))
# print(thirdchars)
# print("fourthchars")
# print(len(fourthchars.keys()))
# print(fourthchars)

for key1 in firstchars.keys():
    print(key1)
    if key1.isupper():
        key1 = key1 + "_"
    path = os.path.join(parentDir, key1)
    newpath = os.path.join(parentDir, "songSplit",key1)
    os.mkdir(newpath)
    for key2 in secondchars.keys():
        if key2.isupper():
            key2 = key2 + "_"
        path2 = os.path.join(path, key2)
        newpath2 = os.path.join(newpath, key2)
        os.mkdir(newpath2)
        for key3 in thirdchars.keys():
            if key3.isupper():
                key3 = key3 + "_"
            path3 = os.path.join(path2, key3)
            newpath3 = os.path.join(newpath2, key3)
            f = open(path3)
            f2 = open(newpath3, "w")
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if not row[16].strip()[:4].isdigit():
                    temp = row[18]
                    row[18] = row[16]
                    row[16] = temp
                f2.write(','.join(row)+"\n")
            f.close()
            f2.close()


# with open('songData/fixed.csv') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     # writer = csv.writer(writeFile, delimiter=',')
#     for idx, row in enumerate(reader):
#         line = row[11].strip()

#         d1 = line[0] + "_" if line[0].isupper() else line[0]
#         d2 = line[1] + "_" if line[1].isupper() else line[1]
#         d3 = line[2] + "_" if line[2].isupper() else line[2]
        
#         path = os.path.join(parentDir, d1, d2, d3)

#         f = open(path, "a")
#         f.write(','.join(row)+"\n")
#         f.close()

#         if idx % 10000 == 0:
#             print(idx)