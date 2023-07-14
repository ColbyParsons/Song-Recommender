import csv
import os

f = open("sorted.csv")
reader = csv.reader(f, delimiter=',')
# writer = open("fixed.csv", "w")

failedRows = []
for idx, row in enumerate(reader):
    if len(row) < 20:
        failedRows.append(row[11].strip())
        continue
    # if row[18].strip().isdigit():
    #     temp = row[18]
    #     row[18] = row[17]
    #     row[17] = temp
    # writer.write(','.join(row)+"\n")
    # if idx % 10000 == 0:
    #     print(idx)


writer = open("missingAnal2", "w")
for line in failedRows:
    writer.write(line+'\n')