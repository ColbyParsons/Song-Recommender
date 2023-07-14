import csv


# duplicates = open('toDelete4')
# currDelete = duplicates.readline().strip()

# writeFile = open('noDup4.csv', 'w')

with open('sorted.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
   # writer = csv.writer(writeFile, delimiter=',')
    for row in reader:
        try:
            if(len(row[11].strip()) != 22):
                print("A")
                print(row)
                print("A")
                # print(currDelete)
                # currDelete = duplicates.readline().strip()
        except:
            print(row)
        