import csv

def read_csv(in_file):
    with open(in_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            print(row)
