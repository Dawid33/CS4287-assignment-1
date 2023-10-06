import csv


class Dataset:
    def __init__(self):
        with open('shannon airport daily data.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            out = []
            for row in reader:
                out.append(row)
            self.data = out

        self.normalize()

    def normalize(self):
        for row in self.data:
            print(row)