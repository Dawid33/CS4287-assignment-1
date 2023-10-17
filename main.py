import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import csv
import pandas as pd

class Dataset:
    def __init__(self):
        with open('shannon airport daily data.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            data= []
            self.y = []
            for row in reader:
                if float(row["rain"]) > 0:
                    self.y.append(True);
                else:
                    self.y.append(False);
                data.append(row)
        self.cols = ['sun', 'mean_cbl_pressure', 'min_air_temp', 'mean_wind_speed']
        self.X = pd.DataFrame(data)[self.cols]
        self.normalize()

    def print(self):
        print(self.X)
        print(self.y[0:5])

    def normalize(self):            
        scaler = MinMaxScaler()
        self.X[self.cols] = scaler.fit_transform(self.X[self.cols])


data = Dataset()
data.print()

X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, stratify=data.y, random_state=1)

clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
prob = clf.predict_proba(X_test[:1])
predict = clf.predict(X_test)
score = clf.score(X_test, y_test)

print(f"Probability {prob}")
print(f"Predictions {predict}")
print(f"Score {score}")

