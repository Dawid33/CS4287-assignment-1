import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from numpy import mean, std
import csv
import pandas as pd

class Dataset:
    def __init__(self):
        # Import the dataset from a csv file and create a new column
        # where each row signifies whether it did or did not rain on each day.
        # This column will be the value our model is trying to predict
        # (whether it did or did not rain on a particular day.)
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
        self.raw = pd.DataFrame(data)
        self.X = pd.DataFrame(data)[self.cols]
        self.normalize()

    def print(self):
        print(self.X)
        print(self.y[0:5])

    def normalize(self):            
        # Normalize all the predictors to be between 0 and 1
        scaler = MinMaxScaler()
        self.X[self.cols] = scaler.fit_transform(self.X[self.cols])

data = Dataset()

# Split the dataset into two parts, one part for training and the other for validating / evaluating the model.
X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, stratify=data.y, random_state=1)

# Optimiser is 'adam'
clf = MLPClassifier(random_state=1, solver='adam', max_iter=3000, hidden_layer_sizes=[100,]).fit(X_train, y_train)
prob = clf.predict_proba(X_test[:1])
predict = clf.predict(X_test)
score = clf.score(X_test, y_test)

print(f"Probability {prob}")
print(f"Predictions {predict}")
print(f"Score {score}")

print(f"\nCross Fold Validation")
cv = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(clf, X_test, predict, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# Draw Graph
import matplotlib.pyplot as plt
data.raw[['rain', 'sun']] = data.raw[['rain', 'sun']].apply(pd.to_numeric)
sun = data.raw['sun'].values.reshape(-1,1)
rain = data.raw['rain'].values.reshape(-1,1)

reg = LinearRegression().fit(sun, rain)

plt.plot(data.raw[['sun']], data.raw[['rain']], 'o')
plt.plot(data.raw['sun'], reg.predict(sun),color="red",linewidth=3)
plt.xlabel('sunlight per day')
plt.ylabel('rain per day')
plt.show()

