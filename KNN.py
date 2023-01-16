import from sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KneighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing


#Reading the data

data=pd.read_csv("car.data")

#Since the data is not all numerical, we need to preprocess it to convert it to all numerical

Le=preprocessing.labelencoder()

buying=Le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

#Putting the data into input(x) and output(y) arrays
X = list(zip(buying, maint, door, persons, lug_boot, safety))

y = list(cls)

#Splitting the data into train data and test data

x_train,,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y, test_size=0.1)

#Implementing the model

model=KNeighborsClassifier(n_neighbors=7)

model.fit(x_train,y_train)

#Checking the accuracy

accuracy=model.score(x_test,y_test)
print('accuracy: ,', accuracy)

#Using the model to check the predictions

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)
