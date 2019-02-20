import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
train = pd.read_csv('/Users/avibanerjee/Documents/Development/Practice/Titanic/titanic/train.csv')
test = pd.read_csv('/Users/avibanerjee/Documents/Development/Practice/Titanic/titanic/test.csv')
survive_data = pd.read_csv('/Users/avibanerjee/Documents/Development/Practice/Titanic/titanic/gender_submission.csv')
from sklearn import metrics

#mappings for strings to digits
gender = {'male': 0, 'female': 1}
embarked = {'C' : 0, 'Q' : 1, 'S' : 2}

survived = train['Survived']
survived = survived.fillna(survived.mean())

train = train.fillna(train.mean())
features = train.loc[:,['Sex']].replace({'Sex' : gender})
features = features.fillna(features.mean())

#prepare test data
test = test.fillna(test.mean())

# print(test)
final = test.loc[:,['Sex']].replace({'Sex' : gender})
print(features.shape, survived.shape)

#create SVM
model = svm.SVC(kernel = 'linear')
print(survived)
#train
model.fit(features, survived)

score = model.score(final, survive_data['Survived'])
print(score)
print('Accuracy: ', metrics.accuracy_score(test, score))

