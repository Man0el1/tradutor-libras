import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dictionary = pickle.load(open('./data.pickle', 'rb'))

data_clean = []
labels_clean = []

for sample, label in zip(data_dictionary['data'], data_dictionary['labels']):
  if len(sample) == 42:
    data_clean.append(sample)
    labels_clean.append(label)

data = np.asarray(data_clean)
labels = np.asarray(labels_clean)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% correct!'.format(score * 100))

f = open('model.pickle', 'wb')
pickle.dump({'model': model}, f)
f.close()