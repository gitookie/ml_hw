from utils.load_data import *
from utils.process import *
import cv2 as cv
import numpy as np
from utils.models import DecisionStump, Adaboost
import pickle

"""pos_samples, pos_integral = load_and_cut(dir_path='Caltech_WebFaces', txt_path='WebFaces_GroundThruth.txt')
pos_y = np.ones(pos_samples.shape[0])
neg_samples, neg_integral = get_negative_samples(dir_path='Caltech_WebFaces', txt_path='WebFaces_GroundThruth.txt')
neg_y = np.full(neg_samples.shape[0], -1)
samples = np.concatenate((pos_samples, neg_samples), axis = 0)
integrals = np.concatenate((pos_integral, neg_integral), axis=0)"""


# test 1
"""print('samples shape:', samples.shape)
y = np.concatenate((pos_y, neg_y), axis = 0)
print('y shape:', y.shape)
samples, y, integrals = shuffle_simultaneously(samples, y, integrals)
print(y)
"""

"""y = np.concatenate((pos_y, neg_y), axis = 0)
print('y shape:', y.shape)

samples, y, integrals = shuffle_simultaneously(samples, y, integrals)
np.save('samples.npy', samples)
np.save('y.npy', y)
np.save('integrals.npy', integrals)"""
# haar = np.load('haar.npy')
"""haar = []
for i in range(len(samples)):
    if i % 5 == 0:
        print(f'getting haar characteristic:{i + 1}/{len(samples)}')
    haar_i = get_haar_horizontal(samples[i], integrals[i])
    haar.append(haar_i)
    # print(len(haar_i))
    # print(haar_i)
haar = np.stack(haar, axis=0)
np.save('haar.npy', haar)"""


haar1 = np.load('haar.npy')
haar2 = np.load('haar_vertical.npy')
haar3 = np.load('haar_centered.npy')
haar = np.concatenate((haar1, haar2, haar3), axis = 1)
# haar = haar2
print(haar.shape)
# print('haar characteristics have been saved')
y = np.load('y.npy')

"""with open('my_classifier2.pkl', 'rb') as clf:
    classifier = pickle.load(clf)"""
train_samples = 1200
test_samples = 100
classifier = Adaboost()
classifier.fit(haar[:train_samples, :], y[:train_samples])
with open('my_classifier4.pkl', "wb") as clf:
    pickle.dump(classifier, clf)
print('classifier has been saved')
y_pred = classifier.predict(haar[train_samples:train_samples + test_samples, :])
# print(y_pred == y[train_samples:])
result = np.zeros(test_samples)
result[y_pred == y[train_samples:train_samples + test_samples]] = 1
accuracy = float(np.sum(result) / len(result))
print(accuracy)


"""samples = np.load('samples.npy')
integrals = np.load('integrals.npy')
haar3 = []
for i in range(len(samples)):
    if i % 5 == 0:
        print(f'getting haar characteristic:{i + 1}/{len(samples)}')
    haar_i = get_haar_centered(samples[i], integrals[i])
    haar3.append(haar_i)
    # print(len(haar_i))
    # print(haar_i)

np.save('haar_centered.npy', haar3)"""


"""
haar2 = np.load('haar_vertical.npy')
print(haar2.shape)
haar = np.load('haar.npy')
print(haar.shape)
haar3 = np.load('haar_centered.npy')
print(haar3.shape)
"""