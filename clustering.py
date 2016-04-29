# Computes chi-2 distance between a set of features
import numpy as np
from sklearn.cluster import	KMeans
import csv
from sklearn import svm

# Load data
data_dir = "./data/"

# Get keypoints 
keypoints = {}
with open(data_dir+"keypoints.txt") as csvfile:
	reader = csv.reader(csvfile)
	for img,keypt in reader:
		img = int(img)
		keypt = np.array(keypt.strip().split(' '))
		keypt = keypt.astype(float)
		if img in keypoints.keys():
			keypoints[img].append(keypt)
		else:
			keypoints[img] = [keypt]

#labels = {}
labels = []
with open(data_dir+"label.txt") as csvfile:
	reader = csv.reader(csvfile)
	for img,label in reader:
		if "FE" in label:
			label = 1
		else:
			label = -1
		#labels[int(img)] = label
		labels.append(label)
#labels = np.array(labels)

# Get keypoints
# num_keypoints = 100
# SIFT_keypoints = np.random.randn(128, num_keypoints)

# Split into training and test set
train = {key:value for key,value in keypoints.items() if key<0.8*len(keypoints.keys())}
train_labels = labels[:int(0.8*len(keypoints.keys()))]
test = {key:value for key,value in keypoints.items() if key>=0.8*len(keypoints.keys())}
test_labels = labels[int(0.8*len(keypoints.keys())): len(keypoints.keys())]

# Cluster SIFT features
features = []
for key in train.keys():
	for item in train[key]:
		features.append(item)

# K-means
num_clusters = [100]
model = KMeans()
for num_cl in num_clusters:
	model = KMeans(n_clusters=num_cl)
	kmeans = model.fit(features)

opt_num_cl = 1000

# Build histograms 
repr_train = {}

for key in train.keys():
	pred = model.predict(train[key])
	histogram, _ = np.histogram(pred, bins=range(opt_num_cl))
	repr_train[key] = histogram 

repr_test = {}

for key in test.keys():
	pred = model.predict(test[key])
	histogram, _ = np.histogram(pred, bins=range(opt_num_cl))
	repr_test[key] = histogram 

# Training data gender
# target_genders = np.random.randn(num_keypoints) < 0
# target_genders = target_genders.astype(int)

# SVM 
clf = svm.SVC()
clf.fit(repr_train.values(), train_labels) 
clf.score(repr_test.values(), test_labels)