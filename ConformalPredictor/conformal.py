import numpy as np

# Algorithm

# 1 Calculate the nonconformity score
#   under each hypotesys

def nearest_neighbor_distances(vector, label, value_index, h):
	'''Calculate the NN distances using a single vector and value, h as the
	hypotesys component.
	'''
	# All the distances from `value`
	distances = abs(vector[value_index] - vector)
	d1 = [] # numerator
	d2 = [] # denominator
	for i, v in enumerate(zip(vector, label)):
		if i == value_index:
			continue
		if v[1] == h:
			d1.append(distances[i])
		else:
			d2.append(distances[i])
	
	if np.min(d1) == 0 and np.min(d2) == 0:
		return 0.
	elif np.min(d2) == 0:
		return np.inf
	else:
		return np.min(d1)/np.min(d2)

		

# Load Data
sepal_length, labels = np.loadtxt('dataset.csv', dtype='float,object', 
	delimiter=',', usecols=(0, 1), unpack=True, skiprows=1)

# FOR A NEW NUMBER = 6.8 AS A SETOSA:
sepal_length_2 = np.append(sepal_length, 6.8)
labels_2 = np.append(labels, 's')

H_setosa = []
for index, value in enumerate(sepal_length_2):
	value = nearest_neighbor_distances(sepal_length_2,
		labels_2, index, labels_2[index])
	
	print(f"z_{index+1} = {round(value, 2)}")
	H_setosa.append(value)
H_setosa = np.array(H_setosa)

# FOR A NEW NUMBER = 6.8 AS A VERSICOLOR:
labels_2 = np.append(labels, 'v')

H_versicolor = []
for index, value in enumerate(sepal_length_2):
	value = nearest_neighbor_distances(sepal_length_2,
		labels_2, index, labels_2[index])
	print(f"z_{index+1} = {round(value, 2)}")
	H_versicolor.append(value)
H_versicolor = np.array(H_versicolor)

# PROBABILITY
## SETOSA
x = len(H_versicolor[H_versicolor >= H_versicolor[-1]])/len(sepal_length_2)
print("p(ponto_novo, versicolor) = ", x)

x = len(H_setosa[H_setosa >= H_setosa[-1]])/len(sepal_length_2)
print("p(ponto_novo, setosa) = ", x)
