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
cats, counts = np.unique(labels, return_counts=True)
print("Unique values: ", cats)
print("Counts: ", counts)
print("")

def return_appended(length_array, label_array, category, value):
	'''Append the category and the value to a new array.
	'''
	new_length_array = np.append(length_array, value)
	new_label_array = np.append(label_array, category)
	return new_length_array, new_label_array

def return_nn_distances(length_array, label_array):
	'''Process the function of NN for every point,
	returning a array with each NN distance.
	'''
	return_array = []
	for index, value in enumerate(length_array):
		value = nearest_neighbor_distances(length_array,
			label_array, index, label_array[index])
		
		return_array.append(value)
	return_array = np.array(return_array)
	return return_array

# Setosa
v, l = return_appended(sepal_length,
	labels,
	's',
	6.8)
H_setosa = return_nn_distances(v, l)

# Versicolor
v, l = return_appended(sepal_length,
	labels,
	'v',
	6.8)
H_versicolor = return_nn_distances(v, l)

print("Index Setosa Versicolor")
print(np.vstack((np.array(range(len(H_setosa))) + 1, H_setosa, H_versicolor)).T.round(2))

print("")

# PROBABILITY
# -- Versicolor
x = len(H_versicolor[H_versicolor >= H_versicolor[-1]])/len(H_setosa)
print("p(ponto_novo, versicolor) = ", x)

# -- Setosa
x = len(H_setosa[H_setosa >= H_setosa[-1]])/len(H_setosa)
print("p(ponto_novo, setosa) = ", x)
