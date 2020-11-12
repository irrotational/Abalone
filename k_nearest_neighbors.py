from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as rnd

test_fraction = 0.1 # Fraction of data used as test set

dataset = pd.read_csv('./abalone.data',error_bad_lines=False)
dataset['Age'] = [ (x+1.5) for x in dataset['Rings'].values ] # data has 'Rings', but we're trying to calculate age = rings + 1.5 (according to abalone.names document)
dataset.drop(columns=['Rings'],inplace=True) # The 'Rings' column is redundant now (instead we have the 'Age' column from the line above)

# Encode 'Sex' column
le_sex = LabelEncoder() # Create a LabelEncoder
le_sex.fit(dataset['Sex']) # Encode the 'Sex' column (originally type string: 'M','F' or 'I') into a format suitable for scikit-learn classification
dataset['Sex'] = le_sex.transform(dataset['Sex']) # Now replace the 'Sex' column with the newly transformed one

# Bin & encode 'Age' column
binned_ages=pd.qcut(dataset['Age'],q=6) # Bin the Ages into 6 bins (can play with number of bins)
dataset['Binned Age']=binned_ages # Insert this into the dataframe and call the column 'Binned Age'
le_age = LabelEncoder() # Create a LabelEncoder
le_age.fit(dataset['Binned Age']) # Encode the Binned Ages into a format suitable for scikit-learn classification
dataset['Binned Age'] = le_age.transform(dataset['Binned Age']) # Now replace the 'Binned Age' column with the newly transformed one
# Print out a description of the Binned Ages to the user
ages_and_their_bin=set(zip(dataset['Binned Age'].values,binned_ages))
dataset.drop(columns=['Age'],inplace=True)
print('Replaced \'Rings\' column with \'Age\' in years (equal to rings+1.5)')
print('Column \'Age\' has been binned into %d bins, which are:' % ( len(ages_and_their_bin) ) )
for val in ages_and_their_bin:
	bin_idx,bin_descr = val[0],val[1]
	print('Bin %d:' % (bin_idx),bin_descr )
print('\n')

print('Dataset snippet (check this matches your expectations):\n')
print(dataset.head(10))
print('\n')

# Plot summary statistics as histogram (uncomment to activate)
#dataset.hist() # histogram summary
#plt.show()
#print(dataset.isna().sum()) # check for missing values (there aren't any here) - uncomment to activate

train,test = train_test_split(dataset,test_size=test_fraction) # splits dataset & shuffles

# The last column (age) is what we want to predict
x_train,y_train = train.values[:,0:-1],train.values[:,-1]
x_test,y_test = test.values[:,0:-1],test.values[:,-1]

########################################################################################################################
# ZeroR + Random classifier

def ZeroR_and_Random(y_train,y_test):
	classes=set(y_train)
	classdict={k:0 for k in classes}
	for label in y_train:
		classdict[label] += 1
	modal_class = max(classdict, key=classdict.get)
	num_entries_in_modal_class = classdict[modal_class]
	print('Modal class in Train Set was %s with %d entries' % (str(modal_class),num_entries_in_modal_class))
	#for key, value in classdict.items(): # uncomment to print out number of occurences of each label
	#	print('%d occurences of label %s' % (value,key))
	ZeroR_correct_count=0
	random_correct_count=0
	for label in y_test:
		ZeroR_guess = modal_class
		random_guess = rnd.sample(classes,1)[0] # gets a random item from a set. This function returns a list of length 1 (so take first element).
		if label == ZeroR_guess:
			ZeroR_correct_count+=1
		if label == random_guess:
			random_correct_count+=1
	ZeroR_accuracy_test = ZeroR_correct_count/len(y_test)
	random_accuracy_test = random_correct_count/len(y_test)
	ZeroR_correct_count=0
	random_correct_count=0
	for label in y_train:
		ZeroR_guess = modal_class
		random_guess = rnd.sample(classes,1)[0] # gets a random item from a set. This function returns a list of length 1 (so take first element).
		if label == ZeroR_guess:
			ZeroR_correct_count+=1
		if label == random_guess:
			random_correct_count+=1
	ZeroR_accuracy_train = ZeroR_correct_count/len(y_train)
	random_accuracy_train = random_correct_count/len(y_train)
	print('ZeroR classifier had an accuracy of %f on the Train set and %f on the Test set'%(ZeroR_accuracy_train,ZeroR_accuracy_test))
	print('Random classifier had an accuracy of %f on the Train set and %f on the Test set'%(random_accuracy_train,random_accuracy_test))

ZeroR_and_Random(y_train,y_test) # ZeroR classifier and a Random classifier - Use as a baseline for performance

########################################################################################################################
# Parameters have only been changed from the scikit-learn default if they gave better performance.

y_train = y_train.flatten() # The KNeighborsClassifier method requires the y data to be of shape (n_samples,) so we flatten here (it's currently of shape (n_samples,1))

neigh = KNeighborsClassifier(n_neighbors=20).fit(x_train, y_train)
train_score = neigh.score(x_train,y_train)
test_score = neigh.score(x_test,y_test)
print('\n')
print('k-Nearest Neighbours with n_neighbours=%d scores:' % (neigh.n_neighbors))
print('Train Set Score: %f' % (train_score))
print('Test Set Score: %f' % (test_score))

preds = neigh.predict(x_test) # predictions on test set from model

num_correct=0
for idx, pred in enumerate(preds):
	if pred == y_test[idx]:
		num_correct+=1
	# Print out each prediction Vs actual, if desired (uncomment to activate)
	#print('Prediction:',pred)
	#print('Actual:',y_train[idx])

