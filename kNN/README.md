# Abalone
Demonstration of various machine-learning models to determine the age of Abalone (a type of sea snail) given various characteristics (weight, diameter, etc).

The original dataset can be obtained from the UCI Machine Learning Respository, here: https://archive.ics.uci.edu/ml/datasets/Abalone

The dataset contains 4177 entries, and this task can be cast as an (integer) regression problem (in which we try to predict the age of the Abalone directly), or a classification problem (in which the age of the Abalone is binned into an age category, and we try to predict this category).

Each sub-directory here contains a ready-to-run python3 script, which features an implementation of a particular model. For example, navigating to the k-NN (k-Nearest Neighbours) directory and typing:

python3 abalone_k_nearest_neighbours.py

Will fit a k-Nearest Neighbours model to a randomly chosen training & test set from the data, and will print out model performance metrics.

The code has been written to be as simple as possible, so that the user can change the train/test sets and model parameters as desired.
