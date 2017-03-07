#! /usr/bin/env python3
# Gradient Descent Algorithm

import numpy as np
import pandas as pd
import scipy

from math import sqrt


# Collect training and test data.
training_data = pd.read_csv('../training_data.csv')
test_data = pd.read_csv('../test_data.csv')


# Predict output function
def predict_output(features, weights):
    """
    ===> Predict output based on features and weights
    """
    predictions = np.dot(features, weights)
    return predictions


# Feature derivative function
def feature_derivative(errors, features):
    derivative = 2*np.dot(errors, features)
    return derivative


# In order for this algorithm to work we need to more functions.
#    1. predict_output function
#    2. feature_derivative function
def gradient_descent(feature_matrix, output,
                     initial_weights, step_size,
                     tolerance):
    """
    ===> The Gradient Descent Algorithm. <===
    """
    converged = False
    weights = np.array(initial_weights)
    gradient_magnitude = 0
    while not converged:
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output
        gradient_sum_squares = 0
        for i in range(len(weights)):
            derivative = feature_derivative(errors, feature_matrix[:, i])
            gradient_sum_squares += derivative * derivative
            weights[i] -= step_size * derivative
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return weights


if __main__ == '__name__':
    # Add functions to call here
