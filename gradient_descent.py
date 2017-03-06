#! /usr/bin/env python3
# Gradient Descent Algorithm

from math import sqrt


# Predict output function
def predict_output(features, weights):
    # Add code to predict ouput
    # predictions = np.dot(features, weights)
    return predictions


# Feature derivative function
def feature_derivative(errors, features):
    # Add code to find feature derivative
    # derivative = 2*np.dot(errors, features)
    return derivative


# In order for this algorithm to work we need to more functions.
#    1. predict_output function
#    2. feature_derivative function
def gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    gradient_magnitude = 0
    while not converged:
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output
        gradient_sum_squares = 0
        for i in range(len(weights)):
            derivative = feature_derivative(errors, feature_matrix[:,i])
            gradient_sum_squares += derivative * derivative
            weights[i] -= step_size * derivative
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return weights
