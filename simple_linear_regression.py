#! /usr/bin/env python3
# Simple Linear Regression

# These are some of the libraries that can be imported
import scipy
import numpy as np
import pandas as pd

# Be sure to import a data set to run the function over.
training_data = pd.read_csv('../training_data.csv')
test_data = pd.read_csv('../test_data.csv')


# f(x) = a + bX
def simple_linear_regression(input_feature, output):
    input_sum = input_feature.sum()
    output_sum = output.sum()
    N = input_feature.size()
    input_mean = input_sum/N
    output_mean = output_sum/N
    in_out_prod = input_feature * output
    in_out_prod_sum = in_out_prod.sum()
    prod_sum = output_sum * input_sum
    prod_mean = prod_sum/N
    sqr_test = input_feature * input_feature
    sqr_test_sum = sqr_test.sum()
    sqr_sum = input_sum * input_sum
    sqr_mean = sqr_sum/N
    slope = (in_out_prod_sum - prod_mean)/(sqr_test_sum - sqr_mean)
    intercept = output_mean - (input_mean * slope)
    return(intercept, slope)


if __main__ == '__name__':
    (intercept, slope) = simple_linear_regression(training_data['feature1'],
                                                  training_data['target'])
    print("Intercept: ", str(intercept))
    print("Slope    : ", str(slope))
