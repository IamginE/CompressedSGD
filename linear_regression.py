import os
import sys
import numpy as np
import datetime
import torch

optml_directory = "OptML-ex06-solution"
os.chdir(optml_directory)
sys.path.insert(1, optml_directory)

from helpers import *

# Source: https://github.com/epfml/OptML_course/blob/master/labs/ex06/solution
height, weight, gender = load_data(sub_sample=False, add_outlier=False)
default_x, default_mean_x, default_std_x = standardize(height)
default_b, default_A = build_model_data(default_x, weight)

def full_objective(targets_b, data_A, params_x):
    # Source: https://github.com/epfml/OptML_course/blob/master/labs/ex06/solution
    """Compute the least squares objective over the whole dataset"""
    return 0.5 * np.mean(((data_A @ params_x) - targets_b)**2)


def minibatch_gradient(targets_b, data_A, params_x):
    # Source: https://github.com/epfml/OptML_course/blob/master/labs/ex06/solution
    """
    Compute a mini-batch stochastic gradient from a subset of `num_examples` from the dataset.

    :param targets_b: a numpy array of shape (num_examples)
    :param data_A: a numpy array of shape (num_examples, num_features)
    :param params_x: compute the mini-batch gradient at these parameters, numpy array of shape (num_features)

    :return: gradient: numpy array of shape (num_features)
    """
    batch_size = len(targets_b)
    err = targets_b - data_A.dot(params_x)
    grad = -data_A.T.dot(err) / batch_size
    return grad


def stochastic_gradient(targets_b, data_A, params_x, batch_size=1):
    # Source: https://github.com/epfml/OptML_course/blob/master/labs/ex06/solution
    """
    Compute a stochastic gradient

    :param targets_b: numpy array of size (num_examples)
    :param data_A: numpy array of size (num_examples, num_features)
    :param params_x: compute the mini-batch gradient at these parameters, numpy array of shape (num_features)
    :param batch_size: integer: number of datapoints to compute the stochastic gradient from

    :return: gradient, numpy array of shape (num_features)
    """
    dataset_size = len(targets_b)
    indices = np.random.choice(dataset_size, batch_size, replace=False)
    return minibatch_gradient(targets_b[indices], data_A[indices, :], params_x)


def stochastic_gradient_descent(
        targets_b,
        data_A,
        initial_x,
        batch_size,
        max_iters,
        initial_learning_rate,
        only_sign,
        rand_zero,
        decreasing_learning_rate=False):
    # Sources: 1. https://github.com/epfml/OptML_course/blob/master/labs/ex06/solution
    #          2. https://github.com/jxbz/signSGD/blob/master/signSGD_zeros.ipynb (for signSGD) - modified
    """
    Mini-batch Stochastic Gradient Descent for Linear Least Squares problems.

    :param targets_b: numpy array of size (num_examples)
    :param data_A: numpy array of size (num_examples, num_features)
    :param initial_x: starting parameters, a numpy array of size (num_features)
    :param batch_size: size of the mini-batches
    :param max_iters: integer, number of updates to do
    :param initial_learning_rate: float
    :param only_sign: boolean, if True, use signSGD, otherwise use SGD
    :param rand_zero: boolean, if True, replace zero gradients with ±1 (chosen randomly)
    :param decreasing_learning_rate: if set to true, the learning rate should decay as 1 / t

    :return:
    - objectives, a list of loss values on the whole dataset, collected at the end of each pass over the dataset (epoch)
    - param_states, a list of parameter vectors, collected at the end of each pass over the dataset
    """
    xs = [initial_x]  # parameters after each update
    objectives = []  # loss values after each update
    x = initial_x

    for iteration in range(max_iters):
        grad = stochastic_gradient(targets_b, data_A, x, batch_size=batch_size)
        if decreasing_learning_rate:
            lr = initial_learning_rate / (iteration + 1)
        else:
            lr = initial_learning_rate

        # Turning the vector of gradients into a vector of their signs
        if only_sign:
            # take sign of gradient
            grad = np.sign(grad)

            # randomise zero gradients to ±1
            if rand_zero:
                grad[grad == 0] = np.random.randint(0, 2, len(grad[grad == 0])) * 2 - 1
                assert not (grad == 0).any()

        # Updating the parameters
        x = x - lr * grad

        # store x and objective
        xs.append(x.copy())
        objective = full_objective(targets_b, data_A, x)
        objectives.append(objective)

        if iteration % 1000 == 0:
            print("SGD({bi:04d}/{ti:04d}): objective = {l:10.2f}".format(
                bi=iteration, ti=max_iters - 1, l=objective))
    return objectives, xs


def optimize_with_sgd(data, best_objective, only_sign=False, rand_zero=True):
    # Sources: 1. https://github.com/epfml/OptML_course/blob/master/labs/ex06/solution - modified
    #          2. https://github.com/jxbz/signSGD/blob/master/signSGD_zeros.ipynb (for signSGD)

    A, b = data

    max_iters = int(1e4)

    mu = np.linalg.norm(A.T @ A, -2) / len(A)
    # L = np.linalg.norm(A.T @ A, 2) / len(A)

    gamma0 = 2 / mu
    batch_size = 1

    x_initial = np.zeros(A.shape[1])

    # Start SGD.
    start_time = datetime.datetime.now()
    if only_sign:
        sgd_objectives_dec_gamma_mu, sgd_xs_dec_gamma_mu = stochastic_gradient_descent(
            b, A, x_initial, batch_size, max_iters, gamma0, only_sign, rand_zero, decreasing_learning_rate=False)
    else:
        sgd_objectives_dec_gamma_mu, sgd_xs_dec_gamma_mu = stochastic_gradient_descent(
            b, A, x_initial, batch_size, max_iters, gamma0, only_sign, rand_zero, decreasing_learning_rate=True)
    end_time = datetime.datetime.now()

    # Print result
    execution_time = (end_time - start_time).total_seconds()
    print("SGD: execution time = {t:.3f} seconds".format(t=execution_time))
    print(f"f(x*) = {best_objective:.5f}")
    print(f"Final value of f(x) = {sgd_objectives_dec_gamma_mu[-1]:.5f}")
    print(f"Final value of f(x) - f(x*) = {(sgd_objectives_dec_gamma_mu[-1] - best_objective):.5f}")


def try_model_on_data(data=(default_A, default_b), optimizer="SGD"):
    # DESCRIPTION:
    # This function trains a simple linear regression model on
    # the given dataset using the chosen optimizer under a
    # least-squares loss. It reports how long it takes, and
    # what the final loss is.

    # INPUT:
    # data: A tuple of (A, b) where A is an n×d matrix
    # containing the values of the independent variables and a
    # column of ones, and b is an n×1 vector containing the
    # values of the dependent variable. By default, it uses
    # the dataset provided in the "solution" folder (check
    # optml_directory to see where it is).
    #
    # optimizer: A string indicating what optimizer should be
    # used. It should be one of these three options:
    #   1. 'SGD' (default): Stochastic Gradient Descent
    #   2. 'signSGD': The signSGD algorithm in its vanilla form
    #   3. 'signSGDplus': The modified version of signSGD

    # OUTPUT:
    # No output (based on the chosen optimizer, calls the
    # proper function)

    A, b = data

    assert A.shape[0] == b.shape[0], "A and b have different number of rows!"

    x_star = np.linalg.solve(A.T @ A, A.T @ b)
    best_objective = full_objective(b, A, x_star)

    if optimizer == "SGD":
        optimize_with_sgd(data, best_objective, only_sign=False,
                          rand_zero=False)
    elif optimizer == "signSGD":
        optimize_with_sgd(data, best_objective, only_sign=True,
                          rand_zero=True)
    elif optimizer == "signSGDplus":
        # TODO
        return
    else:
        raise ValueError(f"Cannot use {optimizer}; invalid "
              f"optimizer name!")


try_model_on_data()
try_model_on_data(optimizer="signSGD")
