from annsa.model_classes import BaseClass
import numpy as np


def construct_baseclass():
    base = BaseClass()

    return base


def test_construct_baseclass():

    _ = construct_baseclass()
    pass


# define the epochs
epochs = np.arange(1, 100, 1)
# define cost function data
# negative linear
not_learning_cost = [(-1 * epoch) + len(epochs) for epoch in epochs]
# 1/sqrt(x) + x
earlystop_cost = [
    (len(epochs) / np.sqrt(epoch) + epoch / 10 - 4) for epoch in epochs]

# create class
base = construct_baseclass()

# define lamba = 1000
# define size = 1x1024(the number of channels)
dim = (1, 1024)
lam = 1000
random_spectrum = np.random.poisson(lam=lam, size=dim)


# data augmentation unit tests
def test_default_data_augmentation():
    data_aug = base.default_data_augmentation(random_spectrum)
    error_msg = "Default data augmentation is not identity function."
    assert(data_aug.all() == random_spectrum.all()), error_msg


def test_poisson_data_augmentation():
    data_aug = base.poisson_data_augmentation(random_spectrum)
    mean = np.mean(data_aug)
    error_msg = "Poisson data augmentation is not poisson sampling."
    assert(abs(lam - mean) / lam < 1), error_msg


# check_earlystop unit tests
def test_check_earlystop_case1():
    """case 1: earlystopping is turned off"""
    earlystop_patience = 0
    epoch = 5
    stopped = base.check_earlystop(epoch,
                                   earlystop_cost[:epoch],
                                   earlystop_patience)
    error_msg = ("With earlystopping_patience=0, "
                 "check_earlystop should return False.")
    assert not stopped, error_msg


def test_check_earlystop_case2():
    """case 2: not enough epochs have passed"""
    earlystop_patience = 70
    epoch = 69
    stopped = base.check_earlystop(
        epoch,
        earlystop_cost[:epoch],
        earlystop_patience)
    error_msg = "If epoch < patience, check_earlystop should return False."
    assert not stopped, error_msg


def test_check_earlystop_case3():
    """case 3: early stopping should be applied."""
    earlystop_patience = 5
    epoch = 69
    stopped = base.check_earlystop(
        epoch,
        earlystop_cost[:epoch],
        earlystop_patience)
    error_msg = "Early stopping should have applied and did not."
    assert stopped, error_msg


def test_check_earlystop_case4():
    """case 4: early stopping was applied too soon"""
    earlystop_patience = 5
    epoch = 50
    stopped = base.check_earlystop(
        epoch,
        earlystop_cost[:epoch],
        earlystop_patience)
    error_msg = "Early stopping was applied too early."
    assert not stopped, error_msg


# not_learning unit tests
def test_not_learning_case1():
    """case 1: not_learning is turned off"""
    not_learning_patience = 0
    not_learning_threshold = 10
    epoch = 5
    stopped = base.not_learning(
        epoch,
        not_learning_cost[:epoch],
        not_learning_patience,
        not_learning_threshold)
    error_msg = "Learning rate is not being checked. Training should continue."
    assert not stopped, error_msg


def test_not_learning_case2():
    """case 2: not enough epochs have passed"""
    not_learning_patience = 7
    not_learning_threshold = 10
    epoch = 6
    stopped = base.not_learning(
        epoch,
        not_learning_cost[:epoch],
        not_learning_patience,
        not_learning_threshold)
    error_msg = "Not enough epochs have passed. Training should continue."
    assert not stopped, error_msg


def test_not_learning_case3():
    """case 3: not learning should be applied."""
    not_learning_patience = 5
    not_learning_threshold = 10
    epoch = 6
    stopped = base.not_learning(
        epoch,
        not_learning_cost[:epoch],
        not_learning_patience,
        not_learning_threshold)
    error_msg = "The network is not learning and should have stopped."
    assert stopped, error_msg
