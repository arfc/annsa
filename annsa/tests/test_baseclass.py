from annsa.model_classes import BaseClass
import numpy as np 
import matplotlib.pyplot as plt 
from annsa.template_sampling import make_random_spectrum

def construct_baseclass():
    base = BaseClass()

    return base
    
def test_construct_baseclass():

    _ = construct_baseclass()
    pass

#define the epochs
epochs = np.arange(1,100,1)
#define cost function data
not_learning_cost = [(-1*epoch) + len(epochs) for epoch in epochs] #negative linear
earlystop_cost = [(len(epochs)/np.sqrt(epoch) + epoch/10 - 4) for epoch in epochs] #1/sqrt(x) + x


#create class
base = construct_baseclass()

#define lamba = 1000
#define size = 1x1024(the number of channels)
dim = (1,1024)
lam = 1000
random_spectrum = np.random.poisson(lam=lam, size=dim)

#data augmentation unit tests
def test_default_data_augmentation():
    data_aug = base.default_data_augmentation(random_spectrum)
    assert(data_aug.all() == random_spectrum.all()), "Default data augmentation is not identity function."

def test_poisson_data_augmentation():
    data_aug = base.poisson_data_augmentation(random_spectrum)
    mean = np.mean(random_spectrum)
    assert(abs(lam - mean)/lam < 1), "Poisson data augmentation is not poisson sampling."

#check_earlystop unit tests
def test_check_earlystop_case1():
    """case 1: earlystopping is turned off"""
    earlystop_patience = 0
    epoch = 5
    stopped = base.check_earlystop(epoch, earlystop_cost[:epoch], earlystop_patience)
    assert(stopped==False), "With earlystopping_patience=0, check_earlystop should return False."

def test_check_earlystop_case2():
    """case 2: not enough epochs have passed"""
    earlystop_patience = 70
    epoch = 69
    stopped = base.check_earlystop(epoch, earlystop_cost[:epoch], earlystop_patience)
    assert(stopped==False), "If epoch < patience, check_earlystop should return False."

def test_check_earlystop_case3():
    """case 3: early stopping should be applied."""
    earlystop_patience = 5
    epoch = 69
    stopped = base.check_earlystop(epoch, earlystop_cost[:epoch], earlystop_patience)
    assert (stopped==True), "Early stopping should have applied and did not."

def test_check_earlystop_case4():
    """case 4: early stopping was applied too soon"""
    earlystop_patience = 5
    epoch = 50
    stopped = base.check_earlystop(epoch, earlystop_cost[:epoch], earlystop_patience)
    assert (stopped==False), "Early stopping was applied too early."

#not_learning unit tests
def test_not_learning_case1():
    """case 1: not_learning is turned off"""
    not_learning_patience = 0
    not_learning_threshold = 10
    epoch = 5
    stopped = base.not_learning(epoch, not_learning_cost[:epoch], not_learning_patience, not_learning_threshold)
    assert (stopped==False), "Not learning is turned off, should not have stopped."

def test_not_learning_case2():
    """case 2: not enough epochs have passed"""
    not_learning_patience = 7
    not_learning_threshold = 10
    epoch = 6
    stopped = base.not_learning(epoch, not_learning_cost[:epoch], not_learning_patience, not_learning_threshold)
    assert (stopped==False), "Not learning applied too early. epoch < patience."

def test_not_learning_case3():
    """case 3: not learning should be applied."""
    not_learning_patience = 5
    not_learning_threshold = 10
    epoch = 6
    stopped = base.not_learning(epoch, not_learning_cost[:epoch], not_learning_patience, not_learning_threshold)
    assert (stopped==True), "Not learning should have applied and did not."
