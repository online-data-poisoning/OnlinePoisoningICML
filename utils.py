from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numpy.linalg import norm
from sklearn import preprocessing

import pandas as pd
import sklearn
from math import sqrt
from random import shuffle
from numpy.linalg import qr
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.examples.tutorials.mnist import input_data

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pickle
import time

import torch

# some utility functions

def test_accuracy(X_test, Y_test, w):
    '''
    This function computes the test accuracy of a linear classifier on the test set.
    Input:
      X_test: nxd data
      Y_test: true label
      w: 1xd weight
    Output:
      the accuracy in a single number
    '''
    n = X_test.shape[0]
    res = np.matmul(X_test, w.T)
    res = np.array([1 if res[i]>0 else -1 for i in range(n)])
    res.reshape(Y_test.shape)
    total = np.sum(res==Y_test)
    return total*1.0/n

def projection(u, v):
    '''
    This function finds v's projection and the perpendicular components on u.
    Input:
      u, v: vectors, either 1-d array or row vectors (1xd)
    Output:
      a: v's projection on u
      b: v's perpendicular component on u
      a,b has the same shape as u,v
    '''
    # u,v are row vectors
    # a: v's projection on u
    # b: v's component perpendicular to u
    a = np.dot(u, v.T)*u/((norm(u))**2)
    b = v - a
    return (a, b)

def veccos(u, v):
    '''
    This function finds cos(u,v) of two vectors.
    Input:
      u,v: 1-d vector or 1xd row vectors.
    Output:
      cos(u,v), a number
    '''
    return np.dot(u, v.T)/(norm(u)*norm(v))

def contaminate_dataset(dataset, level, R):
    
    X, Y = dataset
    n = Y.shape[0]
    to_change = np.random.choice(n, int(n*level), replace=False)
    for i in to_change:
        x = X[i, :]
        x *= -1
        x = np.clip(x, -1, 1)
        X[i, :] = x
    return (X, Y)

'''
def keep_best(res):
    for i in range(len(res)):
        for j in range(1, len(res[0])):
            if res[i][j-1]<res[i][j]:
                res[i][j] = res[i][j-1]

    for j in range(len(res[0])):
        for i in range(1, len(res)):
            if res[i-1][j]<res[i][j]:
                res[i][j] = res[i-1][j]
    return res
'''
    
def binary_search_cx(x, mu, tau, c_min=0, c_max=1):
    '''
    This function scales a vector such that it is within tau from a center mu.
    Input:
      x: the initial vector, either 1-d np array or 2-d row/column vectors.
      mu: the center, same dim as x
      tau: the maximum L2 distance from mu.
      c_min, c_max: the min and max factor of scaling.
    Output:
      a vector x' = cx, c>0, same dimension as the input x. 
    '''
    
    if (norm(x-mu) <= tau+1e-10):
        return x
    
    max_step = 20
    c = (c_min+c_max)/2
    
    for i in range(max_step):
        #print (i, norm(c*x-mu), tau)
        if norm(c*x-mu) > tau:
            c_max = c
        else:
            c_min = c
        c = (c_min+c_max)/2
    
    return c*x
            
def find_exp_score(w,x,y):
    '''
    This function finds the exp_score of a point (x,y) under model w.
    Inputs:
      w: weight vector, either 1-d or row vector.
      x: sample point, either 1-d or column vector
      y: sample label.
    Output:
      exp(yw^Tx)
    (The clipping ensures safe exp without overflow.)
    '''
    return np.exp((np.dot(w, x)*y).clip(-15,15))      # yw_Tx


def project_l2_centroid(mu, x, tau):
    '''
    This function projects a vector x back to the L2 ball centered at x with radius tau.
    Inputs:
      mu: center of the ball, can be a 1d array, row or column vector.
      x: original input point
      tau: ball radius
    Output:
      x's projection back into the ball.
    (x, mu and the output will have the same dimension.)
    '''

    if norm(x-mu)<tau:
        return x
    
    if abs(tau)<1e-5:
        return mu
    
    return mu+(x-mu)/norm(x-mu)*tau

def project_l2_centroid_straight(mu, x, tau):
    '''
    This function projects a point x back to an L2 ball **without** changing its direction.
    Inputs:
      mu: the ball center
      x: the original point
      tau: the ball radius
    Output:
      x's projection back to the ball and equals to cx, for some c>0.
      returns a zero vector of the same dimension to x if no such projection is possible.
    '''
    if norm(x-mu)<tau:
        return x
    if abs(tau)<1e-5:
        return mu
    if norm(mu)<tau:
        return binary_search_cx(x, mu, tau)
    u, v = projection(x, mu)
    if norm(v)>tau:
        return np.zeros(x.shape)
    if veccos(u, x)<0:
        return np.zeros(x.shape)
    return u

    
def find_slab_score(mu, pt):
    '''
    This function finds the outlier score under slab defense.
    Inputs:
      mu: a list of the class centroids
      pt: the sample (x,y)
    Outputs:
      the slab outlier score.
    (x and the centroids should be a 1-d array!)
    '''
    x, y = pt
    mu_pos, mu_neg = mu
    beta = mu_pos-mu_neg
    mu_y = mu_pos if y==1 else mu_neg
    x = x.reshape(beta.shape)
    return np.abs(np.dot(beta, x-mu_y))

def project_slab_straight(mu, pt, tau):
    '''
    This function projects a point to satisfy the slab defense constraint,
    **without** changing its direction.
    
    If no such point is possible, return a zero output with the same dim as x.
    
    Inputs:
      mu: a list/tuple of the class centroids
      pt: the (x,y) pair
      tau: the defense parameter
    Output:
      A scaled version of x, i.e. cx for some c>=0.
    (x and the centroids should be a 1-d array!)
    '''
    if (find_slab_score(mu, pt) <= tau):
        return pt[0]
    x, y = pt
    mu_pos, mu_neg = mu
    beta = mu_pos-mu_neg
    mu_y = mu_pos if y==1 else mu_neg
    x = x.reshape(beta.shape)
    a, b, c = np.dot(beta, x), -np.dot(beta, mu_y), 1
    if (a>=0) and (tau-b<0):
        #print (a, tau, tau-b)
        return np.zeros(x.shape)
    if (a<=0) and (-tau-b>0):
        return np.zeros(x.shape)
    if (a>(tau-b)) and ((tau-b)>=0):
        c = (tau-b)/a
    elif ((-tau-b)>=a) and (a>=0):
        c = (-tau-b)/a
    elif ((tau-b)<a) and (a<=0):
        c = (tau-b)/a
    elif (a<=(-tau-b)) and ((-tau-b)<=0):
        c = (-tau-b)/a
    return c*x
    
def project_slab(mu, pt, tau, R):
    '''
    This function projects a point to satisfy the slab defense constraint.
    
    If no such point is possible, return a zero output with the same dim as x.
    
    Inputs:
      mu: a list/tuple of the class centroids
      pt: the (x,y) pair
      tau: the defense parameter
      R: maximum L2 allowed after projection.
    Output:
      A scaled version of x, i.e. cx for some c>=0.
    (x and the centroids should be a 1-d array!)    
    '''
    if (find_slab_score(mu, pt) <= tau) and (norm(pt[0])<= R+1e-5):
        return pt[0]
    x, y = pt
    mu_pos, mu_neg = mu
    beta = mu_pos-mu_neg
    mu_y = mu_pos if y==1 else mu_neg
    x = x.reshape(beta.shape)
    u, v = projection(beta, x)
    a, b, c = np.dot(beta, u), -np.dot(beta, mu_y), 1
    if a==0:
        return np.zeros(x.shape)
    elif (a>(tau-b)):
        c = (tau-b)/a
    elif (a<(-tau-b)):
        c = (-tau-b)/a
    
    if (norm(c*u+v)<=R):     # if the L2 norm is small, return the res
        return c*u+v
    else:                    # scale the component perpendicular to mu_pos-mu_neg
        norm_v = np.sqrt(R**2-norm(c*u)**2)
        v = v*norm_v/norm(v)
    return c*u+v


def find_regime_threshold(mu, w_0, w_t, defense_method):
    
    if defense_method == "norm":
        return (0, 0)
    elif defense_method == "L2":
        mu1, mu0 = mu[0].flatten(), mu[1].flatten()
        high = min(np.linalg.norm(mu1), np.linalg.norm(mu0))
        b = (w_t-w_0).flatten()
        if ((np.dot(mu1, b))>0) or ((np.dot(mu0, -b))>0):
            low = 0
        else:
            s1 = abs(np.dot(mu1, b)/np.linalg.norm(b))
            s2 = abs(np.dot(mu0, -b)/np.linalg.norm(-b))
            low = min(s1, s2)
            low = max(low, 0)
        return (low, high)
    elif defense_method == "slab":
        mu1, mu0 = mu[0].flatten(), mu[1].flatten()
        b = mu1 - mu0
        if np.dot(b, (w_t-w_0).flatten())<0:
            b = -b
        t1 = np.dot(-b, mu1)
        t2 = np.dot(b, mu0)
        low = min(t1, t2)
        low = max(low, 0)
        high = min(abs(t1), abs(t2))
        return (low, high)