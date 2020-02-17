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

from utils import test_accuracy, projection, veccos, binary_search_cx, \
                  find_exp_score, find_slab_score, project_l2_centroid, \
                  project_l2_centroid_straight, project_slab, project_slab_straight


class Attacker():
    def __init__(self):
        '''
        This methods initialize the attacker.
        '''
        self.n_updates = 0
        self.poisoning_sequence = []
        self.centroid = {}
        self.min_score, self.max_score = -15, 15
        self.y_att = 1
        
    def set_param(self, datasets, w_0, w_t, R, eta, 
                  defense, n_iter_warmup, max_n_att, def_clean_pts=False):
        '''
        This method specifies the initial classifer, the target classifier,
        the num of online steps the learner will take and the step size.
        '''
        self.w_init, self.w_target = w_0, w_t 
        self.K, self.eta = max_n_att, eta
        self.d = w_0.shape[1]  # weight vectors are 1xd
        self.w_curr = w_0
        self.R = R
        self.reg = 0
        self.reg_mode = 'sphere'
        self.defense = defense  # defense is now a dictionary, with value=tau
        self.def_clean_pts = def_clean_pts
        # number of warmup run iterations, different from the warmup set size
        # If exceeds warmup set size, jump to the begining of the warmup set.
        self.n_iter_warmup = n_iter_warmup
        
        self.X_init, self.Y_init = datasets[0] # generate defense constraints and init w if necessary.
        self.X_clean, self.Y_clean = datasets[1] # the clean data stream
        self.X_valid, self.Y_valid = datasets[2] # validation set
        self.X_test, self.Y_test = datasets[3]   # the actual test set
        self.n_valid = self.Y_valid.shape[0]
        self.n_init = self.Y_init.shape[0]
        self.n_clean = self.Y_clean.shape[0]
        self.n_test = self.Y_test.shape[0]
        
        #print (self.X_init.shape, self.Y_init.shape)
        try:
            assert (w_0.shape == w_t.shape)
        except AssertError:
            print ("w_0 and w_t must have the same dimension")
            
    def set_n_attack(self, n):
        # set the attack budget
        self.K = n
            
    def set_target(self, w_target):
        self.w_target = w_target
            
    def set_defense_threshold(self, method, tau):
        if method=="norm":
            self.R = tau
        else:
            self.defense[method] = tau
        
    def set_defense_range(self, flag):
        self.def_clean_pts = (not "att-only" in flag)
    
    def set_test_set(self, X_test, Y_test):
        self.X_test, self.Y_test = X_test, Y_test
                
    def set_init_set(self, X_init, Y_init):
        self.X_init, self.Y_init = X_init, Y_init
        for y in [1, -1]:
            self.centroid[y]=self.compute_centroid(y)

    def train(self, n):
        '''
        This method trains the online model for n steps.
        If the clean dataset is not enough, jump to the beginning 
        of the dataset then keep training on.
        '''
        eta,X,Y,m = self.eta,self.X_clean,self.Y_clean,self.n_clean
        for i in range(n):
            pt = (X[i%m,:], Y[i%m])
            self.update_w(pt)
        
    def warmup(self, n):
        '''
        This method trains the online model for n steps.
        If the clean dataset is not enough, jump to the beginning 
        of the dataset then keep training on.
        '''
        eta,X,Y,m = self.eta,self.X_clean,self.Y_clean,self.n_clean
        for i in range(n):
            pt = (X[i%m,:], Y[i%m])
            if self.def_clean_pts:
                if self.meet_constraints(pt):
                    self.update_w(pt)
            else:
                self.update_w(pt)
        
    def compute_centroid(self, y_att):
        '''
        This method finds the class centroid of label y_att
        using the initialization data set.
        The output is a 1-d np array.
        '''
        X_init = [x for (x,y) in zip(self.X_init, self.Y_init) if y==y_att]
        n = len(X_init)
        self.centroid[y_att] = sum(X_init)/(n*1.0)
        #print (norm(self.centroid[y_att]))
        return self.centroid[y_att].flatten()
            
    
    def compute_objective(self):
        '''
        This method specifies the objective function to be minimized
        by the attacker.
        
        This objective will be called by a problem solver to generate
        the best attack points.
        '''
        pass
    
    def distance_to_center(self, x):
        '''
        This method finds the distance of a point x to the class center.
        Both x and the class centroid are 1-d np array.
        '''
        return norm(x-self.centroid[self.y_att])
    
    def generate_constraints(self):
        '''
        This method will be used to generate constraints for the attacker's
        optimization process.
        '''
        
        defense, R, d = [self.defense, self.R, self.d]
        constraints = []
        bounds = [(-1,1) for i in range(d)]
        constraints.append(optimize.NonlinearConstraint(lambda x: norm(x), 0, R))
        '''
        if "L2" in defense:
            tau = defense["L2"]
            mu = self.centroid[self.y_att].flatten()
            constraints.append(optimize.NonlinearConstraint(lambda x: norm(x-mu), 0, tau))
        '''
        return [bounds, constraints]
    
    def meet_constraints(self, pt):
        '''
        This function checks if a point is within the defense constraints,
        with a small tolerance constant for float point error.
        
        (x should ideally be a 1-d np array!)
        '''
        x, y = pt
        if norm(x) > (self.R+1e-5):        # check L2 norm
            #print ("not in l2 ball")
            return False
        if "L2" in self.defense:           # check distance to class center
            tau = self.defense["L2"]
            mu = self.centroid[y]
            if (norm(mu-x)>(tau+1e-5)) and (norm(x)>1e-10):
                #print (norm(mu-x), tau)
                #print ("not in l2 centroid")
                return False
        if "slab" in self.defense:         # check slab score
            tau = self.defense["slab"]
            mu = (self.centroid[1], self.centroid[-1])
            if (find_slab_score(mu, pt) > tau+1e-5) and (norm(x)>1e-10):
                #print ("slab constraint violated")
                return False
        return True
        
    def initial_guess(self):
        return self.centroid[self.y_att].flatten()
    
    def slab_scores(self):
        X, Y = self.X_clean, self.Y_clean
        mu = (self.centroid[1], self.centroid[-1])
        scores = [find_slab_score(mu,(X[i,:],Y[i])) for i in range(X.shape[0])]
        scores.sort()
        return (mu, scores)
        #return (max(scores), min(scores), sum(scores)/X.shape[0], scores)
        
    def l2_distances_to_centroid(self):
        X, Y = self.X_clean, self.Y_clean
        mu = (self.centroid[1], self.centroid[-1])
        scores = [0 for i in range(X.shape[0])]
        for i in range(X.shape[0]):
            if Y[i]==1:
                scores[i] = np.linalg.norm(X[i,:]-mu[0])
            else:
                scores[i] = np.linalg.norm(X[i,:]-mu[1])
        #scores = [find_slab_score(mu,(X[i,:],Y[i])) for i in range(X.shape[0])]
        scores.sort()
        return (mu, scores)
        #return (max(scores), min(scores), sum(scores)/X.shape[0], scores)

    def l2_norms(self):
        X, Y = self.X_clean, self.Y_clean
        mu = (self.centroid[1], self.centroid[-1])
        scores = [np.linalg.norm(X[i,:]) for i in range(X.shape[0])]
        scores.sort()
        return (mu, scores)
        #return (max(scores), min(scores), sum(scores)/X.shape[0], scores)

        
    def find_best_poisoning_points(self):
        '''
        This method finds the next poisoning point(s).
        
        The poisoning point(s) will be the solution to the minimization problem.
        '''
        pass

    def update_w(self, pt):
        '''
        This method updates the weight following the OGD rules.
        '''
        w0, d, eta = self.w_curr, self.d, self.eta
        x0, y0 = pt
        x0 = x0.reshape(self.d)
        grad = -y0*x0/(1+find_exp_score(w0,x0,y0)) + (self.reg*w0)
        w1 = w0 - eta*grad
                
        self.w_curr = w1
        self.n_updates +=1 
        
    def peek_w(self, pt):
        '''
        This method returns the projected w in the next time step,
        if the model updates on pt.
        '''
        w0, d, eta = self.w_curr, self.d, self.eta
        x0, y0 = pt
        x0 = x0.reshape(self.d)
        grad = -y0*x0/(1+find_exp_score(w0,x0,y0)) + (self.reg*w0)
        return (w0 - eta*grad)        
    
    def reset(self):
        '''
        This function resets the online learner to its initial states.
        The resets are over:
            1) current weight
            2) number of updates counter
            3) poisoning examples
        '''
        self.w_curr = self.w_init
        self.n_updates = 0
        self.poisoning_sequence = []
        #self.centroid = {}

    def distance_to_target(self):
        return norm(self.w_curr-self.w_target)
    
    def get_poisoning_sequence(self):
        return self.poisoning_sequence

################################################################################
################################################################################

class ConcentratedAttack(Attacker):
    
    def find_best_poisoning_points(self, n_iter=2000, step_size=0.002):
        d, R = self.d, self.R
        dtype = torch.double
        w_target = torch.tensor(self.w_target.flatten(), dtype=dtype, requires_grad=True)
        xpos = torch.zeros([d], dtype=dtype, requires_grad=True)
        xneg = torch.zeros([d], dtype=dtype, requires_grad=True)
        
        if np.abs(R)<=1e-5:
            return (xpos.detach().numpy(), xneg.detach().numpy())
        
        clean_grad = torch.tensor(self.grad.flatten(), dtype=dtype)
        npos, nneg = self.n_att_pos, self.n_att_neg
        
        for i in range(n_iter):
            pos_grad = npos*(-xpos/(1+torch.exp(torch.dot(xpos, w_target))))
            neg_grad = nneg*(xneg/(1+torch.exp(-torch.dot(xneg,w_target))))
            loss = torch.norm(clean_grad+pos_grad+neg_grad)
            
            loss.require_grad = True
            
            loss.backward()
            with torch.no_grad():
                xpos -= xpos.grad*step_size
                xneg -= xneg.grad*step_size
                

            xpos = xpos.detach().numpy()
            xneg = xneg.detach().numpy()
            
            xpos = np.clip(xpos,-1,1)
            xneg = np.clip(xneg,-1,1)
            
            if norm(xpos)>R:
                xpos = xpos/norm(xpos)*R
            if norm(xneg)>R:
                xneg = xneg/norm(xneg)*R
                
            if "L2" in self.defense:
                tau = self.defense["L2"]
                mu_pos = self.centroid[1]
                mu_neg = self.centroid[-1] 
                xpos = project_l2_centroid(mu_pos, xpos, tau)
                xneg = project_l2_centroid(mu_neg, xneg, tau)

            if "slab" in self.defense:
                tau = self.defense["slab"]
                mu = (self.centroid[1], self.centroid[-1])
                xpos = project_slab(mu,(xpos,1),tau,R)
                xneg = project_slab(mu,(xneg,-1),tau,R)

            xpos = torch.tensor(xpos, dtype=dtype, requires_grad=True)
            xneg = torch.tensor(xneg, dtype=dtype, requires_grad=True)
            
        print (loss)
            #if i%200==0:
            #    print (loss)
        #print (loss)
        #print (xpos.detach().numpy(), xneg.detach().numpy())
        return (xpos.detach().numpy(), xneg.detach().numpy())
    
    
    def find_clean_gradient(self, w):

        X, Y, d = self.X_clean, self.Y_clean, self.d
        score = np.matmul(X, w.T)
        score = 1+np.exp(score)
        score = np.repeat(score, d).reshape(X.shape)
        Y = np.repeat(Y, self.d).reshape(X.shape)
        assert (score.shape == X.shape)
        dL0dw0 = np.divide(-X, score)
        dL0dw0 = np.multiply(dL0dw0, Y)
        dL0dw0 = np.sum(dL0dw0, axis=0)
        self.grad = dL0dw0
        
        #print (self.grad.shape)

    def attack(self):
        
        eps = 1e-2*np.sqrt(self.d)
        MAX_STEPS = self.K
        R = self.R
        c = MAX_STEPS
        res = [1]
        dist_init = self.distance_to_target()
        n_shuffle = 100
        best_acc = test_accuracy(self.X_test, self.Y_test, self.w_curr)
        w_best, w_init, w_target = np.copy(self.w_curr), np.copy(self.w_curr), np.copy(self.w_target)
        self.find_clean_gradient(w_target)
        #print (best_acc)
        if self.R==0:
            return [np.array([best_acc]), [], self.w_curr]
        
        for n_att_pos in [0, c//2, c]:
            print ("number of positive instances is {}".format(n_att_pos))
            self.n_att_pos = n_att_pos
            self.n_att_neg = c-n_att_pos
            (pt_pos, pt_neg) = self.find_best_poisoning_points()
            
            #if (norm(pt_pos)>1e-5): pt_pos = pt_pos/norm(pt_pos)*self.R 
            #if (norm(pt_neg)>1e-5): pt_neg = pt_neg/norm(pt_neg)*self.R 
            
            if "L2" in self.defense:
                tau = self.defense["L2"]
                mu_pos = self.centroid[1]
                mu_neg = self.centroid[-1] 
                pt_pos = project_l2_centroid(mu_pos, pt_pos, tau)
                pt_neg = project_l2_centroid(mu_neg, pt_neg, tau)
                    
            if "slab" in self.defense:
                tau = self.defense["slab"]
                mu = (self.centroid[1], self.centroid[-1])
                pt_pos = project_slab(mu,(pt_pos,1),tau,R)
                pt_neg = project_slab(mu,(pt_neg,-1),tau,R)
                
            self.w_curr = w_init
            pts = [(pt_pos, 1) for i in range(self.n_att_pos)]+[(pt_neg,-1) for i in range(self.n_att_neg)]
            
            for j in range(n_shuffle+2):
                if j==0:
                    pass
                elif j==1:
                    pts = pts[::-1]
                else:
                    shuffle(pts)
                for i in range(c):
                    self.update_w(pts[i])
                acc = test_accuracy(self.X_test, self.Y_test, self.w_curr)    
                if acc <= best_acc:
                    w_best = self.w_curr
                    best_acc = acc
                self.w_curr = w_init
            #print (best_acc)
            
        self.w_curr = w_best
        res.append(best_acc)
        #print (res, [], w_best)
        return [np.array(res), [], w_best]

###########################################################################
###########################################################################

class GreedyAttack(Attacker):
    
    def find_optimal_point(self, n_iter=1000, step_size=0.05):
        w_0, w_target, eta, d, R = [self.w_curr, self.w_target, 
                                    self.eta, self.d, self.R]
        dtype = torch.double
        y = self.y_att    # set the y label of the poison point
        w_target = torch.tensor(w_target.flatten(), dtype=dtype, requires_grad=True)
        x = torch.tensor(self.initial_guess(),dtype=dtype,requires_grad=True)
        #x = torch.tensor(torch.zeros(x.shape),dtype=dtype,requires_grad=True)
        x = torch.zeros(x.shape,dtype=dtype,requires_grad=True)
        w_0 = torch.tensor(w_0.flatten(), dtype=dtype, requires_grad=True)

        for i in range(n_iter):
            w = w_0 + eta*y*x/(1+torch.exp(y*torch.dot(w_0, x)))
            loss = torch.norm(w-w_target)
            
            loss.require_grad = True
            
            loss.backward()
            
            with torch.no_grad():
                x -= x.grad/torch.norm(x.grad)*step_size
                

            x = x.detach().numpy()
            x = np.clip(x,-1,1)
            
            if norm(x)>R:
                x = x/norm(x)*R
                
            if "L2" in self.defense:
                tau = self.defense["L2"]
                mu = self.centroid[y]
                x = project_l2_centroid(mu, x, tau)
                
            if "slab" in self.defense:
                tau = self.defense["slab"]
                mu = (self.centroid[1], self.centroid[-1])
                x = project_slab(mu,(x,y),tau,R)
                
            x = torch.tensor(x, dtype=dtype, requires_grad=True)
            
            #if i%100==0:
            #    print (loss)
        #print (loss)
        return (x.detach().numpy(), loss.detach().numpy())            
        
    
    def initial_guess(self):
        
        w, w_target = self.w_curr, self.w_target
        R, defense = self.R, self.defense
        y_t = self.y_att
        #x_t = (w_target-w)/norm(w_target-w)*y_t*self.R
        x_t = (w_target - w)
        if abs(R)<=1e-10:
            return np.zeros(x_t.shape).flatten()
        # scale the point back to [-1, 1]^d
        if np.max(np.abs(x_t)) > 1:
            x_t /= np.max(np.abs(x_t))
            
        if norm(x_t) >= R:
            x_t = x_t/norm(x_t)*R
        
        if "L2" in defense:
            tau = defense["L2"]
            mu = self.centroid[y_t]
            x_t = project_l2_centroid_straight(mu, x_t, tau)
                    
        if "slab" in defense:
            tau = defense["slab"]
            mu = (self.centroid[1], self.centroid[-1])
            x_t = project_slab_straight(mu, (x_t, y_t), tau)
        
        return x_t.flatten()
        
    def find_best_poisoning_points(self):
        
        # initialize the points
        w, w_target, d, reg_mode, R = [self.w_curr, self.w_target, self.d, 
                                       self.reg_mode, self.R]
        defense = self.defense
        
        ys = [1, -1]
        opt_x, opt_y, l = np.zeros(d), 1, np.inf
        
        
        for y in ys:
            self.y_att = y
            x, fun = self.find_optimal_point()
            if (fun < l):
                opt_x, opt_y = x, y
                l = fun
            
        return (opt_x, opt_y)
        
    def attack(self):
        
        eps = 1e-2
        MAX_STEPS = self.K
        c, res, res_w, ws = 0, [1], [1], [self.w_curr]
        dist_init = self.distance_to_target()
        init_acc = test_accuracy(self.X_test, self.Y_test, self.w_curr)
        
        while (c < MAX_STEPS) and (self.distance_to_target()>=eps):
            #if c%20==0:
            #    print (c)
            pt = self.find_best_poisoning_points()
            self.poisoning_sequence.append(pt)
            self.update_w(pt)
            
            acc = test_accuracy(self.X_test, self.Y_test, self.w_curr)
            res.append(min(init_acc, acc))
            w_diff = self.distance_to_target()/dist_init
            res_w.append(w_diff)
            ws.append(self.w_curr)
            c += 1

        for i in range(c, MAX_STEPS):
            res.append(acc)
            res_w.append(w_diff)
            
        return [np.array(res), np.array(res_w), self.w_curr]
        #return [np.array(res), np.array(res_w), ws]
#####################################################################
#####################################################################

class SemiOnlineAttack(Attacker):

    def set_param(self, datasets, w_0, w_t, R, eta, 
                  defense, n_iter_warmup, max_n_att, 
                  adv_init, def_clean_pts=False):
        super().set_param(datasets, w_0, w_t, R, eta, defense, 
                          n_iter_warmup, max_n_att)
        self.X_adv_init, self.Y_adv = adv_init
        self.X_adv = np.copy(self.X_adv_init)
        self.W = [w_0 for i in range(max_n_att+1)]
        self.X_valid, self.Y_valid = datasets[0] # new
        #print (self.defense)
        
    def set_param_lite(self, w_0, w_t, n_iter_warmup, max_n_att, 
                  adv_init, def_clean_pts=False):
        self.w_curr = w_0
        self.w_target = w_t
        self.n_iter_warmup = n_iter_warmup
        self.X_adv_init, self.Y_adv = adv_init
        self.X_adv = np.copy(self.X_adv_init)
        self.K = max_n_att
        self.W = [w_0 for i in range(max_n_att+1)]
        #self.X_valid, self.Y_valid = datasets[0] # new
        #print (self.defense)
        
    def set_n_attack(self, max_n_att):
        super().set_n_attack(max_n_att)
        self.W = [self.w_curr for i in range(max_n_att+1)]
        
    def set_adv_init(self, adv_init):
        self.X_adv_init, self.Y_adv = adv_init
        self.X_adv = np.copy(self.X_adv_init)
        
    def set_initial_w(self, w):
        self.W[0] = w
            
    def reset(self):
        super().reset()
        self.X_adv = np.copy(self.X_adv_init)

    def warmup(self,n):
        super().warmup(n)
        self.W = [np.copy(self.w_curr) for i in range(self.K+1)]
        
    def train_model(self):
        '''
            Train the model using the current adversarial data.
            Compute and update weight vector at each time step.
        '''
        [n, X, Y] = [self.K, self.X_adv, self.Y_adv]
        w = self.W[0]
        eta, reg = self.eta, self.reg
        
        for i in range(n):
            [x, y] = [X[i, :], Y[i]]
            x = x.reshape(w.shape)
            exp_score = find_exp_score(w, x.T, y)
            grad = -(y*x)/(1+exp_score)+reg*w
            w = w - eta*grad # make gradient descent step
            self.W[i+1] = np.copy(w) # save weight at time i+1

    def hessian_w(self, i, W=None, diag=False):
        '''
           This method computes the Hessian of loss w.r.t. to w at time i.
           \pd w_{i+1}/\pd w_i
        '''
        [x, y] = [self.X_adv[i, :], self.Y_adv[i]]
        eta_i, reg, n, d = self.eta, self.reg, self.K, self.d

        if W is None:
            W = self.W[i] 
            
        exp_score = find_exp_score(W,x,y)
        
        if diag:
            res = np.multiply(x,x)
            res = res*(exp_score/((1+exp_score)**2))
            res = (1-2*eta_i*reg) - eta_i*res
            #print res
        else:
            res = np.outer(x, x)
            res = res*(exp_score/((1+exp_score)**2))
            res = (1-2*eta_i*reg)*np.identity(d) - eta_i*res
        return res
    
    def deriv_w_and_x(self, i, W=None):
        '''
            This method finds the derivative of loss w.r.t. weight then x_i at time i.
            \pd^2 L_i / \pd w_i \pd x_i
        '''
        [x, y] = [self.X_adv[i, :], self.Y_adv[i]]
        eta_i, reg, n, d = self.eta, self.reg, self.K, self.d
        
        if W is None:
            W = self.W[i]
            
        exp_score = find_exp_score(W,x,y)
        
        res_2 = np.outer(x, W)        # xw^T
        res_2 = res_2*(exp_score/((1+exp_score)**2))
        
        res_1 = (-y)*np.identity(d)/(1+exp_score) 

        res = -eta_i*(res_1 + res_2)
        return res
    
    def deriv_x(self, i, W=None):
        '''
            This method find \pd L_i / \pd x_i
        '''
        [x, y] = [self.X_adv[i, :], self.Y_adv[i]]
        eta, reg, n, d = self.eta, self.reg, self.K, self.d

        if W is None:
            W = self.W[i]
        
        exp_score = find_exp_score(W,x,y)
        res = -y*W/(1+exp_score)
        return res        
    
    def deriv_w(self, i, W=None):
        '''
            This method find \pd L_i / \pd w_i
        '''
        [x, y] = [self.X_adv[i, :], self.Y_adv[i]]
        eta, reg, n, d = self.eta, self.reg, self.K, self.d

        if W is None:
            W = self.W[i]
            
        exp_score = find_exp_score(W,x,y)
        res = -y*x/(1+exp_score) + 2*reg*W
        return res

    def deriv_w_T(self, i, W=None):
        '''
            This method find the grad of loss at i-th validation point w.r.t. w_T
        '''
        [x, y] = [self.X_valid[i, :], self.Y_valid[i]]
        eta, reg, n, d = self.eta, self.reg, self.K, self.d

        if W is None:
            W = self.W[-1]

        exp_score = find_exp_score(W,x,y)
        res = -y*x/(1+exp_score)
        return res

    def deriv_w_T_angle(self, W=None):
        if W is None:
            W = self.W[-1]
        a = self.w_target.flatten()
        W = W.flatten()
        #print (a.shape, W.shape)
        W_norm = np.linalg.norm(W)
        grad = W_norm*a - (np.dot(a, W)/(W_norm+1e-10))*W
        grad /= (W_norm+1e-10)**2
        grad /= np.linalg.norm(a)
        return grad
        
    def find_grad_of_loss_valid(self):
        '''
            This is a method computing \frac{\pd L}{\pd x_j} for all j.
        '''    
        [n, m, d] = [self.K, self.n_valid, self.d]
       
        L = [0 for i in range(n)]
        Xi = [0 for i in range(n)]
        H = [np.identity(d) for i in range(n)]
        dwdx = [np.identity(d) for i in range(n)]
        for i in range(n):
            H[i] = self.hessian_w(i) # has just changed from i-1 to i
            dwdx[i] = self.deriv_w_and_x(i) # has just changed from i-1 to i
        '''
        for i in range(m):
            Xi[-1] += self.deriv_w_T(i)
        '''
        Xi[-1] = self.deriv_w_T_angle()
        for i in range(n-2,-1,-1):
            Xi[i] = np.dot(Xi[i+1], H[i+1])
            
        for i in range(n):
            L[i] = np.dot(Xi[i], dwdx[i]) 
        
        return L     
        
    def attack_together(self, 
                        attack_method='loss_valid', 
                        num_iter=1, 
                        step_size=100,
                        objective='maximize'):
        '''
            Perturb all entries together.
        '''
        n, R = self.K, self.R
        
        grad = self.find_grad_of_loss_valid()
        grad_norm = [np.linalg.norm(grad[i]) for i in range(n)]
        grad_unit = [grad[i]/(grad_norm[i]+1e-10) for i in range(n)]
        grad_unit = np.array(grad_unit).reshape(self.X_adv.shape)

        if objective == 'maximize':
            self.X_adv += (grad_unit*step_size)
        else:
            self.X_adv -= (grad_unit*step_size)
            
        for i in range(n):
            x, y = self.X_adv[i, :], self.Y_adv[i]
            l_x = norm(x)
            if l_x>=R:
                x*= (R/l_x)
            if "L2" in self.defense:
                tau = self.defense["L2"]
                mu = np.array(self.centroid[y]).reshape(self.d)
                diff = x-mu
                if norm(diff) > tau:
                    x = mu + diff*tau/norm(diff)
            if "slab" in self.defense:
                tau = self.defense["slab"]
                mu = (self.centroid[1], self.centroid[-1])
                x = project_slab(mu,(x,y),tau,R)
            
            self.X_adv[i, :] = np.clip(x, -1, 1)
            #if (not self.meet_constraints((x,y))) and (norm(x)>1e-10):
            #    print ("attack violates rules")

        self.train_model() # refresh the model.
        
    def find_loss_angle(self, w1, w2):
        return np.dot(w1, w2)/(np.linalg.norm(w1)*np.linalg.norm(w2)+1e-10)
        
    def attack(self, n_iter=500, step_size=1):
        
        d = self.d
        if d<50:
            n_iter = 500
        else:
            n_iter = 200
   
        step_size = 0.02*np.sqrt(self.d)
        w_target, w_curr = self.w_target.flatten(), self.W[-1].flatten()
        best_loss = self.find_loss_angle(w_target, w_curr)
        best_w = self.W[-1]
        
        for i in range(n_iter):    
            if i%50==0:
                pass
            self.attack_together(step_size=step_size)
            new_loss = self.find_loss_angle(w_target, self.W[-1].flatten())
            if new_loss>best_loss:
                best_loss = new_loss
                best_w = self.W[-1]
        res_acc = [test_accuracy(self.X_test, self.Y_test, best_w)]
        return [res_acc, [], best_w]
    
    def find_best_poisoning_points(self, n_iter=100, step_size=1):
        
        step_size = 0.02*np.sqrt(self.d)
        best_loss = self.find_loss(self.X_valid, self.Y_valid)
        best_w = self.W[-1]
        
        for i in range(n_iter):    
            if i%50==0:
                #print (i)
                if test_accuracy(self.X_test, self.Y_test, best_w) < 0.05:
                    break
            self.attack_together(step_size=step_size)
            new_loss = self.find_loss(self.X_valid, self.Y_valid)
            if new_loss>best_loss:
                best_loss = new_loss
                best_w = self.W[-1]
        return (self.X_adv, self.Y_adv)
        
    def predict(self, X_test, W=None):
        if W is None:
            W = self.W[-1].T
        pred = np.dot(X_test, W)
        return pred
    
    def find_loss(self, X, Y):
        return np.sum(self.find_raw_loss(X,Y))/(X.shape[0])
    
    def find_raw_loss(self, X, Y, valid=False):
        W, reg, n = self.W[-1].T, self.reg, X.shape[0]
        pred = np.dot(X, W).reshape(Y.shape)
        score = np.multiply(pred, Y)
        if valid:
            loss = np.log(1+np.exp(-score))+reg*np.dot(W.T,W)
        else:
            loss = np.log(1+np.exp(-score))
        return loss  
    
    def find_acc(self, X=None, Y=None, W=None):
        if X is None:
            X, Y = self.X_test, self.Y_test
        if W is None:
            W = self.W[-1].T
        return test_accuracy(X, Y, W)

#####################################################################
#####################################################################

class StraightAttack(Attacker):
            
    def find_best_poisoning_points(self):
        
        # initialize the points
        R, defense = self.R, self.defense
        w, w_target, d= self.w_curr, self.w_target, self.d
        best, best_x, best_y = np.inf, 0, 1
        self.gamma = 1/self.eta         #temporary measure
        #print (self.gamma)
        
        for y_t in [1,-1]:
            
            x_t = (w_target-w)*self.gamma*y_t

            # scale the point back to [-1, 1]^d
            if np.max(np.abs(x_t)) > 1:
                x_t /= np.max(np.abs(x_t))

            if norm(x_t) > R:
                x_t = x_t/norm(x_t)*R
                        
            # scale the point back to the L_2 centroid defense
            if "L2" in defense:
                tau = defense["L2"]
                mu = self.centroid[y_t]
                x_t = project_l2_centroid_straight(mu, x_t, tau)
            
            # scale the point back to the slab defense constraint
            if "slab" in defense:
                tau = defense["slab"]
                mu = (self.centroid[1], self.centroid[-1])
                x_t = project_slab_straight(mu, (x_t, y_t), tau)
            
            # if the resulted w is closer to w_target, pick this point.
            w = self.peek_w((x_t,y_t))
            
            if norm(w-w_target)<best:
                best = norm(w-w_target)
                best_x, best_y = x_t, y_t
                
        return (best_x, best_y)
        
    def attack(self):
        
        eps, MAX_STEPS = 1e-2, self.K
        c, res, res_w = 0, [test_accuracy(self.X_test,self.Y_test,self.w_curr)], [1]
        dist_init = self.distance_to_target()
        #self.gamma = min(self.R/(norm(self.w_target-self.w_curr)), 1/self.eta)
        self.gamma = 1/self.eta
        acc = test_accuracy(self.X_test, self.Y_test, self.w_curr)
        w_diff = 1
        
        while (c < MAX_STEPS) and (self.distance_to_target()>=eps):
            
            pt = self.find_best_poisoning_points()
            self.poisoning_sequence.append(pt)
            if (not self.meet_constraints(pt)) and (norm(pt[0])>1e-10):
                print ("Poisoning sample violates rules.")
            else:    
                self.update_w(pt)
                         
            acc = test_accuracy(self.X_test, self.Y_test, self.w_curr)
            res.append(acc)
            w_diff = self.distance_to_target()/dist_init
            res_w.append(w_diff)

            c += 1

        for i in range(c, MAX_STEPS):
            res.append(acc)
            res_w.append(w_diff)
            
        # res tracks the test accuracy at each attack steps
        # res_w tracks the distance to target, normalized by initial distance.
            
        return [np.array(res), np.array(res_w), self.w_curr]

