#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers
"""
import numpy as np
import random
from numpy.random import rand
import os
from shutil import copyfile
from gym import spaces
from scipy.stats import norm


def compute_prob_max(mean_list, sigma_list):
    n_actions = len(mean_list)
    lower_limit = mean_list - 8 * sigma_list
    upper_limit = mean_list + 8 * sigma_list
    epsilon = 1e-5
    _epsilon = 1e-25
    n_trapz = 100
    x = np.zeros(shape=(n_trapz, n_actions))
    y = np.zeros(shape=(n_trapz, n_actions))
    integrals = np.zeros(n_actions)
    for j in range(n_actions):
        if sigma_list[j] < epsilon:
            p = 1
            for k in range(n_actions):
                if k != j:
                    p *= norm.cdf(mean_list[j], loc=mean_list[k], scale=sigma_list[k] + _epsilon)
            integrals[j] = p
        else:
            x[:, j] = np.linspace(lower_limit[j], upper_limit[j], n_trapz)
            y[:, j] = norm.pdf(x[:, j], loc=mean_list[j], scale=sigma_list[j] + _epsilon)
            for k in range(n_actions):
                if k != j:
                    y[:, j] *= norm.cdf(x[:, j], loc=mean_list[k], scale=sigma_list[k] + _epsilon)
            integrals[j] = (upper_limit[j] - lower_limit[j]) / (2 * (n_trapz - 1)) * \
                           (y[0, j] + y[-1, j] + 2 * np.sum(y[1:-1, j]))
    # print(np.sum(integrals))
    #assert np.isclose(np.sum(integrals), 1)
    with np.errstate(divide='raise'):
        try:
            return integrals / (np.sum(integrals))
        except FloatingPointError:
            print(integrals)
            print(mean_list)
            print(sigma_list)
            input()



def power(q, p):
    result = 0
    if q != 0:
        result = np.power(q, p)

    return result

def sample_discrete(probabilities):
    v = rand()
    count = 0
    acc = probabilities[0];
    while v > acc:
        count += 1
        acc += probabilities[count]
    return count

def stable_normalizer(x,temp):
    if np.sum(x) == 0:
        x[0] = 1
    ''' Computes x[i]**temp/sum_i(x[i]**temp) '''
    x = (x / np.max(x))**temp
    return np.abs(x/np.sum(x))

def argmax(x):
    ''' assumes a 1D vector x '''
    x = x.flatten()
    if np.any(np.isnan(x)):
        print('Warning: Cannot argmax when vector contains nans, results will be wrong')
    try:
        winners = np.argwhere(x == np.max(x)).flatten()   
        winner = random.choice(winners)
    except:
        winner = np.argmax(x) # numerical instability ? 
    return winner 

def check_space(space):    
    ''' Check the properties of an environment state or action space '''
    if isinstance(space,spaces.Box):
        dim = space.shape
        discrete = False    
    elif isinstance(space,spaces.Discrete):
        dim = space.n
        discrete = True
    else:
        raise NotImplementedError('This type of space is not supported')
    return dim, discrete

def store_safely(folder,name,to_store):
    ''' to prevent losing information due to interruption of process'''
    new_name = folder+'/'+name+'.npy'
    old_name = folder+'/'+name+'_old.npy'
    if os.path.exists(new_name):
        copyfile(new_name,old_name)
    np.save(new_name,to_store)
    if os.path.exists(old_name):            
        os.remove(old_name)

### Atari helpers ###
    
def get_base_env(env):
    ''' removes all wrappers '''
    while hasattr(env,'env'):
        env = env.env
    return env

def copy_atari_state(env):
    env = get_base_env(env)
    return env.clone_full_state()
#    return env.ale.cloneSystemState()

def restore_atari_state(env,snapshot):
    env = get_base_env(env)
    env.restore_full_state(snapshot)
#    env.ale.restoreSystemState(snapshot)

def is_atari_game(env):
    ''' Verify whether game uses the Arcade Learning Environment '''
    env = get_base_env(env)
    return hasattr(env,'ale')

### Database ##
    
class Database():
    ''' Database '''
    
    def __init__(self,max_size,batch_size):
        self.max_size = max_size        
        self.batch_size = batch_size
        self.clear()
        self.sample_array = None
        self.sample_index = 0
    
    def clear(self):
        self.experience = []
        self.insert_index = 0
        self.size = 0
    
    def store(self,experience):
        if self.size < self.max_size:
            self.experience.append(experience)
            self.size +=1
        else:
            self.experience[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def store_from_array(self,*args):
        for i in range(args[0].shape[0]):
            entry = []
            for arg in args:
                entry.append(arg[i])
            self.store(entry)
        
    def reshuffle(self):
        self.sample_array = np.arange(self.size)
        random.shuffle(self.sample_array)
        self.sample_index = 0
                            
    def __iter__(self):
        return self

    def __next__(self):
        if (self.sample_index + self.batch_size > self.size) and (not self.sample_index == 0):
            self.reshuffle() # Reset for the next epoch
            raise(StopIteration)
          
        if (self.sample_index + 2*self.batch_size > self.size):
            indices = self.sample_array[self.sample_index:]
            batch = [self.experience[i] for i in indices]
        else:
            indices = self.sample_array[self.sample_index:self.sample_index+self.batch_size]
            batch = [self.experience[i] for i in indices]
        self.sample_index += self.batch_size
        
        arrays = []
        for i in range(len(batch[0])):
            to_add = np.array([entry[i] for entry in batch])
            arrays.append(to_add) 
        return tuple(arrays)
            
    next = __next__
    
### Visualization ##

def symmetric_remove(x,n):
    ''' removes n items from beginning and end '''
    odd = is_odd(n)
    half = int(n/2)
    if half > 0:
        x = x[half:-half]
    if odd:
        x = x[1:]
    return x

def is_odd(number):
    ''' checks whether number is odd, returns boolean '''
    return bool(number & 1)

def smooth(y,window,mode):
    ''' smooth 1D vectory y '''    
    return np.convolve(y, np.ones(window)/window, mode=mode)