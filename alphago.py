#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-player Alpha Go
"""
from helpers import (argmax, is_atari_game, copy_atari_state, store_safely,
                     restore_atari_state, stable_normalizer, power, compute_prob_max)
from set_weights import set_weights
from dqn_net import Network
import numpy as np
from numpy.random import seed
import argparse
import os
import time
import copy
import matplotlib.pyplot as plt
import torch
from atari import Atari
from scipy.special import softmax
from numpy.random import rand
from scipy.special import logsumexp
plt.style.use('ggplot')


def alien(Env, cuda):
    model = torch.load('models/alien.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def amidar(Env, cuda):
    model = torch.load('models/amidar.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def asteroids(Env, cuda):
    model = torch.load('models/asteroids.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def bank_heist(Env, cuda):
    model = torch.load('models/bank_heist.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def bowling(Env, cuda):
    model = torch.load('models/bowling.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def centipede(Env, cuda):
    model = torch.load('models/centipede.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def demon_attack(Env, cuda):
    model = torch.load('models/demon_attack.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def gopher(Env, cuda):
    model = torch.load('models/gopher.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def krull(Env, cuda):
    model = torch.load('models/krull.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def phoenix(Env, cuda):
    model = torch.load('models/phoenix.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def robotank(Env, cuda):
    model = torch.load('models/robotank.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def wizard_of_wor(Env, cuda):
    model = torch.load('models/wizard_of_wor.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def atlantis(Env, cuda):
    model = torch.load('models/atlantis.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def enduro(Env, cuda):
    model = torch.load('models/enduro.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def freeway(Env, cuda):
    model = torch.load('models/freeway.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def frostbite(Env, cuda):
    model = torch.load('models/frostbite.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def hero(Env, cuda):
    model = torch.load('models/hero.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def ms_pacman(Env, cuda):
    model = torch.load('models/ms_pacman.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def nets(Env, cuda):
    model = torch.load('models/nets.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def pitfall(Env, cuda):
    model = torch.load('models/pitfall.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def solaris(Env, cuda):
    model = torch.load('models/solaris.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def breakout(Env, cuda):
    model = torch.load('models/breakout.torch', map_location=torch.device('cuda') if cuda else torch.device('cpu'))
    return model

def asterix(Env, cuda=True):
    model = Network(input_shape=(4, 84, 84), output_shape=(Env.info.action_space.n,))
    weights = np.load('models/asterix.npy')
    set_weights(model.parameters(), weights, cuda)
    return model

def beam_rider(Env, cuda=True):
    model = Network(input_shape=(4, 84, 84), output_shape=(Env.info.action_space.n,))
    weights = np.load('models/beam-rider.npy')
    set_weights(model.parameters(), weights, cuda)
    return model

def qbert(Env, cuda=True):
    model = Network(input_shape=(4, 84, 84), output_shape=(Env.info.action_space.n,))
    weights = np.load('models/qbert.npy')
    set_weights(model.parameters(), weights, cuda)
    return model

def space_invaders(Env, cuda=True):
    model = Network(input_shape=(4, 84, 84), output_shape=(Env.info.action_space.n,))
    weights = np.load('models/space-invaders.npy')
    set_weights(model.parameters(), weights, cuda)
    return model

def seaquest(Env, cuda=True):
    model = Network(input_shape=(4, 84, 84), output_shape=(Env.info.action_space.n,))
    weights = np.load('models/seaquest.npy')
    set_weights(model.parameters(), weights, cuda)
    return model


def load_atari_game(argument, Env, cuda):
    switcher = {
        'AlienNoFrameskip-v4': alien,
        'AmidarNoFrameskip-v4': amidar,
        'AsteroidsNoFrameskip-v4': asteroids,
        'BankHeistNoFrameskip-v4': bank_heist,
        'BowlingNoFrameskip-v4': bowling,
        'CentipedeNoFrameskip-v4': centipede,
        'DemonAttackNoFrameskip-v4': demon_attack,
        'GopherNoFrameskip-v4': gopher,
        'KrullNoFrameskip-v4': krull,
        'PhoenixNoFrameskip-v4': phoenix,
        'RobotankNoFrameskip-v4': robotank,
        'WizardOfWorNoFrameskip-v4': wizard_of_wor,
        'AtlantisNoFrameskip-v4': atlantis,
        'EnduroNoFrameskip-v4': enduro,
        'FreewayNoFrameskip-v4': freeway,
        'FrostbiteNoFrameskip-v4': frostbite,
        'HeroNoFrameskip-v4': hero,
        'MsPacmanNoFrameskip-v4': ms_pacman,
        'NetsNoFrameskip-v4': nets,
        'PitfallNoFrameskip-v4': pitfall,
        'SolarisNoFrameskip-v4': solaris,
        'BreakoutNoFrameskip-v4': breakout,
        'AsterixNoFrameskip-v4': asterix,
        'BeamRiderNoFrameskip-v4': beam_rider,
        'QbertNoFrameskip-v4': qbert,
        'SeaquestNoFrameskip-v4': seaquest,
        'SpaceInvadersNoFrameskip-v4': space_invaders
    }
    # Get the function from switcher dictionary
    func = switcher.get(argument, lambda: "Invalid game")
    # Execute the function
    model = func(Env, cuda)
    return model

##### MCTS functions #####
class Action():
    ''' Action object '''
    def __init__(self, index, parent_state, Q_init=0.0, tau=1.0, epsilon=0, lambda_const=0, algorithm='uct', p=1.0,
                 step_size=0., std=0):
        self.index = index #action index
        self.child_state = None
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 1 #visitation count
        self.Q = Q_init  # in w-mcts, self.Q is the mean of Q vectors
        self.Q_init = Q_init
        self.Q_std = std  # standard deviation
        self.tau = tau
        self.epsilon = epsilon # exploration
        self.lambda_const = lambda_const
        self.algorithm = algorithm
        self.p = p #power mean
        self._step_size = step_size

    def add_child_state(self, s1, r, terminal, model):
        self.child_state = State(s1, r, terminal, self, self.parent_state.na, model, tau=self.tau, epsilon=self.epsilon,
                                 lambda_const=self.lambda_const, algorithm=self.algorithm, p=self.p, std=0, gamma=self.parent_state.gamma)
        return self.child_state

    def update(self, R, std=0):
        self.n += 1
        _step_size = 1./np.power(self.n, self._step_size)
        if self.algorithm == 'uct' or self.algorithm == 'power-uct':
            self.W += R
            self.Q = self.W/self.n
        elif self.algorithm == 'maxmcts':
            self.Q = R
            delta = R - self.Q
            self.Q += delta / self.n
        elif self.algorithm == 'w-mcts':
            self.Q = _step_size * self.Q + (1 - _step_size) * R
            self.Q_std = _step_size * self.Q_std + (1 - _step_size) * std
        elif self.algorithm == 'dng':
            self.Q = R
        else:
            self.Q = R

class State():
    ''' State object '''

    def __init__(self, index, r, terminal, parent_action, na, model, tau, epsilon, lambda_const, algorithm, p, std=0, gamma=0.99):
        ''' Initialize a new state '''
        self.index = np.array(index)  # state
        self.r = r  # reward upon arriving in this state
        self.terminal = terminal  # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 1
        self.model = model
        self.tau = tau
        self.epsilon = epsilon
        self.lambda_const = lambda_const
        self.na = na
        self.algorithm = algorithm
        self.p = p
        self.gamma = gamma

        self.V_std = std

        # DNG
        self.alpha = 1
        self.beta = 100
        self.mu = 0 # mean of the value distribution, assign this to self.V
        self.ll = 0.01

        self.evaluate()

    def select(self, c=1.5):
        ''' Select one of the child actions based on uct rule '''
        winner = 0
        if self.algorithm == 'uct' or self.algorithm == 'power-uct' or self.algorithm == 'maxmcts':
            uct = np.array([child_action.Q + prior * c * (np.sqrt(self.n + 1)/(child_action.n + 1)) for child_action, prior
                            in zip(self.child_actions, self.priors)])
            winner = np.argmax(uct)
        elif self.algorithm == 'rents':
            random = rand()
            Qs = [child_action.Q for child_action in self.child_actions]
            max_Q = np.max(Qs)
            UCT = []
            if np.sum(self.priors) == 0:
                UCT = [np.exp((child_action.Q - max_Q) / self.tau) for child_action in self.child_actions]
            else:
                UCT = [prior * np.exp((child_action.Q - max_Q) / self.tau) for child_action, prior in
                       zip(self.child_actions, self.priors)]

            UCT = np.squeeze(UCT)
            UCT = UCT / np.sum(UCT)

            para_lambda = self.epsilon * (self.na / np.log(self.n + 1))
            winner = 0.0
            if random > para_lambda:
                winner = np.random.choice(len(self.child_actions), p=UCT)
            else:
                winner = np.random.randint(self.na)
        elif self.algorithm == 'ments':
            random = rand()
            Qs = [child_action.Q for child_action in self.child_actions]
            max_Q = np.max(Qs)
            uct = [np.exp((child_action.Q - max_Q) / self.tau) for child_action in self.child_actions]
            uct = np.squeeze(uct)
            uct = uct / np.sum(uct)
            para_lambda = self.epsilon * (self.na / np.log(self.n + 1))
            if random > para_lambda:
                winner = np.random.choice(len(self.child_actions), p=uct)
            else:
                winner = np.random.randint(self.na)
        elif self.algorithm == 'tsallis':
            Qs = [child_action.Q/self.tau for child_action in self.child_actions]
            Qs_sorted = np.sort(Qs)[::-1]
            Qs_cumsum = np.cumsum(Qs_sorted)

            K = np.arange(1, self.na + 1)
            Q_check = 1 + K * Qs_sorted > Qs_cumsum
            Q_check = Q_check.astype(int)

            K_sum = np.sum(Q_check)
            Q_sp_max = [q * check for q, check in zip(Qs_sorted, Q_check)]
            sp_max = (np.sum(Q_sp_max) - 1) / K_sum

            pi = [np.maximum(Q - sp_max, 0) for Q in Qs]
            pi = pi/np.sum(pi)  # normalize

            random = rand()
            para_lambda = self.epsilon * (self.na / np.log(self.n + 1))
            if random > para_lambda:
                winner = np.random.choice(len(self.child_actions), p=pi)
            else:
                winner = np.random.randint(self.na)
        elif self.algorithm == 'w-mcts':
            Q = [child_action.Q for child_action in self.child_actions]
            std_next_all = [child_action.Q_std for child_action in self.child_actions]
            mean_next_all = np.array(Q)
            std_next_all = np.array(std_next_all)
            prob = compute_prob_max(mean_next_all, std_next_all)
            chosen_action = np.random.choice(np.size(prob), 1, p=prob)
            winner = int(chosen_action[0])
        elif self.algorithm == 'dng':
            q_values = []
            for action in self.child_actions:
                if action.child_state is None:
                    q_values.append(action.Q_init)
                    continue

                alpha = action.child_state.alpha
                beta = action.child_state.beta
                mu = action.child_state.mu
                ll = action.child_state.ll

                tau = np.random.gamma(alpha, 1/beta)
                x = np.random.normal(mu, np.sqrt(1/(ll*tau)))

                q_values.append(action.Q + self.gamma * x)

            winner = np.argmax(q_values)

        return self.child_actions[winner]

    def evaluate(self):
        ''' Bootstrap the state value '''
        # self.V = np.squeeze(self.model.predict_V(self.index[None,])) if not self.terminal else np.array(0.0)
        self.index = np.expand_dims(self.index, 0)
        self.index = torch.from_numpy(self.index)
        if args.cuda:
            self.index = self.index.cuda()
        self.Q = self.model(self.index)

        Q_value = np.squeeze(self.Q.detach().cpu().numpy())
        if self.algorithm == 'rents':
            max_Q = np.max(self.Q.detach().cpu().numpy())
            uct = np.exp((self.Q.detach().cpu().numpy() - max_Q) / self.tau)
            self.priors = softmax(self.Q.detach().cpu().numpy() / self.tau)
            self.priors = np.squeeze(self.priors)

            self.index = self.index.detach().cpu()
            for a in range(self.na):
                if self.priors[a] == 0:
                    self.priors[a] = 1
            self.V = max_Q + self.tau * np.log(np.mean(uct))

            self.child_actions = [
                Action(a, parent_state=self, Q_init=np.log(self.priors[a]) + (Q_value[a] - self.V)/self.tau, tau=self.tau,
                       epsilon=self.epsilon, lambda_const=self.lambda_const, algorithm=self.algorithm, p=self.p) for a in range(self.na)]
        elif self.algorithm == 'ments':
            max_Q = np.max(self.Q.detach().cpu().numpy())
            uct = np.exp((self.Q.detach().cpu().numpy() - max_Q) / self.tau)
            self.priors = softmax(self.Q.detach().cpu().numpy() / self.tau)
            self.priors = np.squeeze(self.priors)

            self.index = self.index.detach().cpu()
            self.V = max_Q + self.tau * np.log(np.sum(uct))
            self.child_actions = [
                Action(a, parent_state=self, Q_init=(Q_value[a] - self.V) / self.tau, tau=self.tau, epsilon=self.epsilon,
                       lambda_const=self.lambda_const, algorithm=self.algorithm, p=self.p) for a in range(self.na)]
        elif self.algorithm == 'tsallis':
            Q_value = Q_value/self.tau
            Qs = np.squeeze(Q_value)
            Qs_sorted = np.sort(Qs)[::-1]
            Qs_cumsum = np.cumsum(Qs_sorted)

            K = np.arange(1, self.na + 1)
            Q_check = 1 + K * Qs_sorted > Qs_cumsum
            Q_check = Q_check.astype(int)

            K_sum = np.sum(Q_check)
            Q_sp_max = [q * check for q, check in zip(Qs_sorted, Q_check)]
            sp_max = (np.sum(Q_sp_max) - 1) / K_sum

            self.priors = [np.maximum(Q - sp_max, 0) for Q in Qs]

            self.V = 0.0
            second = sp_max * sp_max
            for Q in Q_sp_max:
                if Q == 0:
                    break
                first = Q * Q
                self.V += first - second

            self.V = self.tau * (0.5 * self.V + 0.5)

            self.index = self.index.detach().cpu()
            Q_value = np.squeeze(self.Q.detach().cpu().numpy())

            self.child_actions = [
                Action(a, parent_state=self, Q_init=(Q_value[a] - self.V)/self.tau, tau=self.tau, epsilon=self.epsilon,
                       lambda_const=self.lambda_const, algorithm=self.algorithm, p=self.p) for a in range(self.na)]
        else:
            self.V = np.mean(self.Q.detach().cpu().numpy())
            self.V_std = np.std(self.Q.detach().cpu().numpy())
            self.priors = softmax(self.Q.detach().cpu().numpy() / self.tau)
            self.index = self.index.detach().cpu()
            if self.algorithm == 'dng':
                self.child_actions = [
                    Action(a, parent_state=self, Q_init=self.V, tau=self.tau, epsilon=self.epsilon,
                           lambda_const=self.lambda_const, algorithm=self.algorithm, p=self.p, std=0) for a in range(self.na)]
            else:
                self.child_actions = [
                    Action(a, parent_state=self, Q_init=Q_value[a], tau=self.tau, epsilon=self.epsilon,
                           lambda_const=self.lambda_const, algorithm=self.algorithm, p=self.p, std=0) for a in range(self.na)]

    def update(self, R, update_type='mean'):
        ''' update count on backward pass '''
        self.n += 1
        Q = [child_action.Q for child_action in self.child_actions]
        max_Q = np.max(Q)
        counts = [prior for prior in self.priors]
        if self.algorithm == 'rents':
            if np.sum(counts) == 0:
                self.V = max_Q + self.tau * logsumexp((Q - max_Q) / self.tau)
            else:
                sum = np.sum(counts)
                self.V = max_Q + self.tau * np.log(np.sum((counts/sum)*np.exp((Q - max_Q)/self.tau)))
        elif self.algorithm == 'ments':
            max_Q = np.max(Q)
            MENT = [np.exp((child_action.Q - max_Q) / self.tau) for child_action in self.child_actions]
            self.V = max_Q + self.tau * np.log(np.sum(MENT))
        elif self.algorithm == 'uct' or self.algorithm == 'maxmcts':
            counts = np.array([child_action.n for child_action in self.child_actions])
            self.V = np.sum((counts / np.sum(counts)) * Q)
        elif self.algorithm == 'power-uct':
            self.V = 0.0
            counts = np.array([child_action.n for child_action in self.child_actions])
            for child_action in self.child_actions:
                self.V += (child_action.n / np.sum(counts)) * power(child_action.Q, self.p)
            self.V = power(self.V, 1 / self.p)
        elif self.algorithm == 'tsallis':
            Qs = [child_action.Q/self.tau for child_action in self.child_actions]
            Qs_sorted = np.sort(Qs)[::-1]
            Qs_cumsum = np.cumsum(Qs_sorted)

            K = np.arange(1, self.na + 1)
            Q_check = 1 + K * Qs_sorted > Qs_cumsum
            Q_check = Q_check.astype(int)

            K_sum = np.sum(Q_check)
            Q_sp_max = [q * check for q, check in zip(Qs_sorted, Q_check)]
            sp_max = (np.sum(Q_sp_max) - 1) / K_sum

            self.V = 0.0
            second = (sp_max * sp_max)
            for Q in Q_sp_max:
                if (Q == 0):
                    break
                first = Q * Q
                self.V += first - second

            self.V = self.tau * (0.5 * self.V + 0.5)
        elif self.algorithm == 'w-mcts':
            std_next_all = [child_action.Q_std for child_action in self.child_actions]
            mean_next_all = np.array(Q)
            std_next_all = np.array(std_next_all)

            if update_type == 'max':
                best = np.random.choice(np.argwhere(mean_next_all == np.max(mean_next_all)).ravel())
                self.V = mean_next_all[best]
                self.V_std = std_next_all[best]
            else:
                prob = compute_prob_max(mean_next_all, std_next_all)
                self.V = np.sum(mean_next_all * prob)
                self.V_std = np.sum(std_next_all * prob)
        elif self.algorithm == 'dng':
            self.alpha = self.alpha + 0.5
            self.beta = self.beta + .5 * (self.ll*(R - self.mu)**2 / (self.ll + 1))
            self.mu = (self.ll*self.mu + R)/(self.ll + 1)
            self.ll = self.ll + 1

            self.V = self.mu


class MCTS():
    ''' MCTS object '''

    def __init__(self, root, root_index, model, na, gamma, tau, epsilon, lambda_const, algorithm, p):
        self.root = None
        self.root_index = root_index
        self.model = model
        self.na = na
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.lambda_const = lambda_const
        self.algorithm = algorithm
        self.p = p

    def search(self, n_mcts, c, Env, mcts_env):
        ''' Perform the MCTS search from the root '''
        if self.root is None:
            self.root = State(self.root_index, r=0.0, terminal=False, parent_action=None, na=self.na, model=self.model,
                              tau=self.tau, epsilon=self.epsilon, lambda_const=self.lambda_const, algorithm=self.algorithm,
                              p=self.p, gamma=self.gamma)  # initialize new root
        else:
            self.root.parent_action = None  # continue from current root
        if self.root.terminal:
            return

        is_atari = is_atari_game(Env)
        if is_atari:
            snapshot = copy_atari_state(Env)  # for Atari: snapshot the root at the beginning

        for i in range(n_mcts):
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
            else:
                restore_atari_state(mcts_env, snapshot)

            while not state.terminal:
                action = state.select(c=c)
                s1, r, t, _ = mcts_env.step(action.index)
                r = np.clip(r, 0, 1)

                if action.child_state is not None:
                    state = action.child_state  # select
                    continue
                else:
                    state = action.add_child_state(s1, r, t, self.model)  # expand
                    break

            # Back-up
            R = state.V
            while state.parent_action is not None:  # loop back-up until root is reached
                discount_v_std = 0
                if self.algorithm == 'rents' or self.algorithm == 'ments' or self.algorithm == 'tsallis' \
                        or self.algorithm == 'w-mcts' or self.algorithm == 'dng':
                    R = state.V
                    discount_v_std = self.gamma * state.V_std

                R = state.r + self.gamma * R
                action = state.parent_action
                action.update(R, std=discount_v_std)
                state = action.parent_state
                if self.algorithm == 'maxmcts':
                    Q = [child_action.Q for child_action in state.child_actions]
                    a = np.argmax(Q)
                    R = (1 - self.lambda_const) * Q[a] + self.lambda_const * R
                state.update(R)

    def return_results(self, temp):
        ''' Process the output at the root node '''
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        pi_target = stable_normalizer(counts, temp)
        Q = np.array([child_action.Q for child_action in self.root.child_actions])

        V_target = 0

        if self.algorithm == 'w-mcts':
            std_next_all = [child_action.Q_std for child_action in self.root.child_actions]
            mean_next_all = np.array(Q)
            std_next_all = np.array(std_next_all)
            pi_target = compute_prob_max(mean_next_all, std_next_all)

        if self.algorithm == 'rents':
            max_Q = np.max(Q)
            if np.sum(self.root.priors) == 0:
                Q = [np.exp((child_action.Q - max_Q) / self.tau) for child_action in self.root.child_actions]
            else:
                Q = [(prior * np.exp((child_action.Q - max_Q) / self.tau)) for child_action, prior in
                     zip(self.root.child_actions, self.root.priors)]
            pi_target = Q / np.sum(Q)
        elif self.algorithm == 'ments':
            max_Q = np.max(Q)
            Q = [np.exp((child_action.Q - max_Q) / self.tau) for child_action in
                 self.root.child_actions]
            pi_target = Q / np.sum(Q)
        elif self.algorithm == 'tsallis':
            Qs = [child_action.Q/self.tau for child_action in self.root.child_actions]
            Qs_sorted = np.sort(Qs)[::-1]
            Qs_cumsum = np.cumsum(Qs_sorted)

            K = np.arange(1, self.na + 1)
            Q_check = 1 + K * Qs_sorted > Qs_cumsum
            Q_check = Q_check.astype(int)

            K_sum = np.sum(Q_check)
            Q_sp_max = [q * check for q, check in zip(Qs_sorted, Q_check)]
            sp_max = (np.sum(Q_sp_max) - 1) / K_sum
            pi_target = [np.maximum(Q - sp_max, 0) for Q in Qs]
            pi_target = pi_target / np.sum(pi_target)
        elif self.algorithm == 'uct' or self.algorithm == 'power-uct' or self.algorithm == 'maxmcts':
            # or self.algorithm == 'w-mcts':
            pi_target = stable_normalizer(counts, temp)
        elif self.algorithm == 'dng':
            q_values = [child_action.Q for child_action in self.root.child_actions]
            # q_values = []
            # for action in self.root.child_actions:
            #     if action.child_state is None:
            #         q_values.append(action.Q_init)
            #         continue
            #     alpha = action.child_state.alpha
            #     beta = action.child_state.beta
            #     mu = action.child_state.mu
            #     ll = action.child_state.ll
            #
            #     tau = np.random.gamma(alpha, 1/beta)
            #     x = np.random.normal(mu, np.sqrt(1/(ll*tau)))
            #
            #     q_values.append(action.Q + self.gamma * x)

            pi_target = q_values/sum(q_values)

        return self.root.index, pi_target, V_target

    def forward(self, a, s1):
        s1 = np.array(s1)
        ''' Move the root forward '''
        if self.root.child_actions[a].child_state is None:
            self.root = None
            self.root_index = s1
        elif np.linalg.norm(self.root.child_actions[a].child_state.index - s1) > 0.01:
            self.root = None
            self.root_index = s1
        else:
            self.root = self.root.child_actions[a].child_state

#### Agent ##


def agent(algorithm, game, n_ep, n_mcts, max_ep_len, c, p, gamma, temp, tau, epsilon, lambda_const):
    ''' Outer training loop '''
    # tf.reset_default_graph()
    episode_returns = []  # storage
    timepoints = []
    # Environments
    Env = Atari(game)
    is_atari = is_atari_game(Env)
    mcts_env = Atari(game)

    model = load_atari_game(game, Env, args.cuda)

    t_total = 0  # total steps
    R_best = -np.Inf

    for ep in range(n_ep):
        start = time.time()
        s = Env.reset()
        R = 0.0  # Total return counter
        a_store = []
        seed = np.random.randint(1e7)  # draw some Env seed
        Env.seed(seed)
        if is_atari:
            mcts_env.reset()
            mcts_env.seed(seed)

        mcts = MCTS(root_index=s, root=None, model=model, na=Env.info.action_space.n,
                    gamma=gamma, tau=tau, epsilon=epsilon, lambda_const=lambda_const, algorithm=algorithm, p=p)  # the object responsible for MCTS searches
        for t in range(max_ep_len):
            # MCTS step
            mcts.search(n_mcts=n_mcts, c=c, Env=Env, mcts_env=mcts_env)  # perform a forward search
            state, pi, V = mcts.return_results(temp)  # extract the root output

            # Make the true step
            if algorithm == 'uct' or algorithm == 'power-uct' or algorithm == 'maxmcts' or algorithm == 'w-mcts':
                chosen_action = np.random.choice(np.size(pi), 1, p=pi)
                a = int(chosen_action[0])
            elif algorithm == 'dng':
                a = np.random.choice(np.argwhere(pi == np.max(pi)).ravel())
            else:
                a = np.argmax(pi)

            a_store.append(a)
            s1, r, terminal, _ = Env.step(a)
            R += r
            t_total += n_mcts  # total number of environment steps (counts the mcts steps)

            if terminal:
                break
            else:
                mcts.forward(a, s1)

        # Finished episode
        episode_returns.append(R)  # store the total episode return
        timepoints.append(t_total)  # store the timestep count of the episode return
        store_safely(os.getcwd(), 'result', {'R': episode_returns, 't': timepoints})

        if R > R_best:
            a_best = a_store
            seed_best = seed
            R_best = R
        print('Finished episode {}, total return: {}, total time: {} sec'.format(ep, np.round(R, 2),
                                                                                 np.round((time.time() - start), 1)))

    # Return results
    return episode_returns, timepoints, a_best, seed_best, R_best

#### Command line call, parsing and plotting ##


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='uct', help='uct/power-uct/maxmcts/rents/ments/tsallis/w-mcts/dng')
    parser.add_argument('--game', default='BreakoutNoFrameskip-v4', help='Training environment')
    parser.add_argument('--n_ep', type=int, default=100, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=512, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=2000, help='Maximum number of steps per episode')
    parser.add_argument('--c', type=float, default=1.5, help='uct constant')
    parser.add_argument('--p', type=float, default=1.0, help='Power constant')
    parser.add_argument('--tau', type=float, default=1.0, help='Tau')
    parser.add_argument('--epsilon', type=float, default=.0, help='Epsilon')
    parser.add_argument('--lambda_const', type=float, default=.2, help='Lambda const')
    parser.add_argument('--step_size', type=float, default=1., help='Step Size')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount parameter')
    parser.add_argument('--number', type=int, default=1, help='Iteration number')
    parser.add_argument('--cuda', default=True, action='store_true')

    args = parser.parse_args()
    episode_returns, timepoints, a_best, seed_best, R_best = agent(algorithm=args.algorithm, game=args.game, n_ep=args.n_ep, n_mcts=args.n_mcts,
                                                                   max_ep_len=args.max_ep_len, c=args.c, p=args.p, gamma=args.gamma,
                                                                   temp=args.temp, tau=args.tau, epsilon=args.epsilon, lambda_const=args.lambda_const)

    filename = os.getcwd() + '/logs/' + args.game + '_' + args.algorithm + '.txt' + str(args.tau) + '_' + \
               str(args.epsilon) + '_' + str(args.number)
    file = open(filename, "w+")

    for reward in episode_returns:
        file.write(str(reward) + "\n")

    file.close()
