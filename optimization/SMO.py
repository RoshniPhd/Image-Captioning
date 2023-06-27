from random import uniform, random, choice, randint

import numpy as np

from .root import Root
from copy import deepcopy
from numpy import zeros, any, where


class BaseSMO(Root):

    """ Spider Monkey Optimization """

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, batch_size=10, verbose=True,
                 epoch=10, pop_size=50):
        Root.__init__(self, obj_func, lb, ub, problem_size, batch_size, verbose)
        self.epoch = epoch
        self.pop_size = pop_size
        self.LocalLimit = problem_size * pop_size
        self.GlobalLimit = pop_size
        self.member = 10
        self.MG = int(pop_size / self.member)
        self.pr = 0.1
        self.LocalLimitCount = zeros(int(self.pop_size / self.member))
        self.GlobalLimitCount = 0

    def create_group(self, pop, member):
        idx, group = 0, []
        for g in range(self.MG):
            if g != int(self.pop_size / member) - 1:
                group.append(pop[idx:idx + member])
                idx += member
            else:
                group.append(pop[idx:])
        return group

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        groups = self.create_group(pop, self.member)
        l_best = [self.get_global_best_solution(i, self.ID_FIT, self.ID_MIN_PROB) for i in groups]
        for epoch in range(self.epoch):
            # Local
            for id_grp, group in enumerate(groups):
                for id_mky, mky in enumerate(group):
                    pos_new = zeros(self.problem_size)
                    for id_dim, pos in enumerate(mky[self.ID_POS]):
                        if random() >= self.pr:
                            pos_new[id_dim] = pos + random() * (l_best[id_grp][self.ID_POS][id_dim] - pos) + \
                                              uniform(-1, 1) * (choice(group)[self.ID_POS][id_dim] - pos)
                        else:
                            pos_new[id_dim] = pos
                    pos_new = self.amend_position_random_faster(pos_new)
                    fit_new = self.get_fitness_position(pos_new)
                    if fit_new < mky[self.ID_FIT]:
                        groups[id_grp][id_mky] = [pos_new, fit_new]
                    elif l_best[id_grp][self.ID_FIT] == mky[self.ID_FIT]:
                        self.LocalLimitCount[id_grp] += 1

            l_best = [self.update_global_best_solution(i, self.ID_MIN_PROB, l_best[id_]) for id_, i in enumerate(groups)]

            prob = [mky[self.ID_FIT] for group in groups for mky in group]
            prob_sum = sum(prob)
            prob = self.create_group([i / prob_sum for i in prob], self.member)
            # Global
            for id_grp, group in enumerate(groups):
                for id_mky, mky in enumerate(group):
                    if random() < prob[id_grp][id_mky]:
                        id_dim = randint(0, self.problem_size - 1)
                        pos_new = deepcopy(mky[self.ID_POS])
                        pos_new[id_dim] = pos_new[id_dim] + random() * (g_best[self.ID_POS][id_dim] - pos_new[id_dim])\
                                          + uniform(-1, 1) * (choice(group)[self.ID_POS][id_dim] - pos_new[id_dim])
                        pos_new = self.amend_position_random_faster(pos_new)
                        fit_new = self.get_fitness_position(pos_new)
                        if fit_new < mky[self.ID_FIT]:
                            groups[id_grp][id_mky] = [pos_new, fit_new]
                        elif l_best[id_grp][self.ID_FIT] == mky[self.ID_FIT]:
                            self.LocalLimitCount[id_grp] += 1
                        if g_best[self.ID_FIT] == mky[self.ID_FIT]:
                            self.GlobalLimitCount += 1
            # Local and Global position
            l_best = [self.update_global_best_solution(i, self.ID_MIN_PROB, l_best[id_]) for id_, i in enumerate(groups)]
            g_best = self.update_global_best_solution([j for i in groups for j in i], self.ID_MIN_PROB, g_best)
            # Local Decision Phase
            if any(where(self.LocalLimitCount > self.LocalLimit)):
                id_local = where(self.LocalLimitCount > self.LocalLimit)
                self.LocalLimitCount[id_local] = 0
                for local in id_local:
                    for id_mky, mky in enumerate(groups[local]):
                        pos_new = zeros(self.problem_size)
                        for id_dim, pos in enumerate(mky[self.ID_POS]):

                            if random() >= self.pr:
                                pos_new[id_dim] = self.lb[0] + random() * (self.ub[0] - self.lb[0])
                            else:
                                pos_new[id_dim] = pos + random() * (g_best[self.ID_POS][id_dim] - pos) + random() * \
                                                  (pos - choice(l_best)[self.ID_POS][id_dim])
                        pos_new = self.amend_position_random_faster(pos_new)
                        fit_new = self.get_fitness_position(pos_new)
                        if fit_new < mky[self.ID_FIT]:
                            groups[local][id_mky] = [pos_new, fit_new]
                        elif l_best[local][self.ID_FIT] == mky[self.ID_FIT]:
                            self.LocalLimitCount[local] += 1
                        if g_best[self.ID_FIT] == mky[self.ID_FIT]:
                            self.GlobalLimitCount += 1
            if self.GlobalLimitCount > self.GlobalLimit:
                self.GlobalLimitCount = 0
                if len(groups) < self.MG:
                    pop = [j for i in groups for j in i]
                    groups = self.create_group(pop, 4)
                    l_best = [self.get_global_best_solution(i, self.ID_FIT, self.ID_MIN_PROB) for i in groups]
                    # l_best = [self.update_global_best_solution(i, self.ID_MIN_PROB, l_best[id_]) for id_, i in enumerate(groups)]
                else:
                    groups = [[j for i in groups for j in i]]
                    l_best = [self.update_global_best_solution(i, self.ID_MIN_PROB, g_best) for id_, i in enumerate(groups)]
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class HybridSMO(Root):

    """ Spider Monkey Optimization """

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, batch_size=10, verbose=True,
                 epoch=10, pop_size=50):
        Root.__init__(self, obj_func, lb, ub, problem_size, batch_size, verbose)
        self.epoch = epoch
        self.pop_size = pop_size
        self.LocalLimit = problem_size * pop_size
        self.GlobalLimit = pop_size
        self.member = 10
        self.MG = int(pop_size / self.member)
        self.pr = 0.1
        self.LocalLimitCount = zeros(int(self.pop_size / self.member))
        self.GlobalLimitCount = 0

    def create_group(self, pop, member):
        idx, group = 0, []
        for g in range(self.MG):
            if g != int(self.pop_size / member) - 1:
                group.append(pop[idx:idx + member])
                idx += member
            else:
                group.append(pop[idx:])
        return group

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        groups = self.create_group(pop, self.member)
        l_best = [self.get_global_best_solution(i, self.ID_FIT, self.ID_MIN_PROB) for i in groups]
        for epoch in range(self.epoch):
            # Local
            for id_grp, group in enumerate(groups):
                for id_mky, mky in enumerate(group):
                    pos_new = zeros(self.problem_size)
                    for id_dim, pos in enumerate(mky[self.ID_POS]):
                        if random() >= self.pr:
                            w = range(0, 9)
                            pos_new[id_dim] = pos + random() * (l_best[id_grp][self.ID_POS][id_dim] - pos) + \
                                              uniform(-1, 1) * (choice(group)[self.ID_POS][id_dim] - pos) * \
                                              (np.max(w) - epoch / self.epoch * (np.max(w) - np.min(w)))
                        else:
                            pos_new[id_dim] = pos
                    pos_new = self.amend_position_random_faster(pos_new)
                    fit_new = self.get_fitness_position(pos_new)
                    if fit_new < mky[self.ID_FIT]:
                        groups[id_grp][id_mky] = [pos_new, fit_new]
                    elif l_best[id_grp][self.ID_FIT] == mky[self.ID_FIT]:
                        self.LocalLimitCount[id_grp] += 1

            l_best = [self.update_global_best_solution(i, self.ID_MIN_PROB, l_best[id_]) for id_, i in enumerate(groups)]

            prob = [mky[self.ID_FIT] for group in groups for mky in group]
            prob_sum = sum(prob)
            prob = self.create_group([i / prob_sum for i in prob], self.member)
            # Global
            steps = 5
            X = np.zeros(steps + 1)
            X[0] = 1

            def logistic_map(x):
                x_next = x + 0.0000001
                return x_next

            # map the equation to array step by step using the logistic_map function above
            for i in range(steps):
                x_next = logistic_map(X[i])  # calls the logistic_map function on X[i] as x and Y[i] as y
                X[i + 1] = x_next
            for id_grp, group in enumerate(groups):
                for id_mky, mky in enumerate(group):
                    if random() < prob[id_grp][id_mky]:
                        id_dim = randint(0, self.problem_size - 1)
                        pos_new = deepcopy(mky[self.ID_POS])
                        pos_new[id_dim] = pos_new[id_dim] + (np.sin(random())) - (x_next) * (g_best[self.ID_POS][id_dim] - pos_new[id_dim])\
                                          + (uniform(-1, 1)) - (x_next) * (choice(group)[self.ID_POS][id_dim] - pos_new[id_dim])
                        pos_new = self.amend_position_random_faster(pos_new)
                        fit_new = self.get_fitness_position(pos_new)
                        if fit_new < mky[self.ID_FIT]:
                            groups[id_grp][id_mky] = [pos_new, fit_new]
                        elif l_best[id_grp][self.ID_FIT] == mky[self.ID_FIT]:
                            self.LocalLimitCount[id_grp] += 1
                        if g_best[self.ID_FIT] == mky[self.ID_FIT]:
                            self.GlobalLimitCount += 1
            # Local and Global position
            l_best = [self.update_global_best_solution(i, self.ID_MIN_PROB, l_best[id_]) for id_, i in enumerate(groups)]
            g_best = self.update_global_best_solution([j for i in groups for j in i], self.ID_MIN_PROB, g_best)
            # Local Decision Phase
            if any(where(self.LocalLimitCount > self.LocalLimit)):
                id_local = where(self.LocalLimitCount > self.LocalLimit)
                self.LocalLimitCount[id_local] = 0
                for local in id_local:
                    for id_mky, mky in enumerate(groups[local]):
                        pos_new = zeros(self.problem_size)
                        for id_dim, pos in enumerate(mky[self.ID_POS]):

                            if random() >= self.pr:
                                pos_new[id_dim] = self.lb[0] + random() * (self.ub[0] - self.lb[0])
                            else:
                                pos_new[id_dim] = pos + random() * (g_best[self.ID_POS][id_dim] - pos) + random() * \
                                                  (pos - choice(l_best)[self.ID_POS][id_dim])
                        pos_new = self.amend_position_random_faster(pos_new)
                        fit_new = self.get_fitness_position(pos_new)
                        if fit_new < mky[self.ID_FIT]:
                            groups[local][id_mky] = [pos_new, fit_new]
                        elif l_best[local][self.ID_FIT] == mky[self.ID_FIT]:
                            self.LocalLimitCount[local] += 1
                        if g_best[self.ID_FIT] == mky[self.ID_FIT]:
                            self.GlobalLimitCount += 1
            if self.GlobalLimitCount > self.GlobalLimit:
                self.GlobalLimitCount = 0
                if len(groups) < self.MG:
                    pop = [j for i in groups for j in i]
                    groups = self.create_group(pop, 4)
                    l_best = [self.get_global_best_solution(i, self.ID_FIT, self.ID_MIN_PROB) for i in groups]
                    # l_best = [self.update_global_best_solution(i, self.ID_MIN_PROB, l_best[id_]) for id_, i in enumerate(groups)]
                else:
                    groups = [[j for i in groups for j in i]]
                    l_best = [self.update_global_best_solution(i, self.ID_MIN_PROB, g_best) for id_, i in enumerate(groups)]
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


# from opfunu.cec_basic.cec2014_nobias import *
#
# ## Setting parameters
# obj_func = F5
# verbose = True
# epoch = 10
# pop_size = 50
# problem_size = 50
# batch_size = 10
# # A - Different way to provide lower bound and upper bound. Here are some examples:
#
# ## 1. When you have different lower bound and upper bound for each parameters
# lb = [-3, -5, 1]
# ub = [5, 10, 100]
#
# md1 = BaseSMO(obj_func, lb, ub, problem_size, batch_size, verbose)
# best_pos1, best_fit1, list_loss1 = md1.train()
# print(best_fit1)
