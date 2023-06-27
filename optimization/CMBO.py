#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:06, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
# -------------------------------------------------------------------------------------------------------%
import numpy
from numpy.random import uniform, rand
from numpy import abs, exp, cos, pi, sign, round
from .root import Root
from numpy import where, clip, logical_and, maximum, minimum, power, sin, abs, pi, sqrt, sign, ones, ptp, min, sum, array, ceil, multiply, mean
from numpy.random import uniform, random, normal, choice
from math import gamma

class BaseCMBO(Root):
    """
        The original version of: Cat Mice based Optimization Algorithm
            - In this algorithms: Prey means the best position
    """

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, batch_size=10, verbose=True, epoch=10,
                 pop_size=50):
        Root.__init__(self, obj_func, lb, ub, problem_size, batch_size, verbose)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            half_loc = int(self.pop_size / 2)
            for i in range(self.pop_size):
                l = round(1 + rand(1))


                if i < half_loc:
                    new_position = pop[i + half_loc][self.ID_POS] + rand(1) * (
                            pop[i][self.ID_POS] - l * pop[i + half_loc][self.ID_POS])
                else:
                    new_position = pop[i - half_loc][self.ID_POS] + rand(1) * (
                            g_best[self.ID_POS] - l * pop[i - half_loc][self.ID_POS]) * \
                                   sign(pop[i - half_loc][self.ID_FIT] - g_best[self.ID_FIT])

                new_position = self.amend_position_faster(new_position)
                fit = self.get_fitness_position(new_position)
                if i < half_loc:
                    pop[i + half_loc] = [new_position, fit]
                else:
                    pop[i - half_loc] = [new_position, fit]

                # batch size idea
                if i % self.batch_size:
                    g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_POS)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train





