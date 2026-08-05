import random

import numpy as np
from scipy.special import gamma
from math import pi


class SBOA:

    # Secretary Bird Optimization Algorithm(SBOA)

    def __init__(self, weight, fitness_function, epochs, pop_size):
        self.weight = weight
        self.fitness_function = fitness_function
        self.epochs = epochs
        self.pop_size = pop_size

    @staticmethod
    def levy(dim):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)

        step = u / np.abs(v) ** (1 / beta)

        return step

    def solve(self):
        lb = np.array(sum((self.weight - 0.01).tolist(), []))
        ub = np.array(sum((self.weight + 0.01).tolist(), []))

        dim = len(lb)

        X = lb + np.random.rand(self.pop_size, dim) * (ub - lb)
        fit = np.zeros(self.pop_size)

        for i in range(self.pop_size):
            M = X[i, :]
            fit[i] = self.fitness_function(M)

        # Main loop
        fbest = np.inf
        Bast_P = None
        best_so_far = np.zeros(self.epochs)

        for t in range(self.epochs):
            CF = (1 - t / self.epochs) ** (2 * t / self.epochs)

            # Update the global best (fbest)
            best = np.min(fit)
            location = np.argmin(fit)

            if t == 0:
                Bast_P = X[location, :]
                fbest = best
            elif best < fbest:
                fbest = best
                Bast_P = X[location, :]

            # Secretary bird's predation strategy
            for i in range(self.pop_size):
                if t < self.epochs / 3:  # Secretary bird search prey stage
                    Rn = X.shape[0]
                    X_random_1 = np.random.randint(0, Rn)
                    X_random_2 = np.random.randint(0, Rn)
                    R1 = np.random.rand(dim)
                    X1 = X[i, :] + (X[X_random_1, :] - X[X_random_2, :]) * R1
                    X1 = np.clip(X1, lb, ub)
                elif self.epochs / 3 <= t < 2 * self.epochs / 3:  # Secretary bird approaching prey stage
                    RB = np.random.randn(dim)
                    X1 = Bast_P + np.exp((t / self.epochs) ** 4) * (RB - 0.5) * (Bast_P - X[i, :])
                    X1 = np.clip(X1, lb, ub)
                else:  # Secretary bird attacks prey stage
                    RL = 0.5 * self.levy(dim)
                    X1 = Bast_P + CF * X[i, :] * RL
                    X1 = np.clip(X1, lb, ub)

                f_newP1 = self.fitness_function(X1)
                if f_newP1 <= fit[i]:
                    X[i, :] = X1
                    fit[i] = f_newP1

            # Secretary Bird's escape strategy
            r = np.random.rand()
            k = np.random.choice(self.pop_size)
            Xrandom = X[k, :]

            for i in range(self.pop_size):
                if r < 0.5:
                    # C1: Secretary birds use their environment to hide from predators
                    RB = np.random.rand(dim)
                    X2 = Bast_P + (1 - t / self.epochs) ** 2 * (2 * RB - 1) * X[i, :]
                    X2 = np.clip(X2, lb, ub)
                else:
                    # C2: Secretary birds fly or run away from the predator
                    K = np.round(1 + np.random.rand())
                    R2 = np.random.rand(dim)
                    X2 = X[i, :] + R2 * (Xrandom - K * X[i, :])
                    X2 = np.clip(X2, lb, ub)

                f_newP2 = self.fitness_function(X2)
                if f_newP2 <= fit[i]:
                    X[i, :] = X2
                    fit[i] = f_newP2

            best_so_far[t] = fbest
            average = np.mean(fit)

        return fbest, Bast_P, best_so_far
