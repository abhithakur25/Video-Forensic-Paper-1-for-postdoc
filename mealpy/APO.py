import random
import time

import numpy as np


class APO:
    def __init__(self, weight, fitness_function, epochs, pop_size):

        # Artificial Protozoa Optimization

        self.weight = weight
        self.fitness_function = fitness_function
        self.epochs = epochs
        self.pop_size = pop_size

    def solve(self):
        lb = np.array(sum((self.weight - 0.01).tolist(), []))
        ub = np.array(sum((self.weight + 0.01).tolist(), []))
        # Low = np.min(lb)
        dim = len(lb)

        random.seed(sum([int(x) for x in time.localtime()[:6]]))
        ps = self.pop_size  # Protozoa size
        np_pairs = 1  # Neighbor pairs
        pf_max = 0.1  # Maximum proportion fraction
        protozoa = np.zeros((ps, dim))  # Protozoa
        newprotozoa = np.zeros((ps, dim))  # New protozoa
        epn = np.zeros((np_pairs, dim))  # Effect of paired neighbors

        # Initialization
        for i in range(ps):
            protozoa[i, :] = lb + np.random.rand(dim) * (ub - lb)

        # Evaluate fitness value
        protozoa_Fit = np.array([self.fitness_function(protozoa[i]) for i in range(ps)])

        # Find the best protozoa and best fit
        bestval = np.min(protozoa_Fit)
        bestid = np.argmin(protozoa_Fit)
        bestProtozoa = protozoa[bestid, :]  # Best protozoa
        bestFit = bestval  # Best fit

        # Main loop
        for iter in range(2, self.epochs + 1):
            sorted_indices = np.argsort(protozoa_Fit)
            protozoa = protozoa[sorted_indices, :]
            protozoa_Fit = protozoa_Fit[sorted_indices]

            pf = pf_max * np.random.rand()  # Proportion fraction
            ri = np.random.choice(ps, size=int(np.ceil(ps * pf)), replace=False)  # Rank index of protozoa

            for i in range(ps):
                if i in ri:  # Protozoa in dormancy or reproduction form
                    pdr = 0.5 * (1 + np.cos((1 - i / ps) * np.pi))  # Probability of dormancy and reproduction
                    if np.random.rand() < pdr:  # Dormancy form
                        newprotozoa[i, :] = lb + np.random.rand(dim) * (ub - lb)
                    else:  # Reproduction form
                        flag = random.choice([-1, 1])  # Plus or minus
                        Mr = np.zeros(dim)  # Mapping vector in reproduction
                        Mr[np.random.permutation(dim)[:int(np.ceil(np.random.rand() * dim))]] = 1
                        newprotozoa[i, :] = protozoa[i, :] + flag * np.random.rand() * (
                                lb + np.random.rand(dim) * (ub - lb)) * Mr
                else:  # Protozoa in foraging form
                    f = np.random.rand() * (1 + np.cos(iter / self.epochs * np.pi))  # Foraging factor
                    Mf = np.zeros(dim)  # Mapping vector in foraging
                    Mf[np.random.permutation(dim)[:int(np.ceil(dim * i / ps))]] = 1
                    pah = 0.5 * (1 + np.cos(iter / self.epochs * np.pi))  # Probability of autotroph and heterotroph

                    if np.random.rand() < pah:  # Autotroph form
                        j = np.random.randint(ps)  # Randomly selected protozoa
                        for k in range(np_pairs):  # Neighbor pairs
                            if i == 0:
                                km = i
                                kp = i + np.random.randint(1, ps - i)
                            elif i == ps - 1:
                                km = np.random.randint(ps - 1)
                                kp = i
                            else:
                                km = np.random.randint(0, i)
                                kp = i + np.random.randint(1, ps - i)

                            # Weight factor in the autotroph forms
                            wa = np.exp(-abs(protozoa_Fit[km] / (protozoa_Fit[kp] + np.finfo(float).eps)))
                            epn[k, :] = wa * (protozoa[km, :] - protozoa[kp, :])

                        newprotozoa[i, :] = protozoa[i, :] + f * (
                                protozoa[j, :] - protozoa[i, :] + (1 / np_pairs) * np.sum(epn, axis=0)) * Mf

                    else:  # Heterotroph form
                        for k in range(np_pairs):  # Neighbor pairs
                            if i == 0:
                                imk = i
                                ipk = i + k
                            elif i == ps - 1:
                                imk = ps - 1 - k
                                ipk = i
                            else:
                                imk = i - k
                                ipk = i + k

                            # Neighbor limit range in [0, ps-1]
                            if imk < 0:
                                imk = 0
                            elif ipk >= ps:
                                ipk = ps - 1

                            # Weight factor in the heterotroph form
                            wh = np.exp(-abs(protozoa_Fit[imk] / (protozoa_Fit[ipk] + np.finfo(float).eps)))
                            epn[k, :] = wh * (protozoa[imk, :] - protozoa[ipk, :])

                        flag = random.choice([-1, 1])  # Plus or minus
                        Xnear = (1 + flag * np.random.rand(dim) * (1 - iter / self.epochs)) * protozoa[i, :]
                        newprotozoa[i, :] = protozoa[i, :] + f * (
                                Xnear - protozoa[i, :] + (1 / np_pairs) * np.sum(epn, axis=0)) * Mf

            # Boundary control
            newprotozoa = np.clip(newprotozoa, lb, ub)

            newprotozoa_Fit = np.array([self.fitness_function(newprotozoa[i]) for i in range(ps)])

            # Update protozoa
            bin_mask = protozoa_Fit > newprotozoa_Fit
            idx = np.where(bin_mask == 1)[0]
            protozoa[idx] = newprotozoa[idx]
            protozoa_Fit[idx] = newprotozoa_Fit[idx]

            # Update best protozoa and fit
            bestFit = np.min(protozoa_Fit)
            bestid = np.argmin(protozoa_Fit)
            bestProtozoa = protozoa[bestid, :]

        return bestProtozoa, bestFit

