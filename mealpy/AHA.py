import random

import numpy as np



class AHA:

    # Artificial Hummingbird Algorithm(AHA)

    def __init__(self, weight, fitness_function, epochs, pop_size):
        self.weight = weight
        self.fitness_function = fitness_function
        self.epochs = epochs
        self.pop_size = pop_size

    @staticmethod
    def space_bound(X, Up, Low):
        Dim = len(X)
        S = (X > Up) | (X < Low)  # Use bitwise OR for logical operations
        random_values = np.random.rand(Dim) * (Up - Low) + Low
        X = np.where(S, random_values, X)  # Use np.where for conditional assignment
        return X

    def solve(self):
        lb = np.array(sum((self.weight - 0.01).tolist(), []))
        ub = np.array(sum((self.weight + 0.01).tolist(), []))

        dim = len(lb)

        PopPos = np.random.rand(self.pop_size, dim) * (ub - lb) + lb
        PopFit = np.array([self.fitness_function(PopPos[i, :]) for i in range(self.pop_size)])

        BestF = np.inf
        BestX = np.array([])

        for i in range(self.pop_size):
            if PopFit[i] <= BestF:
                BestF = PopFit[i]
                BestX = PopPos[i, :]

        HisBestFit = np.zeros(self.epochs)
        VisitTable = np.zeros((self.pop_size, self.pop_size))
        np.fill_diagonal(VisitTable, np.nan)

        for It in range(self.epochs):
            DirectVector = np.zeros((self.pop_size, dim))

            for i in range(self.pop_size):
                r = np.random.rand()
                if r < 1 / 3:
                    RandDim = np.random.permutation(dim)
                    RandNum = np.random.randint(1, dim - 1) + 1
                    DirectVector[i, RandDim[:RandNum]] = 1
                elif r > 2 / 3:
                    DirectVector[i, :] = 1
                else:
                    RandNum = np.random.randint(dim)
                    DirectVector[i, RandNum] = 1

                if np.random.rand() < 0.5:
                    MaxUnvisitedTime = np.nanmax(VisitTable[i, :])
                    TargetFoodIndex = np.argmax(VisitTable[i, :] == MaxUnvisitedTime)
                    MUT_Index = np.where(VisitTable[i, :] == MaxUnvisitedTime)[0]
                    if len(MUT_Index) > 1:
                        TargetFoodIndex = MUT_Index[np.argmin(PopFit[MUT_Index])]

                    newPopPos = PopPos[TargetFoodIndex, :] + np.random.randn(dim) * DirectVector[i, :] * (
                            PopPos[i, :] - PopPos[TargetFoodIndex, :])
                    newPopPos = self.space_bound(newPopPos, ub, lb)
                    newPopFit = self.fitness_function(newPopPos)

                    if newPopFit < PopFit[i]:
                        PopFit[i] = newPopFit
                        PopPos[i, :] = newPopPos
                        VisitTable[i, :] += 1
                        VisitTable[i, TargetFoodIndex] = 0
                        VisitTable[:, i] = np.nanmax(VisitTable, axis=1) + 1
                        VisitTable[i, i] = np.nan
                    else:
                        VisitTable[i, :] += 1
                        VisitTable[i, TargetFoodIndex] = 0
                else:
                    newPopPos = PopPos[i, :] + np.random.randn(dim) * DirectVector[i, :] * PopPos[i, :]
                    newPopPos = self.space_bound(newPopPos, ub, lb)
                    newPopFit = self.fitness_function(newPopPos)

                    if newPopFit < PopFit[i]:
                        PopFit[i] = newPopFit
                        PopPos[i, :] = newPopPos
                        VisitTable[i, :] += 1
                        VisitTable[:, i] = np.nanmax(VisitTable, axis=1) + 1
                        VisitTable[i, i] = np.nan
                    else:
                        VisitTable[i, :] += 1

            if It % (2 * self.pop_size) == 0:
                MigrationIndex = np.argmax(PopFit)
                PopPos[MigrationIndex, :] = np.random.rand(dim) * (ub - lb) + lb
                PopFit[MigrationIndex] = self.fitness_function(PopPos[MigrationIndex, :])
                VisitTable[MigrationIndex, :] += 1
                VisitTable[:, MigrationIndex] = np.nanmax(VisitTable, axis=1) + 1
                VisitTable[MigrationIndex, MigrationIndex] = np.nan

            for i in range(self.pop_size):
                if PopFit[i] < BestF:
                    BestF = PopFit[i]
                    BestX = PopPos[i, :]

            HisBestFit[It] = BestF

        return BestX, BestF, HisBestFit, VisitTable



