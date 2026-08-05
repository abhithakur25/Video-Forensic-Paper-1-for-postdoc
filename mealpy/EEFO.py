import numpy as np
from scipy.special import gamma


class EEFO:
    def __init__(self, weight, fitness_function, epochs, pop_size):
        self.weight = weight
        self.fitness_function = fitness_function
        self.epochs = epochs
        self.pop_size = pop_size

    @staticmethod
    def levy(d):
        b = 1.5
        s = (gamma(1 + b) * np.sin(np.pi * b / 2) / (gamma((1 + b) / 2) * b * 2 ** ((b - 1) / 2))) ** (1 / b)
        u = np.random.randn(1, d) * s
        v = np.random.randn(1, d)
        sigma = u / np.abs(v) ** (1 / b)
        return sigma

    @staticmethod
    def SpaceBound(X, Up, Low):
        Dim = len(X)
        S = (X > Up) + (X < Low)
        X = (np.random.rand(Dim) * (Up - Low) + Low) * S + X * (~S)
        return X

    def evolve(self):
        lb = np.array(sum((self.weight - 0.01).tolist(), []))
        ub = np.array(sum((self.weight + 0.01).tolist(), []))
        # Low = np.min(lb)
        Dim = len(lb)
        PopPos = np.random.rand(self.pop_size, Dim) * ub - lb + lb
        PopFit = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            PopFit[i] = self.fitness_function(PopPos[i, :])

        BestF = PopFit[0]
        Xprey = PopPos[0, :]
        for i in range(1, self.pop_size):
            if PopFit[i] <= BestF:
                BestF = PopFit[i]
                Xprey = PopPos[i, :]

        HisBestF = np.zeros(self.epochs)

        for It in range(self.epochs):
            DirectVector = np.zeros((self.pop_size, Dim))
            E0 = 4 * np.sin(1 - It / self.epochs)
            for i in range(self.pop_size):
                E = E0 * np.log(1 / np.random.rand())  # Eq.(30)

                if Dim == 1:
                    DirectVector[i, :] = 1
                else:
                    RandNum = np.ceil((self.epochs - It) / self.epochs * np.random.rand() * (Dim - 2) + 2)  # Eq.(6)
                    RandDim = np.random.permutation(Dim)
                    DirectVector[i, RandDim[:int(RandNum)]] = 1

                if E > 1:
                    K = np.setdiff1d(np.arange(self.pop_size), i)
                    j = np.random.choice(K)
                    # Eq.(7), interacting
                    if PopFit[j] < PopFit[i]:
                        if np.random.rand() > 0.5:
                            newPopPos = PopPos[j, :] + np.random.randn() * DirectVector[i, :] * (
                                    np.mean(PopPos, axis=0) - PopPos[i, :])
                        else:
                            xr = np.random.rand(Dim) * (ub - lb) + lb
                            newPopPos = PopPos[j, :] + 1 * np.random.randn() * DirectVector[i, :] * (xr - PopPos[i, :])
                    else:
                        if np.random.rand() > 0.5:
                            newPopPos = PopPos[i, :] + np.random.randn() * DirectVector[i, :] * (
                                    np.mean(PopPos, axis=0) - PopPos[j, :])
                        else:
                            xr = np.random.rand(Dim) * (ub - lb) + lb
                            newPopPos = PopPos[i, :] + np.random.randn() * DirectVector[i, :] * (xr - PopPos[j, :])
                else:
                    if np.random.rand() < 1 / 3:
                        # resting
                        Alpha = 2 * (np.exp(1) - np.exp(It / self.epochs)) * np.sin(2 * np.pi * np.random.rand())  # Eq.(15)
                        rn = np.random.randint(self.pop_size)
                        rd = np.random.randint(Dim)
                        # if not isinstance(Low, int):
                        z = (PopPos[rn, rd] - lb[rd]) / (ub[rd] - lb[rd])  # Eq.(13)
                        Z = lb + z * (ub - lb)  # Eq.(12)
                        # else:
                        #     Z = PopPos[rn, rd] * np.ones(Dim)
                        Ri = Z + Alpha * np.abs(Z - Xprey)  # Eq.(14)
                        newPopPos = Ri + np.random.randn() * (Ri - np.round(np.random.rand()) * PopPos[i, :])  # Eq.(16)
                    elif np.random.rand() > 2 / 3:
                        # migrating
                        rn = np.random.randint(self.pop_size)
                        rd = np.random.randint(Dim)
                        # if not isinstance(Low, int):
                        z = (PopPos[rn, rd] - lb[rd]) / (ub[rd] - lb[rd])
                        Z = lb + z * (ub - lb)
                        # else:
                        #     Z = PopPos[rn, rd] * np.ones(Dim)
                        Alpha = 2 * (np.exp(1) - np.exp(It / self.epochs)) * np.sin(2 * np.pi * np.random.rand())
                        Ri = Z + Alpha * np.abs(Z - Xprey)  # resting area
                        Beta = 2 * (np.exp(1) - np.exp(It / self.epochs)) * np.sin(2 * np.pi * np.random.rand())  # Eq.(21)
                        Hr = Xprey + Beta * np.abs(np.mean(PopPos, axis=0) - Xprey)  # Eq.(25) hunting area
                        L = 0.01 * np.abs(self.levy(Dim))  # Eq.(26)
                        newPopPos = -np.random.rand() * Ri + np.random.rand() * Hr - L * (Hr - PopPos[i, :])  # Eq.(24)
                    else:
                        # Hunting
                        Beta = 2 * (np.exp(1) - np.exp(It / self.epochs)) * np.sin(2 * np.pi * np.random.rand())  # Eq.(21)
                        Hprey = Xprey + Beta * np.abs(np.mean(PopPos, axis=0) - Xprey)  # Eq.(20) hunting area
                        r4 = np.random.rand()
                        Eta = np.exp(r4 * (1 - It) / self.epochs) * (np.cos(2 * np.pi * r4))  # Eq.(23)
                        newPopPos = Hprey + Eta * (

                                Hprey - np.round(np.random.rand()) * PopPos[i, :])  # Eq.(22) hunting

                newPopPos = self.SpaceBound(newPopPos, ub, lb)
                newPopFit = self.fitness_function(newPopPos)
                if newPopFit < PopFit[i]:
                    PopFit[i] = newPopFit
                    PopPos[i, :] = newPopPos
                    if PopFit[i] <= BestF:
                        BestF = PopFit[i]
                        Xprey = PopPos[i, :]

            HisBestF[It] = BestF

        return Xprey, BestF, HisBestF
