import numpy as np


class COOT:
    def __init__(self, weight, fitness_function, epochs, pop_size):
        self.weight = weight
        self.fitness_function = fitness_function
        self.epochs = epochs
        self.pop_size = pop_size
        
    def evolve(self):
        lb = np.array(sum((self.weight - 0.01).tolist(), []))
        ub = np.array(sum((self.weight + 0.01).tolist(), []))
        # Low = np.min(lb)
        dim = len(lb)

        NLeader = int(np.ceil(0.1 * self.pop_size))
        Ncoot = self.pop_size - NLeader
        Convergence_curve = np.zeros(self.epochs)
        gBest = np.zeros(dim)
        gBestScore = np.inf

        # Initialize the positions of Coots
        CootPos = np.random.rand(Ncoot, dim) * (ub - lb) + lb
        CootFitness = np.zeros(Ncoot)

        # Initialize the locations of Leaders
        LeaderPos = np.random.rand(NLeader, dim) * (ub - lb) + lb
        LeaderFit = np.zeros(NLeader)

        for i in range(CootPos.shape[0]):
            CootFitness[i] = self.fitness_function(CootPos[i, :])
            if gBestScore > CootFitness[i]:
                gBestScore = CootFitness[i]
                gBest = CootPos[i, :]

        for i in range(LeaderPos.shape[0]):
            LeaderFit[i] = self.fitness_function(LeaderPos[i, :])
            if gBestScore > LeaderFit[i]:
                gBestScore = LeaderFit[i]
                gBest = LeaderPos[i, :]

        Convergence_curve[0] = gBestScore
        l = 2  # Loop counter

        while l < self.epochs + 1:
            B = 2 - l * (1 / self.epochs)
            A = 1 - l * (1 / self.epochs)

            for i in range(CootPos.shape[0]):
                if np.random.rand() < 0.5:
                    R = -1 + 2 * np.random.rand()
                    R1 = np.random.rand()
                else:
                    R = -1 + 2 * np.random.rand(dim)
                    R1 = np.random.rand(dim)

                k = 1 + (i % NLeader)
                if np.random.rand() < 0.5:
                    CootPos[i, :] = 2 * R1 * np.cos(2 * np.pi * R) * (LeaderPos[k - 1, :] - CootPos[i, :]) + LeaderPos[
                                                                                                             k - 1, :]
                    # Check boundaries
                    Tp = CootPos[i, :] > ub
                    Tm = CootPos[i, :] < lb
                    CootPos[i, :] = (CootPos[i, :] * ~(Tp + Tm)) + ub * Tp + lb * Tm
                else:
                    if np.random.rand() < 0.5 and i != 0:
                        CootPos[i, :] = (CootPos[i, :] + CootPos[i - 1, :]) / 2
                    else:
                        Q = np.random.rand(dim) * (ub - lb) + lb
                        CootPos[i, :] = CootPos[i, :] + A * R1 * (Q - CootPos[i, :])
                    Tp = CootPos[i, :] > ub
                    Tm = CootPos[i, :] < lb
                    CootPos[i, :] = (CootPos[i, :] * ~(Tp + Tm)) + ub * Tp + lb * Tm

            # Fitness of location of Coots
            for i in range(CootPos.shape[0]):
                CootFitness[i] = self.fitness_function(CootPos[i, :])
                k = 1 + (i % NLeader)
                # Update the location of coot
                if CootFitness[i] < LeaderFit[k - 1]:
                    Temp = LeaderPos[k - 1, :]
                    TemFit = LeaderFit[k - 1]
                    LeaderFit[k - 1] = CootFitness[i]
                    LeaderPos[k - 1, :] = CootPos[i, :]
                    CootFitness[i] = TemFit
                    CootPos[i, :] = Temp

            # Fitness of location of Leaders
            for i in range(LeaderPos.shape[0]):
                if np.random.rand() < 0.5:
                    R = -1 + 2 * np.random.rand()
                    R3 = np.random.rand()
                else:
                    R = -1 + 2 * np.random.rand(dim)
                    R3 = np.random.rand(dim)

                if np.random.rand() < 0.5:
                    Temp = B * R3 * np.cos(2 * np.pi * R) * (gBest - LeaderPos[i, :]) + gBest
                else:
                    Temp = B * R3 * np.cos(2 * np.pi * R) * (gBest - LeaderPos[i, :]) - gBest

                Tp = Temp > ub
                Tm = Temp < lb
                Temp = (Temp * ~(Tp + Tm)) + ub * Tp + lb * Tm
                TempFit = self.fitness_function(Temp)

                # Update the location of Leader
                if gBestScore > TempFit:
                    LeaderFit[i] = gBestScore
                    LeaderPos[i, :] = gBest
                    gBestScore = TempFit
                    gBest = Temp

            Convergence_curve[l - 1] = gBestScore
            l = l + 1

        return Convergence_curve, gBest, gBestScore