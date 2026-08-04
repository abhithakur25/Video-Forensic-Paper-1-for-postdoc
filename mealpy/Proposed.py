
import numpy as np
from mealpy.optimizer import Optimizer


class HYBRID(Optimizer):


    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:

        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialize_variables(self):
        self.FADS = 0.2
        self.P = 0.5

    def evolve(self, epoch):

        CF = (1 - epoch/self.epoch)**(2 * epoch/self.epoch)
        RL = self.get_levy_flight_step(beta=1.5, multiplier=0.05, size=(self.pop_size, self.problem.n_dims), case=-1)
        RB = self.generator.standard_normal((self.pop_size, self.problem.n_dims))
        per1 = self.generator.permutation(self.pop_size)
        per2 = self.generator.permutation(self.pop_size)
        pop_new = []
        for idx in range(0, self.pop_size):
            R = self.generator.random(self.problem.n_dims)
            if epoch < self.epoch / 3:
                step_size = RB[idx] * (self.g_best.solution - RB[idx] * self.pop[idx].solution)
                MarinePredators = self.pop[idx].solution + self.P * R * step_size   # EQN (1)
                LB, UB = self.problem.lb / epoch, self.problem.ub / epoch
                Walrus = self.pop[idx].solution + LB + (UB - self.generator.random() * LB)   # EQN (2)

                pos_new = 0.5 * (MarinePredators + Walrus)   # EQN (3)(HYBRID)



            elif self.epoch / 3 < epoch < 2*self.epoch / 3:
                if idx > self.pop_size / 2:
                    step_size = RB[idx] * (RB[idx] * self.g_best.solution - self.pop[idx].solution)
                    MarinePredators = self.g_best.solution + self.P * CF * step_size   # EQN (4)
                    LB, UB = self.problem.lb / epoch, self.problem.ub / epoch
                    Walrus = self.pop[idx].solution + LB + (UB - self.generator.random() * LB)  # EQN (5)
                    pos_new = 0.5 * (MarinePredators + Walrus)  # EQN (6)(HYBRID)

                else:
                    step_size = RL[idx] * (self.g_best.solution - RL[idx] * self.pop[idx].solution)
                    MarinePredators = self.pop[idx].solution + self.P * R * step_size   # EQN (7)
                    LB, UB = self.problem.lb / epoch, self.problem.ub / epoch
                    Walrus = self.pop[idx].solution + LB + (UB - self.generator.random() * LB)  # EQN (8)
                    pos_new = 0.5 * (MarinePredators + Walrus)  # EQN (9)(HYBRID)

            else:       # Phase 3 (Eq. 15)
                step_size = RL[idx] * (RL[idx] * self.g_best.solution - self.pop[idx].solution)
                MarinePredators = self.g_best.solution + self.P * CF * step_size   # EQN (10)
                LB, UB = self.problem.lb / epoch, self.problem.ub / epoch
                Walrus = self.pop[idx].solution + LB + (UB - self.generator.random() * LB)  # EQN (11)
                pos_new = 0.5 * (MarinePredators + Walrus)  # EQN (12)(HYBRID)

            pos_new = self.correct_solution(pos_new)
            if self.generator.random() < self.FADS:
                u = np.where(self.generator.random(self.problem.n_dims) < self.FADS, 1, 0)
                pos_new = pos_new + CF * (self.problem.lb + self.generator.random(self.problem.n_dims) * (self.problem.ub - self.problem.lb)) * u
            else:
                r = self.generator.random()
                step_size = (self.FADS * (1 - r) + r) * (self.pop[per1[idx]].solution - self.pop[per2[idx]].solution)
                pos_new = pos_new + step_size
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
