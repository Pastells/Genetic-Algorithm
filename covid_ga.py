"""Genetic algorithm to obtain covid model"""
import random
import traceback
from copy import deepcopy
import numpy as np
from numpy import loadtxt
import scipy.integrate
import matplotlib.pyplot as plt

# ------------------------------------------
# Global parameters
# ------------------------------------------

# Import data to compare with
TIME_SERIES = loadtxt("data.dat")

# Individual numbers, should be even
INITIAL = 1000
NEW = 200  # Should be divisible by KEEP
KEEP = 50
ELITE = 10

N_GENERATIONS = 50
PARAM_NUMBER = 14  # parameters + IC

# Sensible allowed ranges for the parameters
RANGES = np.array(
    [
        [0.05, 0.5],  # beta
        [0, 1],  # phi
        [0, 1],  # eps_i
        [0, 1],  # eps_y
        [0.05, 2],  # sigma
        [0.05, 2],  # gamma1
        [0.05, 2],  # gamma2
        [0.005, 0.5],  # kappa
        [0.001, 1],  # prob
        [0.001, 0.5],  # alpha
        [0.001, 0.3],  # delta
        [1, 100],  # E_0
        [1, 100],  # I1_0
        [1, 100],  # A_0
    ]
)

# -----------------------------------------------------------------
# ------------- Individual Class ----------------------------------
# -----------------------------------------------------------------


class Individual:
    """Individual with variables:
    number: integer to identify the individual
    fitness: how good is the individual, ideal is 0
    params: vector with parameters

    Methods:
    solve: solve ODE and return result
    compute_fitness: compare result with data
    plot: show result and comparison with data
    """

    def __init__(self, number):
        self.number = number  # id
        self.fitness = None  # yet to be computed

        rand_vect = np.random.rand(PARAM_NUMBER)
        self.params = np.zeros(PARAM_NUMBER)
        for param in range(PARAM_NUMBER):
            self.params[param] = RANGES[param, 0] + rand_vect[param] * (
                RANGES[param, 1] - RANGES[param, 0]
            )

    def solve(self):
        """Solve ODE"""
        # Initial conditions
        P = 1000000
        S0 = P - self.params[-3:].sum()
        x0 = (S0, self.params[11], self.params[12], self.params[13], 0, 0, 0, 0, 0)
        # x0 = (1000000, 1, 1, 1, 0, 0, 0, 0, 0)

        ts = np.linspace(0, 101, num=1010)
        return scipy.integrate.odeint(model, x0, ts, args=tuple(self.params[:11]))

    def compute_fitness(self, fitness_list):
        """Max os sum of squares"""
        fitness = 0

        result = self.solve()
        for day in range(101):
            _fitness = (
                (TIME_SERIES[day, 1] - result[day * 10, 3]) ** 2  # A
                + (TIME_SERIES[day, 2] - result[day * 10, 5]) ** 2  # I2
                + (TIME_SERIES[day, 3] - result[day * 10, 6]) ** 2  # Y
                + (TIME_SERIES[day, 4] - result[day * 10, 7]) ** 2  # R
                + (TIME_SERIES[day, 5] - result[day * 10, 8]) ** 2  # D
            ) / 1e6
            fitness = max(fitness, _fitness)
        self.fitness = fitness

        if (len(fitness_list) - 1) < self.number:
            fitness_list.append([self.number, fitness])
        else:
            fitness_list[self.number] = [self.number, fitness]

    def plot(self):
        """Generate plots"""

        ts = np.linspace(0, 101, num=1010)
        result = self.solve()

        # Result
        compartments = ["E", "I1", "A", "Ad", "I2", "Y", "R", "D"]
        for i, compartment in enumerate(compartments):
            plt.plot(ts, result[:, i + 1], label=compartment)
        plt.xlabel("Time (days)")
        plt.ylabel("Individuals")
        plt.legend()
        plt.savefig("sol.pdf")
        plt.yscale("log")
        plt.savefig("sol_log.pdf")
        plt.show()

        # Comparison
        compartments = ["A", "I2", "Y", "R", "D"]
        indices = [3, 5, 6, 7, 8]

        def subplots(log=False):
            fig = plt.figure(figsize=(16, 10))
            for i, compartment in enumerate(compartments):
                fig.add_subplot(3, 2, i + 1)
                plt.plot(ts, result[:, indices[i]], label=compartment)
                plt.plot(TIME_SERIES[:, i + 1], label=f"data {compartment}")
                plt.xlabel("Time (days)")
                plt.ylabel("Individuals")
                plt.legend()
                if log is True:
                    plt.yscale("log")
                else:
                    plt.ticklabel_format(axis="both", style="sci")

        subplots()
        plt.savefig("comp.pdf")
        subplots(True)
        plt.savefig("comp_log.pdf")
        plt.show()


# -----------------------------------------------------------------
# --------------- ODE Model ---------------------------------------
# -----------------------------------------------------------------


def model(x, t, *params):
    """Model with all the diferential equations
    Arguments
        x: vector of state variables
        t: time
        *params: vector with the parameters
    """

    S, E, I1, A, Ad, I2, Y, R, D = x
    beta, phi, eps_i, eps_y, sigma, gamma1, gamma2, kappa, prob, alpha, delta = params

    # Population size P
    P = S + E + I1 + A + Ad + I2 + Y + R

    # Lambda parameter (depends on time)
    lambd = beta * (phi * I1 + A + (1 - eps_i) * (Ad + I2) + (1 - eps_y) * Y) / P

    # We create dx/dt = (dS/dt, dE/dt, ...)
    dx_dt = (
        -lambd * S,
        lambd * S - sigma * E,
        sigma * E - gamma1 * I1,
        gamma1 * (1 - prob) * I1 - (kappa + gamma2) * A,
        kappa * A - gamma2 * Ad,
        gamma1 * prob * I1 - (alpha + gamma2) * I2,
        alpha * I2 - (delta + gamma2) * Y,
        gamma2 * (A + Ad + I2 + Y),
        delta * Y,
    )
    return dx_dt


# -----------------------------------------------------------------
# --------------- Genetic Algorithm -------------------------------
# -----------------------------------------------------------------


def initialize(individuals, fitness_list):
    """Fill vector with NEW individuals"""
    for ind in range(len(individuals), INITIAL):
        individuals.append(Individual(ind))
        individuals[ind].compute_fitness(fitness_list)
    return individuals, fitness_list


# -----------------------------------------------------------------


def select(individuals, fitness_list, keep=KEEP):
    """Select KEEP fittest individuals"""

    # sort by fitness
    fitness_list.sort(key=lambda x: x[1])
    fitness_list = fitness_list[:keep]

    # keep only best individuals
    selection = [ind[0] for ind in fitness_list]
    individuals = [individuals[ind] for ind in selection]

    # relable individuals
    for ind, v in enumerate(individuals):
        fitness_list[ind][0] = ind
        v.number = ind

    print(f"    fitness {individuals[0].fitness}\n")

    return individuals, fitness_list


# -----------------------------------------------------------------


def crossover(v, w):
    """Swap between 1 and 6 parameters for individuals v and w"""
    swap = np.unique(np.random.randint(0, 10, random.randint(1, 6)))
    for param in swap:
        v.params[param] = w.params[param], w.params[param] = v.params[param]


# -----------------------------------------------------------------


def line_recombination(v, w):
    """Combine individuals v and w"""
    p = 0.25
    r = np.random.rand(2)
    alpha = -p + (1 + 2 * p) * r[0]
    beta = -p + (1 + 2 * p) * r[1]
    for param in range(PARAM_NUMBER):
        t = alpha * v.params[param] + (1 - alpha) * w.params[param]
        s = beta * w.params[param] + (1 - beta) * v.params[param]

        if (
            RANGES[param, 0] < t < RANGES[param, 1]
            and RANGES[param, 0] < s < RANGES[param, 1]
        ):
            v.params[param] = t
            w.params[param] = s


# -----------------------------------------------------------------


def intermediate_recombination(v, w):
    """Similar to line_recombination,
    but with different alpha and beta for each param"""
    p = 0.25
    for param in range(PARAM_NUMBER):
        while True:
            r = np.random.rand(2)
            alpha = -p + (1 + 2 * p) * r[0]
            beta = -p + (1 + 2 * p) * r[1]
            t = alpha * v.params[param] + (1 - alpha) * w.params[param]
            s = beta * w.params[param] + (1 - beta) * v.params[param]
            if (
                RANGES[param, 0] < t < RANGES[param, 1]
                and RANGES[param, 0] < s < RANGES[param, 1]
            ):
                break
        v.params[param] = t
        w.params[param] = s


# -----------------------------------------------------------------


def mutate(individuals, fitness_list, ind, current_gen, ratio=False):
    """Non-uniform mutation, changes between 1 and 11 parameters.
    Each individual generates (NEW/KEEP - 1) new individuals + itself"""
    b = 0.5
    if ratio is False:
        ratio = int(NEW / KEEP) - 1

    for _ in range(ratio):
        swap = np.unique(np.random.randint(0, 10, random.randint(1, 11)))
        tau = random.choice([-1, 1])
        rand = random.random()

        # number of carried mutations, if 0 the new individual gets deleted
        carried = 0
        individuals.append(deepcopy(individuals[ind]))
        for param in swap:
            temp = individuals[-1].params[param] + tau * (
                RANGES[param, 0] - RANGES[param, 1]
            ) * (1 - rand ** ((1 - current_gen / N_GENERATIONS) ** b))
            if RANGES[param, 0] < temp < RANGES[param, 1]:
                carried += 1
                individuals[-1].params[param] = temp

        if carried == 0:
            del individuals[-1]
        else:
            individuals[-1].number = len(individuals) - 1
            individuals[-1].compute_fitness(fitness_list)

    return individuals, fitness_list


# -----------------------------------------------------------------


def recombine(individuals, fitness_list):
    """Recombine individuals by pairs,
    except for ELITE, that get a copy
    Choose between line or intermediate recombination (commented)"""

    fitness_list = fitness_list[:ELITE]

    # Create copy for individuals in ELITE
    for ind in range(ELITE):
        individuals.append(deepcopy(individuals[ind]))
        individuals[-1].number = len(individuals) - 1

    # Order in which individuals get recombined together
    # If not even a single individual is left out
    order = list(range(ELITE, len(individuals) // 2 * 2))
    random.shuffle(order)
    for i in range(0, len(order), 2):
        # line_recombination(individuals[order[i]], individuals[order[i + 1]])
        intermediate_recombination(individuals[ind], individuals[ind + 1])
        individuals[order[i]].compute_fitness(fitness_list)
        individuals[order[i + 1]].compute_fitness(fitness_list)

    return individuals, fitness_list


# ------------------------------------------------
# ---------------- MAIN --------------------------
# ------------------------------------------------


def main():
    np.random.seed(0)
    random.seed(0)
    # Create individuals vector
    individuals = []
    fitness_list = []

    # Initialization and selection
    individuals, fitness_list = initialize(individuals, fitness_list)
    print(f"Generation {0} best:")
    individuals, fitness_list = select(individuals, fitness_list)

    # First "round", more exploratory
    for gen in range(1, N_GENERATIONS):
        individuals, fitness_list = recombine(individuals, fitness_list)
        for ind in range(ELITE, int(len(individuals))):
            individuals, fitness_list = mutate(individuals, fitness_list, ind, gen)
        print(f"Generation {gen} best:")
        # individuals, fitness_list = initialize(individuals, fitness_list)
        individuals, fitness_list = select(individuals, fitness_list)

    # Second "round" starting only with the ELITE individuals, and more mutations
    # The idea is to be more exploitatory
    individuals = individuals[:ELITE]
    print("Only ELITE")
    for gen in range(1, N_GENERATIONS):
        individuals, fitness_list = recombine(individuals, fitness_list)
        for ind in range(ELITE, int(len(individuals))):
            individuals, fitness_list = mutate(individuals, fitness_list, ind, gen, 7)
        print(f"Generation {N_GENERATIONS + gen} best:")
        individuals, fitness_list = select(individuals, fitness_list)

    print("Best individual:")
    print(f"    Params {individuals[0].params}")

    individuals[0].plot()
    # Save best configuration
    with open("fittest.dat", "w") as file:
        file.write("params = [\n")
        for param in range(PARAM_NUMBER):
            file.write(f"\t{individuals[0].params[param]},\n")
        file.write("]")

    # -----------------------------------------------------------------

    # Comute and plot specific individual
    # Comment above and uncomment here
    """
    individual = Individual(1)
    individual.params = np.array(
        [
            0.35770687448545474,
            0.5209888214654312,
            0.43913739182637307,
            0.07727504196907982,
            0.108958753111666,
            0.07996800343837741,
            0.06761867881672093,
            0.10933907182588708,
            0.5222269240825622,
            0.050592826333673686,
            0.0277588689648371,
            41.18828987486013,
            3.4070067713056202,
            34.91848744981746,
        ]
    )
    individual.plot()
    """


# ------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(repr(ex))
        traceback.print_exc(ex)
