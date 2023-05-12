import copy
import os
import sys
from typing import Callable

import numpy as np
from random import seed
from time import perf_counter
from types import FunctionType
import pathos.multiprocessing as mp
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from base_gp_algorithm.fitness_functions import sigmoid, sse
from base_gp_algorithm.parents_selection import tournament
from base_gp_algorithm.survivors_selection import survivors_generational
from tree_based_gp.ephemeral_constants import uniform_ephemeral


class BaseGPAlgorithm:
    DEFAULT_NUM_RUNS = 5
    DEFAULT_NUM_GENERATIONS = 100
    DEFAULT_POPULATION_SIZE = 50
    DEFAULT_PROB_MUTATION_NODE = 0.1
    DEFAULT_PROB_CROSSOVER = 0.7
    DEFAULT_TOURNAMENT_SIZE = 3
    DEFAULT_ELITE_SIZE = 0.1
    DEFAULT_RANDOM_FOREIGNERS_INJECTED_SIZE = 0.2
    DEFAULT_RANDOM_FOREIGNERS_INJECTION_PERIOD = 50
    DEFAULT_FITNESS_FUNCTION = sse
    DEFAULT_FUNC_SELECTION_SURVIVORS = survivors_generational
    DEFAULT_FUNC_SELECTION_PARENTS = tournament
    DEFAULT_TARGET_FITNESS = 1.0
    DEFAULT_INVALID_FITNESS = sys.maxsize
    DEFAULT_LOG_FILE_PATH = 'output.log'

    def __init__(self,
                 problem_file_path: str,
                 num_runs: int = DEFAULT_NUM_RUNS,
                 num_generations: int = DEFAULT_NUM_GENERATIONS,
                 population_size: int = DEFAULT_POPULATION_SIZE,
                 prob_mutation: float = DEFAULT_PROB_MUTATION_NODE,
                 prob_crossover: float = DEFAULT_PROB_CROSSOVER,
                 tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
                 elite_size: float = DEFAULT_ELITE_SIZE,
                 inject_random_foreigners: bool = True,
                 random_foreigners_injected_size: float = DEFAULT_RANDOM_FOREIGNERS_INJECTED_SIZE,
                 random_foreigners_injection_period: int = DEFAULT_RANDOM_FOREIGNERS_INJECTION_PERIOD,
                 fitness_function: Callable = DEFAULT_FITNESS_FUNCTION,
                 target_fitness: float = DEFAULT_TARGET_FITNESS,
                 invalid_fitness: float = DEFAULT_INVALID_FITNESS,
                 func_selection_survivors: Callable = DEFAULT_FUNC_SELECTION_SURVIVORS,
                 func_selection_parents: Callable = DEFAULT_FUNC_SELECTION_PARENTS,
                 normalize: bool = True,
                 seed_rng: int = None,
                 use_multiprocessing: bool = False,
                 log_file_path: str = DEFAULT_LOG_FILE_PATH,
                 verbose: bool = True):

        self.initial_time = perf_counter()
        self.problem_file_path = problem_file_path
        self.num_runs = num_runs
        if not isinstance(num_generations, int) or num_generations < 1:
            self.num_generations = sys.maxsize # keep going until reaching the target fitness
        else:
            self.num_generations = num_generations
        self.population_size = population_size
        self.prob_mutation = prob_mutation
        self.prob_crossover = prob_crossover
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.inject_random_foreigners = inject_random_foreigners
        self.random_foreigners_injected_size = random_foreigners_injected_size
        self.random_foreigners_injection_period = random_foreigners_injection_period
        self.func_fitness = fitness_function
        self.func_selection_survivors = func_selection_survivors
        self.func_selection_parents = func_selection_parents
        self.target_fitness = target_fitness
        self.invalid_fitness = invalid_fitness
        self.seed_rng = seed_rng
        self.use_multiprocessing = use_multiprocessing
        self.log_file_path = log_file_path
        self.verbose = verbose

        self._reset()
        self.problem_name = os.path.splitext(os.path.basename(self.problem_file_path))[0]
        self.log_file = open(log_file_path, 'w')
        self.header, self.fit_cases = self._get_gp_problem_data()
        self.normalize = normalize

        self.denorm_fit_cases = copy.deepcopy(self.fit_cases)
        if normalize:
            self.scaler = MinMaxScaler()
            self.fit_cases = self.scaler.fit_transform(self.fit_cases).tolist()

        if seed_rng is not None:
            seed(seed_rng)

    def _log(self, msg: str, args: tuple = (), run_id: int = None):
        if self.verbose:
            current_time = perf_counter() - self.initial_time

            if run_id is None:
                current_time_str = ('[GP@%.6fs] ' % current_time)
            else:
                current_time_str = ('[GP/%d@%.6fs] ' % (run_id, current_time))

            msg_formatted = current_time_str + (msg % args)
            print(msg_formatted)
            self.log_file.write(msg_formatted + '\n')

        plt.show()

    def _reset(self):
        self.chromosomes = []
        self.population = []
        self.best_fitness = []
        self.statistics = []
        self.best_individual = []
        self.count = []

        for i in range(self.num_runs):
            self.chromosomes.append([])
            self.population.append([])
            self.best_fitness.append(0)
            self.statistics.append({})
            self.best_individual.append([])
            self.count.append(0)

    def execute(self):
        self._log('Starting genetic programming algorithm...')
        results = []
        if self.use_multiprocessing:
            num_processes = self.num_runs

            self._log('Starting %d workers for %d runs...', (num_processes, self.num_runs))

            with mp.Pool(processes=self.num_runs) as pool:
                results = pool.map(self._gp, [i for i in range(self.num_runs)])

        else:
            results = [
                self._gp(i) for i in range(self.num_runs)
            ]
        return results

    def end(self):
        self.log_file.close()

    def _get_gp_problem_data(self):
        with open(self.problem_file_path, 'r') as f_in:
            lines = f_in.readlines()
            header_line = lines[0][:-1]  # retrieve header
            header = [int(header_line.split()[0]), []]

            for i in range(1, len(header_line.split()), 2):
                header[1].append([header_line.split()[i], int(header_line.split()[i + 1])])

            data = lines[1:]  # ignore header
            fit_cases = [[float(elem) for elem in case[:-1].split()] for case in data]

        return header, fit_cases

    def plot_data(self):
        data = self.denorm_fit_cases
        x = list(map(lambda pair: pair[0], data))
        y = list(map(lambda pair: pair[1], data))
        plt.xlabel('Input variable')
        plt.ylabel('Output variable')
        plt.title(self.problem_name)
        plt.scatter(x, y)
        plt.show()

        if self.normalize:
            data = self.fit_cases
            x = list(map(lambda pair: pair[0], data))
            y = list(map(lambda pair: pair[1], data))
            plt.xlabel('Input variable')
            plt.ylabel('Output variable')
            plt.title(self.problem_name + ' | Normalized')
            plt.scatter(x, y)
            plt.show()

    def _gp(self, run_id: int):
        pass
