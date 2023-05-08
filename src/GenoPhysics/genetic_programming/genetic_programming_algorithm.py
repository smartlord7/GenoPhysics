import math
from copy import deepcopy
from operator import itemgetter
from random import seed, random, sample, choice
import sympy as sp
import numpy as np
import pathos.multiprocessing as mp
from time import perf_counter
from types import FunctionType

from matplotlib import pyplot as plt

from genetic_programming.ephemeral_constants import uniform_ephemeral
from genetic_programming.fitness_functions import sigmoid
from genetic_programming.survivors_selection import survivors_generational
from genetic_programming.parents_selection import tournament
from genetic_programming.util import is_var, generate_vars, interpreter, individual_size, \
    tree_to_inline_expression


class GeneticProgrammingAlgorithm:
    DEFAULT_NUM_RUNS = 5
    DEFAULT_NUM_GENERATIONS = 100
    DEFAULT_POPULATION_SIZE = 50
    DEFAULT_INITIAL_MAX_DEPTH = 6
    DEFAULT_PROB_MUTATION_NODE = 0.1
    DEFAULT_PROB_CROSSOVER = 0.7
    DEFAULT_TOURNAMENT_SIZE = 3
    DEFAULT_ELITE_SIZE = 0.1
    DEFAULT_RANDOM_FOREIGNERS_INJECTED_SIZE = 0.2
    DEFAULT_RANDOM_FOREIGNERS_INJECTION_PERIOD = 50
    DEFAULT_FITNESS_FUNCTION = sigmoid
    DEFAULT_FUNC_SELECTION_SURVIVORS = survivors_generational
    DEFAULT_FUNC_SELECTION_PARENTS = tournament
    DEFAULT_DIST_FUNC_EPHEMERAL = uniform_ephemeral()
    DEFAULT_TARGET_FITNESS = 1.0
    DEFAULT_LOG_FILE_PATH = 'output.log'

    MODULE_REGISTER_FUNCTION_SET = 'function_wrappers'

    def __init__(self, problem_file_path: str,
                 num_runs: int = DEFAULT_NUM_RUNS,
                 num_generations: int = DEFAULT_NUM_GENERATIONS,
                 population_size: int = DEFAULT_POPULATION_SIZE,
                 initial_max_depth: int = DEFAULT_INITIAL_MAX_DEPTH,
                 prob_mutation_node: float = DEFAULT_PROB_MUTATION_NODE,
                 prob_crossover: float = DEFAULT_PROB_CROSSOVER,
                 tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
                 elite_size: float = DEFAULT_ELITE_SIZE,
                 inject_random_foreigners: bool = True,
                 random_foreigners_injected_size: float = DEFAULT_RANDOM_FOREIGNERS_INJECTED_SIZE,
                 random_foreigners_injection_period: int = DEFAULT_RANDOM_FOREIGNERS_INJECTION_PERIOD,
                 fitness_function: FunctionType = DEFAULT_FITNESS_FUNCTION,
                 func_selection_survivors: FunctionType = DEFAULT_FUNC_SELECTION_SURVIVORS,
                 func_selection_parents: FunctionType = DEFAULT_FUNC_SELECTION_PARENTS,
                 dist_func_ephemeral: FunctionType = DEFAULT_DIST_FUNC_EPHEMERAL,
                 const_set: list = (),
                 target_fitness: float = DEFAULT_TARGET_FITNESS,
                 seed_rng: int = None,
                 use_multiprocessing: bool = False,
                 log_file_path: str = DEFAULT_LOG_FILE_PATH,
                 verbose: bool = True):

        """
        Initializes the GeneticProgrammingAlgorithm class.

        :param problem_file_path: The path to the file containing the problem data.
        :param num_generations: The number of generations to run the algorithm for.
        :param population_size: The size of the population.
        :param initial_max_depth: The initial maximum depth of the tree individuals.
        :param prob_mutation_node: The probability of mutating a node in an individual.
        :param prob_crossover: The probability of performing crossover between two individuals.
        :param tournament_size: The size of the tournament selection.
        :param seed_rng: The seed for the random number generator.
        """
        self.problem_file_path = problem_file_path
        self.num_runs = num_runs
        self.num_generations = num_generations
        self.population_size = population_size
        self.initial_max_depth = initial_max_depth
        self.prob_mutation_node = prob_mutation_node
        self.prob_crossover = prob_crossover
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.inject_random_foreigners = inject_random_foreigners
        self.random_foreigners_injected_size = random_foreigners_injected_size
        self.random_foreigners_injection_period = random_foreigners_injection_period
        self.fitness_function = fitness_function
        self.func_selection_survivors = func_selection_survivors
        self.func_selection_parents = func_selection_parents
        self.dist_func_ephemeral = dist_func_ephemeral
        self.target_fitness = target_fitness
        self.seed_rng = seed_rng
        self.use_multiprocessing = use_multiprocessing

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
            self.statistics.append([])
            self.best_individual.append([])
            self.count.append(0)

        if seed_rng is not None:
            seed(seed_rng)

        self.header, self.fit_cases = self._get_gp_problem_data()
        self.num_vars, self.function_set = self.header
        self.vars_set = generate_vars(self.num_vars)
        self.const_set = list(const_set)
        self.const_set.append(self.dist_func_ephemeral)
        self.terminal_set = self.vars_set + self.const_set
        self.log_file_path = log_file_path
        self.verbose = verbose
        self.log_file = open(log_file_path, 'w')
        self.initial_time = perf_counter()

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

    def plot_data(self):
        data = self.fit_cases
        x = list(map(lambda pair: pair[0], data))
        y = list(map(lambda pair: pair[1], data))
        plt.plot(x, y)
        plt.show()

    def start(self):
        self._log('Starting genetic programming algorithm...')
        if self.use_multiprocessing:
            num_processes = self.num_runs

            self._log('Starting %d workers for %d runs...', (num_processes, self.num_runs))

            with mp.Pool(processes=self.num_runs) as pool:
                results = pool.map(self._gp, [i for i in range(self.num_runs)])

            fig, ax = plt.subplots()
            for i in range(self.num_runs):
                ax.plot(results[i]['bests'], 'r-o', label='Best')
                mn = np.mean(np.asarray(results[i]['all']), axis=1)
                ax.plot(mn, 'g-s', label='Mean')
                # create a boxplot
                # bp = ax.boxplot(results[i]['all'], widths=0.5)
                ax.set_xlabel('Generation')
                ax.set_ylabel('Fitness ([0...1])')
                plt.show()
        else:
            results = [
                self._gp(i) for i in range(self.num_runs)]

    def end(self):
        self.log_file.close()

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

    def _gp(self, run_id: int):
        self._log('Starting run no %d...', (run_id,), run_id)
        # Reset algorithm variables, important when > 1 run
        self._reset()

        # Define initial population
        self._log('Initializing population with ramped-half-and-half...', (), run_id)
        self.chromosomes[run_id] = self._ramped_half_and_half(self.chromosomes[run_id], self.population_size)

        # Evaluate population
        self._log('Gen 0 - Evaluating initial population...', (), run_id)
        self.population[run_id] = [[chromosome, self._evaluate(chromosome)] for chromosome in self.chromosomes[run_id]]
        self.best_individual[run_id], self.best_fitness[run_id] = self._get_best_individual(run_id)
        self._log('Gen 0 - Best fitness %.8f', (self.best_fitness[run_id],), run_id)
        self.statistics[run_id]['bests'] = [self.best_fitness[run_id]]
        self.statistics[run_id]['all'] = [[individual[1] for individual in self.population[run_id]]]

        # Evolve
        for gen in range(self.num_generations):
            self._inject_random_foreigners(gen, run_id)
            # offspring after variation
            offspring = []
            for j in range(self.population_size):
                if random() < self.prob_crossover:
                    # subtree crossover
                    parent_1 = self.func_selection_parents(self.population[run_id], self.tournament_size)[0]
                    parent_2 = self.func_selection_parents(self.population[run_id], self.tournament_size)[0]
                    new_offspring = self._subtree_crossover(parent_1, parent_2, run_id)
                else:  # prob mutation = 1 - prob crossover!
                    # mutation
                    parent = self.tournament(run_id)[0]
                    new_offspring = self._point_mutation(parent)
                offspring.append(new_offspring)

            # Evaluate new population (offspring)
            offspring = [[chromosome, self._evaluate(chromosome)] for chromosome in offspring]

            # Merge parents and offspring
            self.population[run_id] = self.func_selection_survivors(self.population[run_id], offspring)

            # Statistics
            self.best_individual[run_id], self.best_fitness[run_id] = self._get_best_individual(run_id)
            self.statistics[run_id]['bests'].append(self.best_fitness[run_id])
            self.statistics[run_id]['all'].append([individual[1] for individual in self.population[run_id]])
            self._log('Gen %d - Best fitness %.8f', (gen + 1, self.best_fitness[run_id]), run_id)
            # self._log('Expression: %s', (tree_to_inline_expression(self.best_individual[run_id]), ), run_id)

        print('Final best fitness %.8f - run %d' % (self.best_fitness[run_id], run_id))
        print('Expression: %s' % tree_to_inline_expression(self.best_individual[run_id]))

        return self.statistics[run_id]

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

    def tournament(self, run_id):
        pool = sample(self.population[run_id], self.tournament_size)

        pool.sort(key=itemgetter(1), reverse=True)

        return pool[0]

    def _get_best_individual(self, run_id):
        all_fit_values = [indiv[1] for indiv in self.population[run_id]]
        max_fit = max(all_fit_values)

        index_max_fit = all_fit_values.index(max_fit)

        return self.population[run_id][index_max_fit]

    def _evaluate(self, individual):
        individual_copy = deepcopy(individual)

        predicted = []
        real = []
        for case in self.fit_cases:
            predicted.append(interpreter(individual_copy, case[:-1]))
            real.append(case[-1])

        predicted = np.asarray(predicted)
        real = np.asarray(real)

        return self.fitness_function(predicted, real)

    def _inject_random_foreigners(self, generation: int, run_id: int):
        if self.inject_random_foreigners and \
                generation != 0 and \
                (generation % self.random_foreigners_injection_period) == 0:
            size = math.ceil(self.random_foreigners_injected_size * len(self.population[run_id]))
            chromosomes = []
            rdm_foreigners_chromosomes = self._ramped_half_and_half(chromosomes, size)
            rdm_foreigners_population = [[chromosome, self._evaluate(chromosome)] for chromosome in
                                         rdm_foreigners_chromosomes]
            self.population[run_id].sort(key=itemgetter(1), reverse=False)
            rdm_foreigners_population.sort(key=itemgetter(1), reverse=True)

            self.population[run_id] = rdm_foreigners_population + self.population[run_id][size:]

    def _point_mutation(self, parent):
        parent_muted = deepcopy(parent)

        if random() < self.prob_mutation_node:
            if isinstance(parent_muted, list):
                # Function case
                symbol = parent_muted[0]
                parent_muted[0] = self._change_function(symbol)
                parent_muted[1:] = [self._point_mutation(arg) for arg in parent_muted[1:]]
            elif isinstance(parent_muted, (float, int)):
                # Constant case
                cst = self.const_set[0]

                if isinstance(cst, FunctionType):
                    return cst()
                elif isinstance(cst, (float, int)):
                    return cst
            elif is_var(parent_muted):
                # Variable case
                parent_muted = self._change_variable(parent_muted)
            else:
                raise TypeError  # should not happen

        return parent_muted

    def _change_variable(self, variable):
        if len(self.vars_set) == 1:
            return variable

        new_var = choice(self.vars_set)

        while new_var == variable:
            new_var = choice(self.vars_set)

        return new_var

    def _change_function(self, symbol):
        new_function = choice(self.function_set)

        while (new_function[0] == symbol) or (new_function[1] != self._arity(symbol)):
            new_function = choice(self.function_set)

        return new_function[0]

    def _arity(self, symbol):
        return next(func[1] for func in self.function_set if func[0] == symbol)

    def _gen_random_expression(self, method, max_depth):
        if (max_depth == 0) or (method == 'grow' and (
                random() < len(self.terminal_set) / (len(self.terminal_set) + len(self.function_set)))):

            index = choice(range(len(self.terminal_set)))

            if index == len(self.terminal_set) - 1:
                ephemeral_const = self.terminal_set[index]
                expr = ephemeral_const()
            else:
                expr = self.terminal_set[index]
        else:
            func = choice(self.function_set)
            expr = [func[0]] + [self._gen_random_expression(method, max_depth - 1) for _ in range(func[1])]

        return expr

    def _ramped_half_and_half(self, population: list, population_size: int):
        depth = list(range(3, self.initial_max_depth))

        for i in range(population_size // 2):
            population.append(self._gen_random_expression('grow', choice(depth)))

        for i in range(population_size // 2):
            population.append(self._gen_random_expression('full', choice(depth)))

        if population_size % 2 != 0:
            population.append(self._gen_random_expression('full', choice(depth)))

        return population

    def _sub_tree(self, tree, position, run_id):
        def sub_tree_(tree, position):
            if position == self.count[run_id]:
                self.count[run_id] = 0

                return tree
            else:
                self.count[run_id] += 1
                res_aux = None

                if isinstance(tree, list):
                    for i, sub in enumerate(tree[1:]):
                        res_aux = sub_tree_(sub, position)

                        if res_aux:
                            break

                    return res_aux

        return sub_tree_(tree, position)

    def _replace_sub_tree(self, tree, sub_tree_1, sub_tree_2):
        if tree == sub_tree_1:
            return sub_tree_2

        elif isinstance(tree, list):
            for i, sub in enumerate(tree[1:]):
                res = self._replace_sub_tree(sub, sub_tree_1, sub_tree_2)

                if res and (res != sub):
                    return [tree[0]] + tree[1:i + 1] + [res] + tree[i + 2:]
            return tree
        else:
            return tree

    def _subtree_crossover(self, parent1, parent2, run_id):
        size_1 = individual_size(parent1)
        size_2 = individual_size(parent2)

        cross_point_1 = choice(list(range(size_1)))
        cross_point_2 = choice(list(range(size_2)))

        # identify subtrees to exchange
        sub_tree_1 = self._sub_tree(parent1, cross_point_1, run_id)
        sub_tree_2 = self._sub_tree(parent2, cross_point_2, run_id)

        # Exchange
        new_par_1 = deepcopy(parent1)
        offspring = self._replace_sub_tree(new_par_1, sub_tree_1, sub_tree_2)

        return offspring
