import math
import numpy as np
from copy import deepcopy
from typing import Callable
from types import FunctionType
from operator import itemgetter
from matplotlib import pyplot as plt
from random import random, sample, choice
from base_gp_algorithm.base_gp_algorithm import BaseGPAlgorithm
from base_gp_algorithm.fitness_functions import sse
from tree_based_gp.ephemeral_constants import uniform_ephemeral
from tree_based_gp.util import is_var, generate_vars, interpreter, individual_size, \
    tree_to_inline_expression


class TreeBasedGPAlgorithm(BaseGPAlgorithm):
    DEFAULT_INITIAL_MAX_DEPTH = 6
    DEFAULT_DIST_FUNC_EPHEMERAL = uniform_ephemeral()
    MODULE_REGISTER_FUNCTION_SET = 'function_wrappers'
    __name__ = 'Tree-based GP'

    def __init__(self,
                 problem_file_path: str,
                 num_runs: int = BaseGPAlgorithm.DEFAULT_NUM_RUNS,
                 num_generations: int = BaseGPAlgorithm.DEFAULT_NUM_GENERATIONS,
                 population_size: int = BaseGPAlgorithm.DEFAULT_POPULATION_SIZE,
                 initial_max_depth: int = DEFAULT_INITIAL_MAX_DEPTH,
                 prob_mutation: float = BaseGPAlgorithm.DEFAULT_PROB_MUTATION_NODE,
                 prob_crossover: float = BaseGPAlgorithm.DEFAULT_PROB_CROSSOVER,
                 tournament_size: int = BaseGPAlgorithm.DEFAULT_TOURNAMENT_SIZE,
                 elite_size: float = BaseGPAlgorithm.DEFAULT_ELITE_SIZE,
                 inject_random_foreigners: bool = True,
                 random_foreigners_injected_size: float = BaseGPAlgorithm.DEFAULT_RANDOM_FOREIGNERS_INJECTED_SIZE,
                 random_foreigners_injection_period: int = BaseGPAlgorithm.DEFAULT_RANDOM_FOREIGNERS_INJECTION_PERIOD,
                 fitness_function: Callable = BaseGPAlgorithm.DEFAULT_FITNESS_FUNCTION,
                 target_fitness: float = BaseGPAlgorithm.DEFAULT_TARGET_FITNESS,
                 invalid_fitness: float = BaseGPAlgorithm.DEFAULT_INVALID_FITNESS,
                 func_selection_survivors: Callable = BaseGPAlgorithm.DEFAULT_FUNC_SELECTION_SURVIVORS,
                 func_selection_parents: Callable = BaseGPAlgorithm.DEFAULT_FUNC_SELECTION_PARENTS,
                 dist_func_ephemeral: Callable = DEFAULT_DIST_FUNC_EPHEMERAL,
                 const_set: list = (),
                 seed_rng: int = None,
                 normalize: bool = True,
                 use_multiprocessing: bool = False,
                 log_file_path: str = BaseGPAlgorithm.DEFAULT_LOG_FILE_PATH,
                 verbose: bool = True):

        """
        Initializes the GeneticProgrammingAlgorithm class.

        :param problem_file_path: The path to the file containing the problem data.
        :param num_generations: The number of generations to run the algorithm for.
        :param population_size: The size of the population.
        :param initial_max_depth: The initial maximum depth of the tree individuals.
        :param prob_mutation: The probability of mutating a node in an individual.
        :param prob_crossover: The probability of performing crossover between two individuals.
        :param tournament_size: The size of the tournament selection.
        :param seed_rng: The seed for the random number generator.
        """
        super().__init__(problem_file_path, num_runs, num_generations, population_size,
                         prob_mutation, prob_crossover, tournament_size, elite_size, inject_random_foreigners,
                         random_foreigners_injected_size, random_foreigners_injection_period, fitness_function,
                         target_fitness, invalid_fitness, func_selection_survivors, func_selection_parents, normalize, seed_rng,
                         use_multiprocessing, log_file_path, verbose)

        self.initial_max_depth = initial_max_depth
        self.dist_func_ephemeral = dist_func_ephemeral
        self.num_vars, self.function_set = self.header
        self.vars_set = generate_vars(self.num_vars)
        self.const_set = list(const_set)
        self.const_set.append(self.dist_func_ephemeral)
        self.terminal_set = self.vars_set + self.const_set
        self.min_fit = 0

    def plot_results(self, results: list) -> None:
        plt.ioff()
        fig, ax = plt.subplots()
        ax.set_xlim([0, self.num_generations + 2])
        bests = np.array([results[i]['bests'] for i in range(self.num_runs)])
        np.savetxt(fname='bests_gp.txt', X=bests, delimiter=' ', fmt='%.10f')
        mn = np.mean(bests, axis=0)
        ax.plot([i for i in range(1, self.num_generations + 2)], mn, 'g-s', label='Mean best')
        # create a boxplot
        _ = ax.boxplot(bests, widths=0.25)
        ax.set_xlabel('Generation')
        ax.set_ylabel(self.func_fitness.__name__)
        plt.title('Performance over generation | %d runs' % self.num_runs)
        plt.legend(fontsize=12)
        plt.savefig('box_plot_gp.png')
        #plt.show()

        best_of_bests = self.best_individual[np.argmin(self.best_fitness)]
        x = [ds[:-1] for ds in self.fit_cases]
        y = [ds[-1] for ds in self.fit_cases]
        best_copy = deepcopy(best_of_bests)

        predicted = []
        real = []
        for case in self.fit_cases:
            predicted.append(interpreter(best_copy, case[:-1]))
            real.append(case[-1])

        data = np.asarray([[x[i][0], y[i]] for i in range(len(x))])
        data_pred = np.asarray([[x[i][0], predicted[i]] for i in range(len(x))])

        if self.normalize:
            data = np.asarray(self.scaler.inverse_transform(data))
            data_pred = np.asarray(self.scaler.inverse_transform(data_pred))
        sse_de_norm = sse(data, data_pred)
        plt.figure()
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.plot(data[:, 0], data[:, 1], label='Real')
        plt.plot(data_pred[:, 0], data_pred[:, 1], label='Predicted')
        plt.legend(loc='best')
        plt.title('Best of bests | %s=%.8f' % (self.func_fitness.__name__, sse_de_norm))
        print(tree_to_inline_expression(best_copy))
        plt.savefig('best_of_bests_gp.png')
        #plt.show()

        return self.func_fitness(np.asarray(predicted), np.asarray(real))

    def _initialize_population(self, run_id):
        self._log('Initializing population with ramped-half-and-half...', (), run_id)
        self.chromosomes[run_id] = self._ramped_half_and_half(self.chromosomes[run_id], self.population_size)

    def _prepare_plots(self):
        plt.ion()
        fig_best = plt.figure()
        ax_best = fig_best.add_subplot(111)
        ax_best.set_title(self.__name__ + ' Training - Best Fitness')
        ax_best.set_xlabel('Generation')
        ax_best.set_ylabel(self.func_fitness.__name__)
        best_fitnesses = [-1 for gen in range(self.num_generations)]
        ax_best.set_ylim(0, 1)
        line_best, = ax_best.plot([gen for gen in range(self.num_generations)],
                                  best_fitnesses, 'r-', label='best')
        plt.legend()

        fig_avg = plt.figure()
        ax_avg = fig_avg.add_subplot(111)
        ax_avg.set_title(self.__name__ + ' Training - Average Fitness')
        ax_avg.set_xlabel('Generation')
        ax_avg.set_ylabel(self.func_fitness.__name__)
        avgs = [-1 for gen in range(self.num_generations)]
        ax_avg.set_ylim(0, 1)
        line_avg, = ax_avg.plot([gen for gen in range(self.num_generations)],
                                avgs, 'g-', label='average')
        plt.legend()

        return fig_best, ax_best, line_best, fig_avg, ax_avg, line_avg, best_fitnesses, avgs

    def _evaluate_population(self, run_id, gen):
        self._log('Gen %d - Evaluating population...', (gen,), run_id)
        self.population[run_id] = [[chromosome, self._evaluate(chromosome)] for chromosome in self.chromosomes[run_id]]
        self.best_individual[run_id], self.best_fitness[run_id] = self._get_best_individual(run_id)
        self._log('Gen %d - Best fitness %.8f', (gen, self.best_fitness[run_id],), run_id)
        self.statistics[run_id]['bests'] = [self.best_fitness[run_id]]
        self.statistics[run_id]['all'] = [[individual[1] for individual in self.population[run_id]]]

    def _evolve_population(self, run_id, gen, fig_best, ax_best, line_best, fig_avg, ax_avg, line_avg, best_fitnesses,
                           avgs):
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

        if self.best_fitness[run_id] >= self.target_fitness:
            self._log('Gen %d Target fitness %.10f reached. Terminating...', (gen, self.best_fitness[run_id],), run_id)
            self.end()

        fitnesses = list(map(lambda x: x[1] if x[1] != math.inf else 0, self.population[run_id]))
        min_fit = min(fitnesses)
        if min_fit > self.min_fit:
            self.min_fit = min_fit
        max_avg = max(avgs)
        ax_best.set_ylim(0, self.min_fit + self.min_fit / 5)
        ax_avg.set_ylim(0, max_avg + max_avg / 3)
        avgs[gen] = np.mean(fitnesses)
        best_fitnesses[gen] = self.best_fitness[run_id]
        line_best.set_ydata(best_fitnesses)
        line_avg.set_ydata(avgs)
        fig_best.canvas.draw()
        fig_best.canvas.flush_events()
        fig_avg.canvas.draw()
        fig_avg.canvas.flush_events()

        self.statistics[run_id]['bests'].append(self.best_fitness[run_id])
        self.statistics[run_id]['all'].append([individual[1] for individual in self.population[run_id]])
        self._log('Gen %d - Best fitness %.8f', (gen + 1, self.best_fitness[run_id]), run_id)
        non_simplified_expr, simplified_expr = tree_to_inline_expression(self.best_individual[run_id])
        self._log('Expression: %s', (simplified_expr,), run_id)

    def _gp(self, run_id: int):
        self._log('Starting run no %d...', (run_id,), run_id)
        # Reset algorithm variables, important when > 1 run
        self._reset()

        # Define initial population
        self._initialize_population(run_id)

        # Prepare real time plots
        fig_best, ax_best, line_best, fig_avg, ax_avg, line_avg, best_fitnesses, avgs = self._prepare_plots()

        # Evaluate population
        self._evaluate_population(run_id, 0)

        # Evolve
        for gen in range(self.num_generations):
            self._evolve_population(run_id, gen, fig_best, ax_best, line_best, fig_avg, ax_avg, line_avg,
                                    best_fitnesses, avgs)
        fig_best.savefig('best_gp_%d.png' % run_id)
        fig_avg.savefig('avg_gp_%d.png' % run_id)
        plt.close('all')

        return self.statistics[run_id]

    def tournament(self, run_id):
        pool = sample(self.population[run_id], self.tournament_size)

        pool.sort(key=itemgetter(1), reverse=False)

        return pool[0]

    def _get_best_individual(self, run_id):
        all_fit_values = [indiv[1] for indiv in self.population[run_id]]
        min_fit = min(all_fit_values)

        index_min_fit = all_fit_values.index(min_fit)

        return self.population[run_id][index_min_fit]

    def _evaluate(self, individual):
        individual_copy = deepcopy(individual)

        predicted = []
        real = []
        for case in self.fit_cases:
            predicted.append(interpreter(individual_copy, case[:-1]))
            real.append(case[-1])

        predicted = np.asarray(predicted)
        real = np.asarray(real)

        return self.func_fitness(predicted, real)

    def _inject_random_foreigners(self, generation: int, run_id: int):
        if self.inject_random_foreigners and \
                generation != 0 and \
                (generation % self.random_foreigners_injection_period) == 0:
            size = math.ceil(self.random_foreigners_injected_size * len(self.population[run_id]))
            self._log('Injected %d random foreigners', (size, ), run_id)
            chromosomes = []
            rdm_foreigners_chromosomes = self._ramped_half_and_half(chromosomes, size)
            rdm_foreigners_population = [[chromosome, self._evaluate(chromosome)] for chromosome in
                                         rdm_foreigners_chromosomes]
            self.population[run_id].sort(key=itemgetter(1), reverse=True)
            rdm_foreigners_population.sort(key=itemgetter(1), reverse=True)

            self.population[run_id] = rdm_foreigners_population + self.population[run_id][size:]

    def _point_mutation(self, parent):
        parent_muted = deepcopy(parent)

        if random() < self.prob_mutation:
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
