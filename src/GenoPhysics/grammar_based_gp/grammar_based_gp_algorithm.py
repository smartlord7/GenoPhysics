import copy
from typing import Callable
import numpy as np
from random import random, randint
from matplotlib import pyplot as plt
from base_gp_algorithm import function_wrappers
from base_gp_algorithm.base_gp_algorithm import BaseGPAlgorithm
from grammar_based_gp.crossover_operators import one_point_crossover


class GrammarBasedGPAlgorithm(BaseGPAlgorithm):
    DEFAULT_GENOTYPE_SIZE = 256
    DEFAULT_GRAMMAR_AXIOM = 'start'
    DEFAULT_GRAMMAR_WRAPPER = 2
    DEFAULT_FUNC_CROSSOVER = one_point_crossover

    def __init__(self,
                 problem_file_path: str,
                 grammar: dict,
                 num_runs: int = BaseGPAlgorithm.DEFAULT_NUM_RUNS,
                 num_generations: int = BaseGPAlgorithm.DEFAULT_NUM_GENERATIONS,
                 population_size: int = BaseGPAlgorithm.DEFAULT_POPULATION_SIZE,
                 prob_mutation: float = BaseGPAlgorithm.DEFAULT_PROB_MUTATION_NODE,
                 genotype_size: int = DEFAULT_GENOTYPE_SIZE,
                 grammar_axiom: str = DEFAULT_GRAMMAR_AXIOM,
                 grammar_wrapper: int = DEFAULT_GRAMMAR_WRAPPER,
                 prob_crossover: float = BaseGPAlgorithm.DEFAULT_PROB_CROSSOVER,
                 func_crossover: Callable = DEFAULT_FUNC_CROSSOVER,
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
                 seed_rng: int = None,
                 use_multiprocessing: bool = False,
                 log_file_path: str = BaseGPAlgorithm.DEFAULT_LOG_FILE_PATH,
                 verbose: bool = True):

        super().__init__(problem_file_path, num_runs, num_generations, population_size,
                         prob_mutation, prob_crossover, tournament_size, elite_size, inject_random_foreigners,
                         random_foreigners_injected_size, random_foreigners_injection_period, fitness_function,
                         target_fitness, invalid_fitness, func_selection_survivors, func_selection_parents, seed_rng,
                         use_multiprocessing, log_file_path, verbose)

        self.func_crossover = func_crossover
        self.grammar = grammar
        self.genotype_size = genotype_size
        self.grammar_axiom = grammar_axiom
        self.grammar_wrapper = grammar_wrapper

    def plot_results(self, individual):
        x = [ds[:-1] for ds in self.fit_cases]
        y = [ds[-1] for ds in self.fit_cases]

        y_pred = []

        for ds in self.fit_cases:
            ind_cp = copy.deepcopy(individual)
            ind = ind_cp[0][0]

            for i in range(len(ds) - 1):
                ind = ind.replace('x[' + str(i) + ']', str(ds[i]))
            y_pred.append(eval(ind))

        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.plot(x, y, label='Y')
        plt.plot(x, y_pred, label='Y_pred')
        plt.legend(loc='best')
        plt.show()

    def _gp(self, run_id: int):
        self._log('Starting run no %d...', (run_id,), run_id)
        self._reset()

        self.bests = []
        self._log('Initializing population...', (), run_id)
        self.population[run_id] = self.generate_initial_population()
        elite_size_ = int(self.population_size * self.elite_size)

        for gen in range(self.num_generations):
            self.population[run_id] = [self._evaluate(individual) for individual in self.population[run_id]]
            self.population[run_id].sort(key=lambda x: x[1])
            best = (self._mapping(self.population[run_id][0][0]), self.population[run_id][0][1])
            self._log('Gen %d - Best fitness %.10f', (gen, best[1],), run_id)

            if best[1] <= self.target_fitness:
                self._log('Gen %d Target fitness %.10f reached. Terminating...', (gen, best[1],), run_id)
                self.end()

            self.bests.append(best)
            new_population = self.population[run_id][:elite_size_]

            for i in range(elite_size_, self.population_size):
                if random() < self.prob_crossover:
                    p1 = self.func_selection_parents(self.population[run_id], self.tournament_size)
                    p2 = self.func_selection_parents(self.population[run_id], self.tournament_size)
                    ni = self.func_crossover(p1, p2)
                else:
                    # tournament
                    ni = self.func_selection_parents(self.population[run_id], self.tournament_size)
                ni = self.mutate(ni)
                new_population.append(ni)

            self.population[run_id] = new_population

        return best

    def generate_initial_population(self):
        return [self.generate_random_individual() for _ in range(self.population_size)]

    def generate_random_individual(self):
        genotype = [randint(0, 256) for _ in range(self.genotype_size)]

        return [genotype, None]

    def mutate(self, parent):
        parent = copy.deepcopy(parent)
        parent[1] = None

        for i in range(len(parent[0])):
            if random() < self.prob_mutation:
                parent[0][i] = randint(0, 256)

        return parent

    def _mapping(self, genotype):
        """
        @used_gene: pointer for the integer to be used to decide the production rule
        @production_options: list of production alternatives
        @symbols_to_expand: frontier of the derivation tree. treated as a stack for a depth-first expansion
        @output: word generated = individual
        @current_production: index of the chosen production
        """
        wraps = 0
        used_gene = 0
        output = []
        production_options = []
        symbols_to_expand = [self.grammar_axiom]

        while (wraps < self.grammar_wrapper) and (len(symbols_to_expand) > 0):
            # test: end of genotype but still symbols to expand ==> wrap
            if used_gene % len(genotype) == 0 and used_gene > 0 and len(production_options) > 1:
                wraps += 1
                used_gene = 0

            current_symbol = symbols_to_expand.pop(0)

            if current_symbol in self.grammar.keys():  # Non terminal?
                production_options = self.grammar[current_symbol]
                current_production = genotype[used_gene] % len(production_options)
                # current_production = genotype[used_gene % len(genotype)] % len(production_options)
                # consume gene only if needed

                if len(production_options) > 1:
                    used_gene += 1
                symbols_to_expand = production_options[current_production] + symbols_to_expand
            else:
                # Terminal!
                output.extend([current_symbol])
        # Impossible to Expand?

        if len(symbols_to_expand) > 0:
            return None
        return "".join(output)

    def _evaluate(self, individual):
        phenotype = self._mapping(individual[0])

        if phenotype is not None:
            predicted = []

            for ds in self.fit_cases:
                ind = phenotype[:]

                for i in range(len(ds) - 1):
                    ind = ind.replace('x[' + str(i) + ']', str(ds[i]))

                predicted.append(eval(ind))

            real = np.asarray([self.fit_cases[i][-1] for i in range(len(predicted))])
            predicted = np.asarray(predicted)
            individual[1] = self.func_fitness(predicted, real)
        else:
            individual[1] = self.invalid_fitness
        return individual
