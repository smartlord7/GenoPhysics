from random import seed
from time import perf_counter
from types import FunctionType

from matplotlib import pyplot as plt

from tree_based_gp.ephemeral_constants import uniform_ephemeral
from tree_based_gp.fitness_functions import sigmoid
from tree_based_gp.parents_selection import tournament
from tree_based_gp.survivors_selection import survivors_generational


class BaseGPAlgorithm:
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

    def __init__(self,
                 problem_file_path: str,
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
                 target_fitness: float = DEFAULT_TARGET_FITNESS,
                 seed_rng: int = None,
                 use_multiprocessing: bool = False,
                 log_file_path: str = DEFAULT_LOG_FILE_PATH,
                 verbose: bool = True):

        self.initial_time = perf_counter()
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
        self.target_fitness = target_fitness
        self.seed_rng = seed_rng
        self.use_multiprocessing = use_multiprocessing
        self.log_file_path = log_file_path
        self.verbose = verbose
        self.log_file = open(log_file_path, 'w')

        self._reset()

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
