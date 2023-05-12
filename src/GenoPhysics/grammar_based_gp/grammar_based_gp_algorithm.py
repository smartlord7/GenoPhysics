"""
------------GenoPhysics: Kepler's Third Law of Planetary Motion------------
 University of Coimbra
 Masters in Intelligent Systems
 Evolutionary Computation
 1st year, 2nd semester
 Authors:
 Sancho Amaral Simões, 2019217590, uc2019217590@student.uc.pt
 Tiago Filipe Santa Ventura, 2019243695, uc2019243695@student.uc.pt
 Credits to:
 Ernesto Costa
 João Macedo
 Coimbra, 12th May 2023
 ---------------------------------------------------------------------------
"""
import copy
from typing import Callable
import numpy as np
from random import random, randint
from matplotlib import pyplot as plt
from base_gp_algorithm import function_wrappers
from base_gp_algorithm.base_gp_algorithm import BaseGPAlgorithm
from base_gp_algorithm.fitness_functions import sse
from grammar_based_gp.crossover_operators import one_point_crossover


class GrammarBasedGPAlgorithm(BaseGPAlgorithm):
    """
       A genetic programming algorithm based on a given grammar.

       Parameters:
       -----------
       problem_file_path: str
           The path to the file containing the problem definition.
       grammar: dict
           A dictionary containing the grammar rules.
       num_runs: int = BaseGPAlgorithm.DEFAULT_NUM_RUNS
           The number of times to run the algorithm.
       num_generations: int = BaseGPAlgorithm.DEFAULT_NUM_GENERATIONS
           The number of generations to run the algorithm for.
       population_size: int = BaseGPAlgorithm.DEFAULT_POPULATION_SIZE
           The number of individuals in the population.
       prob_mutation: float = BaseGPAlgorithm.DEFAULT_PROB_MUTATION_NODE
           The probability of a mutation occurring.
       genotype_size: int = DEFAULT_GENOTYPE_SIZE
           The size of the genotype.
       grammar_axiom: str = DEFAULT_GRAMMAR_AXIOM
           The starting symbol of the grammar.
       grammar_wrapper: int = DEFAULT_GRAMMAR_WRAPPER
           The number of non-terminals to wrap around a terminal in the grammar.
       prob_crossover: float = BaseGPAlgorithm.DEFAULT_PROB_CROSSOVER
           The probability of a crossover occurring.
       func_crossover: Callable = DEFAULT_FUNC_CROSSOVER
           The crossover function to use.
       tournament_size: int = BaseGPAlgorithm.DEFAULT_TOURNAMENT_SIZE
           The number of individuals to select for a tournament.
       elite_size: float = BaseGPAlgorithm.DEFAULT_ELITE_SIZE
           The percentage of the population to select as elite individuals.
       inject_random_foreigners: bool = True
           Whether or not to inject random foreign individuals into the population.
       random_foreigners_injected_size: float = BaseGPAlgorithm.DEFAULT_RANDOM_FOREIGNERS_INJECTED_SIZE
           The percentage of the population that is made up of random foreign individuals.
       random_foreigners_injection_period: int = BaseGPAlgorithm.DEFAULT_RANDOM_FOREIGNERS_INJECTION_PERIOD
           The period of time at which to inject random foreign individuals.
       fitness_function: Callable = BaseGPAlgorithm.DEFAULT_FITNESS_FUNCTION
           The fitness function to use.
       target_fitness: float = DEFAULT_TARGET_FITNESS
           The target fitness to achieve.
       invalid_fitness: float = BaseGPAlgorithm.DEFAULT_INVALID_FITNESS
           The fitness value to assign to individuals that violate constraints.
       func_selection_survivors: Callable = BaseGPAlgorithm.DEFAULT_FUNC_SELECTION_SURVIVORS
           The function to use for selecting survivors.
       func_selection_parents: Callable = BaseGPAlgorithm.DEFAULT_FUNC_SELECTION_PARENTS
           The function to use for selecting parents.
       normalize: bool = True
           Whether or not to normalize fitness values.
       seed_rng: int = None
           The seed for the random number generator.
       use_multiprocessing: bool = False
           Whether or not to use multiprocessing to speed up computation.
       log_file_path: str = BaseGPAlgorithm.DEFAULT_LOG_FILE_PATH
           The path to the file where the log will be written.
       verbose: bool = True
           Whether or not to print status updates during the algorithm.
    """
    DEFAULT_GENOTYPE_SIZE = 256
    DEFAULT_GRAMMAR_AXIOM = 'start'
    DEFAULT_GRAMMAR_WRAPPER = 2
    DEFAULT_FUNC_CROSSOVER = one_point_crossover
    DEFAULT_TARGET_FITNESS = 0.0
    __name__ = 'Grammar-based GP'

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
                 target_fitness: float = DEFAULT_TARGET_FITNESS,
                 invalid_fitness: float = BaseGPAlgorithm.DEFAULT_INVALID_FITNESS,
                 func_selection_survivors: Callable = BaseGPAlgorithm.DEFAULT_FUNC_SELECTION_SURVIVORS,
                 func_selection_parents: Callable = BaseGPAlgorithm.DEFAULT_FUNC_SELECTION_PARENTS,
                 normalize: bool = True,
                 seed_rng: int = None,
                 use_multiprocessing: bool = False,
                 log_file_path: str = BaseGPAlgorithm.DEFAULT_LOG_FILE_PATH,
                 verbose: bool = True):

        super().__init__(problem_file_path, num_runs, num_generations, population_size,
                         prob_mutation, prob_crossover, tournament_size, elite_size, inject_random_foreigners,
                         random_foreigners_injected_size, random_foreigners_injection_period, fitness_function,
                         target_fitness, invalid_fitness, func_selection_survivors, func_selection_parents, normalize,
                         seed_rng,
                         use_multiprocessing, log_file_path, verbose)

        self.func_crossover = func_crossover
        self.grammar = grammar
        self.genotype_size = genotype_size
        self.grammar_axiom = grammar_axiom
        self.grammar_wrapper = grammar_wrapper

    def plot_results(self, results):
        """
            Function to plot the performance results of a genetic algorithm optimization.

            Parameters:
            -----------
            results: list
                A list of lists where each inner list contains tuples of the form (individual, fitness) for each generation.

            Returns:
            --------
            None
                The function generates plots and saves them to disk, but does not return any values.
        """
        plt.ioff()
        fig, ax = plt.subplots()
        ax.set_xlim([0, self.num_generations + 2])
        bests = np.array([[t[-1] for t in lst] for lst in results])
        np.savetxt(fname='bests_ge.txt', X=bests, delimiter=' ', fmt='%.10f')
        mn = np.mean(bests, axis=0)
        ax.plot([i for i in range(1, self.num_generations + 2)], mn, 'g-s', label='Mean best')
        # create a boxplot
        _ = ax.boxplot(bests, widths=0.25)
        ax.set_xlabel('Generation')
        ax.set_ylabel(self.func_fitness.__name__)
        plt.title('Performance over generation | %d runs' % self.num_runs)
        plt.legend(fontsize=12)
        plt.savefig('box_plot_ge.png')
        #plt.show()

        flat_list = [tup for sublist in results for tup in sublist]

        # Find the tuple with the maximum value in the second position
        max_tuple = min(flat_list, key=lambda x: x[1])

        # Get the phenotype associated with the maximum value
        ind = max_tuple[0]

        cases = self.fit_cases

        x = [ds[:-1] for ds in cases]
        y = [ds[-1] for ds in cases]

        y_pred = []

        print(ind)
        for ds in cases:
            ind_cp = ind[:]
            for i in range(len(ds) - 1):
                ind_cp = ind_cp.replace('x[' + str(i) + ']', str(ds[i]))
            y_pred.append(eval(ind_cp))

        data = np.asarray([[x[i][0], y[i]] for i in range(len(x))])
        data_pred = np.asarray([[x[i][0], y_pred[i]] for i in range(len(x))])
        if self.normalize:
            data = np.asarray(self.scaler.inverse_transform(data))
            data_pred = np.asarray(self.scaler.inverse_transform(data_pred))

        sse_de_norm = sse(data, data_pred)
        plt.figure()
        plt.title(self.problem_name + ' | Real vs Predicted')
        plt.xlabel('Input variable')
        plt.ylabel('Output variable')
        plt.plot(data[:, 0], data[:, 1], label='Real')
        plt.plot(data_pred[:, 0], data_pred[:, 1], label='Predicted')
        plt.legend(loc='best')
        plt.title('Best of bests | %s=%.8f' % (self.func_fitness.__name__, sse_de_norm))
        plt.legend(loc='best')
        plt.savefig('best_of_bests_ge.png')
        ##plt.show()

    def _gp(self, run_id: int):
        """
            Runs the genetic programming algorithm for a given run ID.

            Parameters:
            -----------
            run_id: int
                ID number of the current run.

            Returns:
            --------
            List[Tuple[Any, float]]
                A list of tuples containing the best individuals and their corresponding fitness scores for each generation.
        """
        self._log('Starting run no %d...', (run_id,), run_id)
        self._reset()

        self._log('Initializing population...', (), run_id)
        self.population[run_id] = self.generate_initial_population()
        self.population[run_id] = [self._evaluate(individual) for individual in self.population[run_id]]
        self.population[run_id].sort(key=lambda x: x[1])
        best = (self._mapping(self.population[run_id][0][0]), self.population[run_id][0][1])
        self.best_individual[run_id].append(best)
        elite_size_ = int(self.population_size * self.elite_size)

        best_fitnesses = [-1 for _ in range(self.num_generations)]
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(self.__name__ + ' Training - Best')
        ax.set_xlabel('Generation')
        ax.set_ylabel(self.func_fitness.__name__)
        ax.set_ylim(0, 10)
        line_best, = ax.plot([gen for gen in range(self.num_generations)],
                             best_fitnesses, 'r-', label='best')

        for gen in range(self.num_generations):
            self.population[run_id] = [self._evaluate(individual) for individual in self.population[run_id]]
            self.population[run_id].sort(key=lambda x: x[1])

            best = (self._mapping(self.population[run_id][0][0]), self.population[run_id][0][1])
            self.best_individual[run_id].append(best)
            best_fitnesses[gen] = best[1]
            self._log('Gen %d - Best fitness %.10f', (gen, best[1],), run_id)
            ax.set_ylim(0, max(best_fitnesses) + max(best_fitnesses) / 10)
            line_best.set_ydata(best_fitnesses)
            fig.canvas.draw()
            fig.canvas.flush_events()

            if best[1] <= self.target_fitness:
                self._log('Gen %d Target fitness %.10f reached. Terminating...', (gen, best[1],), run_id)
                self.end()

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

        plt.savefig('avg_ge_%d.png' % run_id)
        plt.close('all')

        return self.best_individual[run_id]

    def generate_initial_population(self):
        """
          Generates an initial population of random individuals.

          Returns:
          --------
          list
              A list of `population_size` individuals, each represented as a list with two elements:
              a genotype (a list of integers between 0 and 256) and a `None` fitness value.
        """
        return [self.generate_random_individual() for _ in range(self.population_size)]

    def generate_random_individual(self):
        """
            Generates a random individual.

            Returns:
            --------
            list
                An individual, represented as a list with two elements:
                a genotype (a list of integers between 0 and 256) and a `None` fitness value.
        """
        genotype = [randint(0, 256) for _ in range(self.genotype_size)]

        return [genotype, None]

    def mutate(self, parent):
        """
           Mutates an individual.

           Parameters:
           -----------
           parent: list
               An individual, represented as a list with two elements:
               a genotype (a list of integers between 0 and 256) and a fitness value.

           Returns:
           --------
           list
               The mutated individual, represented as a list with two elements:
               a genotype (a list of integers between 0 and 256) and a `None` fitness value.
        """
        parent = copy.deepcopy(parent)
        parent[1] = None

        for i in range(len(parent[0])):
            if random() < self.prob_mutation:
                parent[0][i] = randint(0, 256)

        return parent

    def _mapping(self, genotype):
        """
        Maps a genotype to a phenotype using the grammar defined in the class.

        Parameters:
        -----------
        genotype : list
            A list of integers that represents the genotype to be mapped.

        Returns:
        --------
        str or None
            The phenotype obtained after expanding the genotype using the grammar,
            or None if it was impossible to expand.

        Notes:
        ------
        The function uses a depth-first expansion strategy to derive the phenotype
        from the genotype. It wraps the genotype when it is necessary to do so, and
        selects randomly among the possible productions when there is more than one
        for a non-terminal symbol. This function is intended to be used as a helper
        function by the main genetic programming algorithm.
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
        """
            Evaluates the fitness of an individual using the phenotype obtained from its genotype.

            Parameters:
            -----------
            individual : list
                A list containing the genotype of the individual and its fitness value.

            Returns:
            --------
            list
                The updated list containing the genotype and the new fitness value.

            Notes:
            ------
            The function first maps the genotype to a phenotype using the `_mapping` method.
            If it was possible to derive a phenotype, it evaluates the individual's fitness
            by comparing the predicted output of the phenotype to the expected output in the
            given test cases. The fitness value is computed using the `func_fitness` function
            provided to the class during initialization. If it was impossible to derive a
            phenotype, the individual is assigned the `invalid_fitness` value, indicating that
            it cannot be evaluated.
        """
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
