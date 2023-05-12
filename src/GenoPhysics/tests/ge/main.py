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
import math
import matplotlib
from base_gp_algorithm.fitness_functions import rmse, sse
from base_gp_algorithm.survivors_selection import survivors_elite
from grammar_based_gp.grammar_based_gp_algorithm import GrammarBasedGPAlgorithm
from hyperopt import fmin, tpe, hp

if __name__ == '__main__':
    """
      Function to run a genetic programming algorithm using a defined grammar and fitness function.
      Reads data from a file and generates a population of individuals with a specified genotype size.
      The algorithm runs for a set number of generations, selecting survivors through a tournament style selection.
      Elite individuals are guaranteed to survive each generation. The best individual of each generation is tracked
      and returned along with the population at each generation. The results are plotted at the end of the run.

      Parameters:
      -----------
      None

      Returns:
      --------
      None
    """
    matplotlib.use('QtAgg')
    file_name = '../../../../data/solar_system.txt'
    grammar = {

        'start': [['expr']],
        'expr': [
            ['op', '(', 'expr', ',', 'expr', ')'],
            ['var']],
        'op': [
            ['function_wrappers.mult_w'],
            ['function_wrappers.add_w'],
            ['function_wrappers.sub_w'],
            ['function_wrappers.div_prot_w']],
        'var': [
            ['x[0]'],
            ['1.0']]
    }

    gp = GrammarBasedGPAlgorithm(file_name,
                                 grammar,
                                 population_size=900,
                                 genotype_size=512,
                                 prob_mutation=0.034031659639900486,
                                 prob_crossover=0.71504110934997,
                                 elite_size=0.16595647596149682,
                                 tournament_size=80,
                                 # Fixed
                                 log_file_path='ge_runs.log',
                                 normalize=True,
                                 num_generations=75,
                                 func_selection_survivors=survivors_elite,
                                 use_multiprocessing=False,
                                 num_runs=30,
                                 seed_rng=1,
                                 fitness_function=sse)
    #plot_data()
    results = gp.execute()
    gp.plot_results(results)
