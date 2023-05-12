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
import matplotlib

from base_gp_algorithm.fitness_functions import sigmoid, sse
from tree_based_gp.tree_based_gp_algorithm import TreeBasedGPAlgorithm
from base_gp_algorithm.survivors_selection import survivors_elite

if __name__ == '__main__':
    """
      The main function that executes the genetic programming algorithm.

      Parameters:
      -----------
      None

      Returns:
      --------
      None
    """
    matplotlib.use('TkAgg')
    filename_1 = '../../../../data/solar_system.txt'
    elite_size = 0.13393895258058527
    elite_survivors = survivors_elite(elite_size)

    gp = TreeBasedGPAlgorithm(filename_1,
                              population_size=350,
                              prob_mutation=0.21995605947639038,
                              prob_crossover=0.8868236215582409,
                              random_foreigners_injected_size=0.20900192638164344,
                              random_foreigners_injection_period=5,
                              elite_size=elite_size,
                              tournament_size=3,
                              # Fixed
                              num_generations=75,
                              func_selection_survivors=elite_survivors,
                              inject_random_foreigners=True,
                              log_file_path='gp_runs.log',
                              num_runs=30,
                              seed_rng=1,
                              fitness_function=sse)
    #gp.plot_data()
    results = gp.execute()
    gp.plot_results(results)
