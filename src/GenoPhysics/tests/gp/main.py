import matplotlib

from base_gp_algorithm.fitness_functions import sigmoid
from tree_based_gp.tree_based_gp_algorithm import TreeBasedGPAlgorithm
from base_gp_algorithm.survivors_selection import survivors_elite

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    filename_1 = '../../../../data/solar_system.txt'
    elite_size = 0.1
    elite_survivors = survivors_elite(elite_size)

    gp = TreeBasedGPAlgorithm(filename_1,
                              population_size=300,
                              prob_mutation=0.3,
                              prob_crossover=0.7,
                              random_foreigners_injected_size=0.5,
                              random_foreigners_injection_period=50,
                              elite_size=0.1,
                              tournament_size=3,
                              # Fixed
                              num_generations=1,
                              func_selection_survivors=elite_survivors,
                              inject_random_foreigners=True,
                              log_file_path='gp_hyperopt.log',
                              num_runs=1,
                              seed_rng=1,
                              fitness_function=sigmoid)
    #gp.plot_data()
    gp.execute()
    #gp.plot_results(list())
