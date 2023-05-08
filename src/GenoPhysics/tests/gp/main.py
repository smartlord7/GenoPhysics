from matplotlib import pyplot as plt

from tree_based_gp.tree_based_gp_algorithm import TreeBasedGPAlgorithm
from tree_based_gp.survivors_selection import survivors_elite

if __name__ == '__main__':
    filename_1 = 'solar_system.txt'
    elite_size = 0.2
    elite_survivors = survivors_elite(elite_size)

    gp = TreeBasedGPAlgorithm(filename_1, num_runs=30,
                              func_selection_survivors=elite_survivors,
                              num_generations=1000,
                              population_size=100,
                              prob_mutation_node=0.2,
                              prob_crossover=0.7,
                              use_multiprocessing=False,
                              verbose=True)
    gp.plot_data()
    gp.start()
