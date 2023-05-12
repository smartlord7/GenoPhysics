import matplotlib

from base_gp_algorithm.fitness_functions import sigmoid, sse
from tree_based_gp.tree_based_gp_algorithm import TreeBasedGPAlgorithm
from base_gp_algorithm.survivors_selection import survivors_elite

if __name__ == '__main__':
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
                              num_generations=75,
                              func_selection_survivors=elite_survivors,
                              inject_random_foreigners=True,
                              log_file_path='gp_runs3.log',
                              num_runs=10,
                              seed_rng=3,

                              fitness_function=sse)
    gp.plot_data()
    results = gp.execute()
    gp.plot_results(results)
