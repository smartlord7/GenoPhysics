import matplotlib
from hyperopt import fmin, tpe, hp
from base_gp_algorithm.fitness_functions import sigmoid, sse
from tree_based_gp.tree_based_gp_algorithm import TreeBasedGPAlgorithm
from base_gp_algorithm.survivors_selection import survivors_elite


def main():
    # Set the backend for matplotlib
    matplotlib.use('TkAgg')

    # Constants
    FILE_PATH = '../../../../data/solar_system.txt'
    NUM_GENERATIONS = 75
    NUM_OPT_RUNS = 100

    # Define the search space for each hyperparameter
    SEARCH_SPACE = {
        'population_size': hp.quniform('population_size', 100, 500, 50),
        'prob_mutation': hp.uniform('prob_mutation', 0.1, 0.5),
        'prob_crossover': hp.uniform('prob_crossover', .5, 0.9),
        'random_foreigners_injected_size': hp.uniform('random_foreigners_injected_size', 0.1, 0.5),
        'random_foreigners_injection_period': hp.quniform('random_foreigners_injection_period', 5, 50, 5),
        'tournament_size': hp.quniform('tournament_size', 2, 5, 1),
        'elite_size': hp.uniform('elite_size', 0.1, 0.3)
    }

    # Define the objective function
    def objective(params):
        # Create a new instance of the tree-based genetic programming algorithm with the given hyperparameters
        elite_survivors = survivors_elite(params['elite_size'])
        gp = TreeBasedGPAlgorithm(FILE_PATH,
                                  population_size=int(params['population_size']),
                                  prob_mutation=params['prob_mutation'],
                                  prob_crossover=params['prob_crossover'],
                                  random_foreigners_injected_size=params['random_foreigners_injected_size'],
                                  random_foreigners_injection_period=int(params['random_foreigners_injection_period']),
                                  elite_size=float(params['elite_size']),
                                  tournament_size=int(params['tournament_size']),
                                  num_generations=NUM_GENERATIONS,
                                  func_selection_survivors=elite_survivors,
                                  inject_random_foreigners=True,
                                  log_file_path='gp_hyperopt.log',
                                  num_runs=1,
                                  seed_rng=1,
                                  fitness_function=sse)

        # Execute the genetic programming algorithm and return the negative of the maximum fitness value
        results = gp.execute()
        gp.plot_results(results)
        return min(gp.best_fitness)

    # Perform the hyper-parameter optimization
    BEST = fmin(
        fn=objective,
        space=SEARCH_SPACE,
        algo=tpe.suggest,
        max_evals=NUM_OPT_RUNS
    )

    # Print the best hyper-parameters found
    print('Best hyperparameters:', BEST)


if __name__ == '__main__':
    main()