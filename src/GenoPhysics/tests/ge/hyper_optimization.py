import matplotlib
from base_gp_algorithm.fitness_functions import rmse, sse
from base_gp_algorithm.survivors_selection import survivors_elite
from grammar_based_gp.grammar_based_gp_algorithm import GrammarBasedGPAlgorithm
from hyperopt import fmin, tpe, hp

if __name__ == '__main__':
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

    # Define the search space for each parameter
    search_space = {
        'population_size': hp.quniform('population_size', 500, 1000, 100),
        'genotype_size': hp.quniform('genotype_size', 128, 512, 32),
        'prob_mutation': hp.uniform('prob_mutation', 0.01, 0.4),
        'prob_crossover': hp.uniform('prob_crossover', 0.3, 0.8),
        'elite_size': hp.uniform('elite_size', 0.1, 0.3),
        'tournament_size': hp.quniform('tournament_size', 20, 100, 20),
    }

    # Define the objective function
    def objective(params):
        # Create a new instance of the GP algorithm with the given hyperparameters
        gp = GrammarBasedGPAlgorithm(file_name,
                                     grammar,
                                     population_size=int(params['population_size']),
                                     genotype_size=int(params['genotype_size']),
                                     prob_mutation=params['prob_mutation'],
                                     prob_crossover=params['prob_crossover'],
                                     elite_size=params['elite_size'],
                                     tournament_size=int(params['tournament_size']),
                                     num_generations=100,
                                     inject_random_foreigners=True,
                                     func_selection_survivors=survivors_elite,
                                     log_file_path='ge_hyperopt.log',
                                     num_runs=1,
                                     seed_rng=1,
                                     fitness_function=sse)
        # Execute the GP algorithm and return the fitness value

        #gp.plot_data()
        best = gp.execute()[0][-1] # get the fitness of the last one, since using elite
        #gp.plot_results(best)

        return best[1]

    # Perform the optimization

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
    )

    # Print the best hyperparameters found
    print('Best hyperparameters:', best)
