import math
import matplotlib
from base_gp_algorithm.fitness_functions import rmse
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


    gp = GrammarBasedGPAlgorithm(file_name,
                                 grammar,
                                 population_size=800,
                                 genotype_size=256,
                                 prob_mutation=0.2,
                                 prob_crossover=0.6,
                                 random_foreigners_injected_size=0.2,
                                 elite_size=0.15,
                                 tournament_size=3,
                                 random_foreigners_injection_period=50,
                                 # Fixed
                                 num_generations=500,
                                 inject_random_foreigners=True,
                                 func_selection_survivors=survivors_elite,
                                 num_runs=1,
                                 seed_rng=1,
                                 fitness_function=rmse)
    gp.plot_data()
    result = gp.execute()[0]
    #gp.plot_results(results)
