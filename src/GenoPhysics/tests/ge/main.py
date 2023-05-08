from base_gp_algorithm.fitness_functions import rmse
from grammar_based_gp.grammar_based_gp_algorithm import GrammarBasedGPAlgorithm

if __name__ == '__main__':
    file_name = '../../../../data/trappist1.txt'
    grammar = {

        'start': [['expr']],
        'expr': [
            ['op', '(', 'expr', ',', 'expr', ')'],
            ['var']],
        'op': [
            ['function_wrappers.mult_w'],
            ['function_wrappers.add_w'],
            ['function_wrappers.sub_w'],
            ['function_wrappers.div_prot_w'],
            ['function_wrappers.power_prot_w']],
        'var': [
            ['x[0]'],
            ['1.0']]
    }

    gp = GrammarBasedGPAlgorithm(file_name,
                                 grammar,
                                 elite_size=0.2,
                                 target_fitness=0.001,
                                 num_runs=1,
                                 num_generations=-1,
                                 population_size=200,
                                 prob_mutation=0.6,
                                 fitness_function=rmse)
    results = gp.execute()
    gp.plot_results(results)
