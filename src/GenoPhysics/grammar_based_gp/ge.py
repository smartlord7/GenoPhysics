#  Grammatical Evolution
# Implementation of the canonical version of Grammatical Evolution
# (see Ryan, C., Collins, J., Neill, M.O. (1998). Grammatical evolution: Evolving programs for an arbitrary language.
# In: Banzhaf, W., Poli, R., Schoenauer, M., Fogarty, T.C. (eds) Genetic Programming. EuroGP 1998. Lecture Notes
# in Computer Science, vol 1391. Springer, Berlin, Heidelberg.)


# imports
import sys
import random
import copy
import numpy as np
import matplotlib.pyplot as plt


# General Code
# Main evolutionary loop
def main(generations, population_size, genotype_size, elite_size, p_cross, p_mut, evaluate, mapping, choose_indiv,
         crossover, mutation):
    """
    @elite-size: is an integer
    """
    bests = []
    population = generate_initial_population(population_size, genotype_size)
    for it in range(generations):
        population = [evaluate(indiv) for indiv in population]
        population.sort(key=lambda x: x['fitness'])
        best = (mapping(population[0]['genotype']), population[0]['fitness'])
        bests.append(best)
        print('Best at', it, best[1], best[0])
        new_population = population[:elite_size]
        for it in range(elite_size, population_size):
            if random.random() < p_cross:
                p1 = choose_indiv(population)
                p2 = choose_indiv(population)
                ni = crossover(p1, p2)
            else:
                # tournament
                ni = choose_indiv(population)
            ni = mutation(ni, p_mut)
            new_population.append(ni)
        population = new_population

    return bests[-1]


# Generating an individual and population
def generate_random_individual(size):
    # individual = dictionary (genotype, fitness)
    genotype = [random.randint(0, 256) for c in range(size)]
    return {'genotype': genotype, 'fitness': None}


def generate_initial_population(population_size, genotype_size):
    return [generate_random_individual(genotype_size) for i in range(population_size)]


# Compute the fitness of an individual
def fitness(individual, dataset):
    """
    @individual is a string and so to compute the result we use eval.
    @dataset: list of lists [x_1...x_n y]
    Fitness is equal to the mean squared error made by the individual
    when executed for each value of the dataset.
    Works for one or many inputs and one output.
    """
    b = []
    for ds in dataset:
        # instantiate the individual = replace all variables by values from the data set
        ind = individual[:]
        for i in range(len(ds) - 1):
            ind = ind.replace('x[' + str(i) + ']', str(ds[i]))
        # evaluate the individual = run the program
        b.append(eval(ind))
    # fitness = mean squared error
    mse = np.mean([(b[i] - dataset[i][-1]) ** 2 for i in range(len(b))])
    return mse


# From genotype to phenotype
def mapping_func(grammar_axiom, wrapping, grammar):
    """
    @wrapping: number of accepted wraps.
    @grammar: our problem.
    """

    def _mapping(genotype):
        """
        @used_gene: pointer for the integer to be used to decide the production rule
        @production_options: list of production alternatives
        @symbols_to_expand: frontier of the derivation tree. treated as a stack for a depth-first expansion
        @output: word generated = individual
        @current_production: index of the chosen production
        """
        wraps = 0
        used_gene = 0
        output = []
        production_options = []
        symbols_to_expand = [grammar_axiom]
        while (wraps < wrapping) and (len(symbols_to_expand) > 0):
            # test: end of genotype but still symbols to expand ==> wrap
            if used_gene % len(genotype) == 0 and used_gene > 0 and len(production_options) > 1:
                wraps += 1
                used_gene = 0
            current_symbol = symbols_to_expand.pop(0)
            if current_symbol in grammar.keys():  # Non terminal?
                production_options = grammar[current_symbol]
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

    return _mapping


# evaluating an individual
def eval_func(mapping, invalid_fitness, dataset):
    def _evaluate(ind):
        pheno = mapping(ind['genotype'])
        if pheno is not None:
            ind['fitness'] = fitness(pheno, dataset)
        else:
            ind['fitness'] = invalid_fitness
        return ind

    return _evaluate


# Parents' selection
def parent_sel(tournament):
    """
    Determninistic tournament.
    Minimization!
    """

    def _choose_indiv(population):
        # minimization
        pool = random.sample(population, tournament)
        pool.sort(key=lambda i: i['fitness'])
        return copy.deepcopy(pool[0])

    return _choose_indiv


# Variation operators
def one_point_crossover(p1, p2):
    at = random.randint(0, len(p1['genotype']) - 1)
    genotype = [i < at and p1['genotype'][i] or p2['genotype'][i] for i in range(len(p1['genotype']))]
    return {'genotype': genotype, 'fitness': None}


def uniform_crossover(p1, p2):
    mask = [random.choice([True, False]) for i in range(len(p1['genotype']))]
    genotype = [mask[i] and p1['genotype'][i] or p2['genotype'][i] for i in range(len(p1['genotype']))]
    return {'genotype': genotype, 'fitness': None}


def mutate(p, p_mut):
    p = copy.deepcopy(p)
    p['fitness'] = None
    for i in range(len(p['genotype'])):
        if (random.random() < p_mut):
            p['genotype'][i] = random.randint(0, 256)
    return p


# Visualization

def plot_behaviour(individual, dataset, name_func):
    # decompose into input (x) and output(y)
    x = [ds[:-1] for ds in dataset]
    y = [ds[-1] for ds in dataset]
    # run the individual for each input
    y_pred = []
    for ds in dataset:
        ind = individual[0][:]
        for i in range(len(ds) - 1):
            ind = ind.replace('x[' + str(i) + ']', str(ds[i]))
        y_pred.append(eval(ind))

    # visualize
    # one input variable only
    plt.title(name_func)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.plot(x, y, label='Y')
    plt.plot(x, y_pred, label='Y_pred')
    plt.legend(loc='best')
    plt.show()


#  Examples

# Functions
def mult(a, b):
    return a * b


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def pdiv(a, b):
    if (abs(b) <= 1e-3):
        return 1
    else:
        return a / b


def sin(a):
    return np.sin(a)


def cos(a):
    return np.cos(a)


def exp(a):
    try:
        a = np.exp(a)
        if (np.isinf(a)):
            a = sys.maxsize
    except:
        a = sys.maxsize
    return a


def log(a):
    if (a != 0):
        try:
            return np.log(abs(a))
        except:
            return 0
    return 0


if __name__ == '__main__':

    dataset = get_fit_cases('data_sin.txt')
    name = 'Sin'
    # dataset = get_fit_cases('data_symb.txt')
    # name = 'Symbolic Regression'

    grammar = {

        'start': [['expr']],
        'expr': [
            ['op', '(', 'expr', ',', 'expr', ')'],
            ['var']],
        'op': [
            ['mult'],
            ['add'],
            ['sub'],
            ['pdiv']],
        'var': [
            ['x[0]'],
            ['1.0']]
    }
    # more than one input variable?
    for i in range(1, len(dataset[0]) - 1):
        grammar['var'].append(['x[' + str(i) + ']'])

    grammar_axiom = 'start'

    # grammar for including unary operators
    grammar_op1 = {

        'start': [['expr']],
        'expr': [
            ['op2', '(', 'expr', ',', 'expr', ')'],
            ['op1', '(', 'expr', ')'],
            ['var']],
        'op1': [
            ['sin'],
            ['cos'],
            ['log'],
            ['exp']
        ],
        'op2': [
            ['mult'],
            ['add'],
            ['sub'],
            ['pdiv']],
        'var': [
            ['x[0]'],
            ['1.0']]
    }

    # Configuration
    pop_size = 20
    generations = 10
    elite_size = 1
    genotype_size = 256
    wrapping = 2
    tournament = 3
    p_cross = 0.8
    p_mut = 2.0 / genotype_size

    invalid_fitness = sys.maxsize  # minimization...

    _mapping = mapping_func(grammar_axiom, wrapping, grammar)
    _evaluate = eval_func(_mapping, invalid_fitness, dataset)
    _choose_indiv = parent_sel(tournament)
    _crossover = one_point_crossover
    _mutation = mutate
    # _crossover = uniform_crossover

    best = main(generations, pop_size, genotype_size, elite_size, p_cross, p_mut, _evaluate, _mapping, _choose_indiv,
                _crossover, _mutation)
    plot_behaviour(best, dataset, name)







