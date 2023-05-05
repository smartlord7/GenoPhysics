#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
sgp_sol.py
Implementation of GP. Simple version. Inspired by tinyGP by R. Poli
Ernesto Costa, March 2012
Adapted for Python 3 -  March 2015
Revised for runs - March 2016
Revised April 2022

Individuals are represented by a pair [indiv,fit]
where indiv is an individual represented recursively as a list of lists. For example, f(t_1,...,t_n)
is represented as [f, rep(t_1), ..., rep(t_n)]

"""
import matplotlib.pyplot as plt
from random import random, choice, uniform, sample, seed
from types import FunctionType
from operator import itemgetter
from copy import deepcopy
import math

MIN_RND = -5
MAX_RND = 5


# Evolver
def sgp(problem, numb_gen, pop_size, in_max_depth, max_len, prob_mut_node, prob_cross, mutation, crossover,
        select_parent, select_survivors, seed_=False):
    """
    Problem dependent data, i.e., terminal set, function set and fitness cases are kept in a file.
    Could be implemented as a class object...

    problem = file name where the data for the problem are stored
    num_gen = number of generations
    pop_size = population size
    in_max_depth = max depth for initial individuals
    max_len = maximum number of nodes
    prob_mut_node = mutation probability for node mutation
    prob_cross = crossover probability (mutation probability = 1 - prob_cross)
    t_size = tournament size
    seed = define seed for the random generator. Default = False.

    """
    # initialize the random numbers generator
    if seed_:
        seed(123456789)
    # Extract information about the problem.  problem is the name of
    # the file where that information is stored
    # Fitness Cases = [[X1,...,Xn Y], ...]
    header, fit_cases = get_data(problem)
    # Header = Numb_Input_Vars, Function_Set
    numb_vars, function_set = header
    vars_set = generate_vars(numb_vars)
    ephemeral_constant = 'uniform(MIN_RND,MAX_RND)'
    const_set = [ephemeral_constant]
    terminal_set = vars_set + const_set
    # Define initial population
    chromosomes = ramped_half_and_half(function_set, terminal_set, pop_size, in_max_depth)
    # Evaluate population
    population = [[chromo, evaluate(chromo, fit_cases)] for chromo in chromosomes]
    best_indiv, best_fitness = best_indiv_population(population)
    print('Best at Generation %d:\n%s\nFitness: %s\n----------------' % (0, best_indiv, best_fitness))
    # Evolve
    for i in range(numb_gen):
        # offspring after variation
        offspring = []
        for j in range(pop_size):
            if random() < prob_cross:
                # subtree crossover
                parent_1 = select_parent(population)
                parent_2 = select_parent(population)
                new_offspring = crossover(parent_1, parent_2)
            else:  # prob mutation = 1 - prob crossover!
                # mutation
                parent = select_parent(population)
                new_offspring = mutation(parent, prob_mut_node, function_set, vars_set, const_set)
            offspring.append(new_offspring)
        # Evaluate new population (offspring)
        offspring = [[chromo, evaluate(chromo, fit_cases)] for chromo in offspring]
        # Merge parents and offspring
        population = select_survivors(population, offspring)
        # Statistics
        best_indiv, best_fitness = best_indiv_population(population)
        print('Best at Generation %d:\n%s\nFitness: %s\n----------------' % (i + 1, best_indiv, best_fitness))

    print('\n\nFINAL BEST\n%s\nFitness ---> %f\n\n' % (best_indiv, best_fitness))
    return best_fitness


# --------------------------------------- Variation operators
# Crossover
def sub_tree(tree, position):
    def sub_tree_aux(tree, position):
        global count
        if position == count:
            count = 0
            return tree
        else:
            count += 1
            if isinstance(tree, list):
                for i, sub in enumerate(tree[1:]):
                    res_aux = sub_tree(sub, position)
                    if res_aux:
                        break
                return res_aux

    return sub_tree_aux(tree, position)


def replace_sub_tree(tree, sub_tree_1, sub_tree_2):
    if tree == sub_tree_1:
        return sub_tree_2
    elif isinstance(tree, list):
        for i, sub in enumerate(tree[1:]):
            res = replace_sub_tree(sub, sub_tree_1, sub_tree_2)
            if res and (res != sub):
                return [tree[0]] + tree[1:i + 1] + [res] + tree[i + 2:]
        return tree
    else:
        return tree


def subtree_crossover(par_1, par_2):
    """ATENTION:if identical sub_trees it replaces the first ocorrence..."""
    # Choose crossover point (indepently)
    size_1 = indiv_size(par_1)
    size_2 = indiv_size(par_2)
    cross_point_1 = choice(list(range(size_1)))
    cross_point_2 = choice(list(range(size_2)))
    # identify subtrees to echange
    sub_tree_1 = sub_tree(par_1, cross_point_1)
    sub_tree_2 = sub_tree(par_2, cross_point_2)
    # Exchange
    new_par_1 = deepcopy(par_1)
    offspring = replace_sub_tree(new_par_1, sub_tree_1, sub_tree_2)
    return offspring


# Mutation
def point_mutation(par, prob_mut_node, func_set, vars_set, const_set):
    par_mut = deepcopy(par)
    if random() < prob_mut_node:
        if isinstance(par_mut, list):
            # Function
            symbol = par_mut[0]
            return [change_function(symbol, func_set)] + [
                point_mutation(arg, prob_mut_node, func_set, vars_set, const_set) for arg in par_mut[1:]]
        elif isinstance(par_mut, (float, int)):
            # It's a constant
            return eval(const_set[0])
        elif var_b(par_mut):
            # It's a variable
            return change_variable(par_mut, vars_set)
        else:
            raise TypeError  # should not happen
    return par_mut


def change_function(symbol, function_set):
    new_function = choice(function_set)
    while (new_function[0] == symbol) or (new_function[1] != arity(symbol, function_set)):
        new_function = choice(function_set)
    return new_function[0]


def arity(symbol, function_set):
    for func in function_set:
        if func[0] == symbol:
            return func[1]


def change_variable(variable, vars_set):
    if len(vars_set) == 1:
        return variable
    new_var = choice(vars_set)
    while new_var == variable:
        new_var = choice(vars_set)
    return new_var


# ------------------------------------- Population
# Generate an individual: method full or grow
# FGGP: algorithm 2.1, pg.14
def gen_rnd_expr(func_set, term_set, max_depth, method):
    """Generation of tree structures using full or grow."""
    if (max_depth == 0) or (method == 'grow'
                            and (random() <
                                 (len(term_set) / (len(term_set) + len(func_set))))):
        index = choice(list(range(len(term_set))))
        if index == (len(term_set) - 1):
            # ephemeral constant
            ephemeral_const = term_set[index]
            expr = eval(ephemeral_const)
        else:
            # variable: 'Xn'
            expr = term_set[index]
    else:
        func = choice(func_set)
        # func = [name_function, arity]
        expr = [func[0]] + [gen_rnd_expr(func_set, term_set, max_depth - 1, method)
                            for i in range(int(func[1]))]
    return expr


# Method ramped half-and-half.
def ramped_half_and_half(func_set, term_set, size, max_depth):
    depth = list(range(3, max_depth))
    pop = []
    for i in range(size // 2):
        pop.append(gen_rnd_expr(func_set, term_set, choice(depth), 'grow'))
    for i in range(size // 2):
        pop.append(gen_rnd_expr(func_set, term_set, choice(depth), 'full'))
    if (size % 2) != 0:
        pop.append(gen_rnd_expr(func_set, term_set, choice(depth), 'full'))
    return pop


# ------------------------------------------ Parents' Selection

def tournament(size):
    """Maximization Problem.Deterministic"""

    def choose_parent(population):
        pool = sample(population, size)
        pool.sort(key=itemgetter(1), reverse=True)
        return pool[0][0]

    return choose_parent


# ---------------------------------------------- Survivors' Selection

def survivors_generational(population, offspring):
    """Change all population with the new individuals."""
    return offspring


# ------------------------------------------------ Fitness Evaluation
def evaluate(individual, fit_cases):
    """
    Evaluate an individual. *** Maximization ***
    Gives the inverse of the sum of the absolute error for each fitness cases.
    fit_cases = [[X1, ..., XN, Y], ...]
    The smaller the error the better the fitness.
    """
    indiv = deepcopy(individual)
    error = 0
    for case in fit_cases:
        result = interpreter(indiv, case[:-1])
        error += abs(result - case[-1])
    return 1.0 / (1.0 + error)


# Interpreter. FGGP, algorithm 3.1 - pg.25
def interpreter(indiv, variables):
    if isinstance(indiv, list):
        func = eval(indiv[0])
        if isinstance(func, FunctionType) and (len(indiv) > 1):
            # Function: evaluate
            value = func(*[interpreter(arg, variables) for arg in indiv[1:]])
        else:
            # Macro: don't evaluate arguments
            value = indiv
    elif isinstance(indiv, (float, int)):
        # It's a constant
        value = indiv
    elif var_b(indiv):
        # It's a variable
        index = get_var_index(indiv)
        value = variables[index]  # binding value
    elif isinstance(eval(indiv), FunctionType):
        # Terminal 0-ary function: execute
        value = eval(indiv)(*())
    return value


## Auxiliary
# get data for problem
def get_data(file_problem):
    """
    the problem is defined in a file.
    the first line of the file is the header
    the other lines are the fitness cases.
    """
    with open(file_problem, 'r') as f_in:
        data = f_in.readlines()
        # header
        header_line = data[0]
        header_line = header_line.split()
        header = [int(header_line[0])] + [[[header_line[i], int(header_line[i + 1])]
                                           for i in range(1, len(header_line), 2)]]
        # fitness cases
        fit_cases_str = [case.split() for case in data[1:]]
        fit_cases = [[float(elem) for elem in case] for case in fit_cases_str]
        return header, fit_cases


def get_var_index(var):
    return int(var[1:])


def generate_vars(n):
    """ generate n vars, X1, ..., Xn."""
    vars_set = []
    for i in range(n):
        vars_set.append('X' + str(i))
    return vars_set


def var_b(name):
    """Test: is name a variable?"""
    return isinstance(name, str) and (name[0] == 'X') and (name[1:].isdigit())


def indiv_size(indiv):
    """ Number of nodes of an individual."""
    if not isinstance(indiv, list):
        return 1
    else:
        return 1 + sum(map(indiv_size, indiv[1:]))


def best_indiv_population(population):
    # max value of fitness
    all_fit_values = [indiv[1] for indiv in population]
    max_fit = max(all_fit_values)
    # index max value
    index_max_fit = all_fit_values.index(max_fit)
    # find indiv
    return population[index_max_fit]


# ---------------------   SOLUTIONS   -----------------------
# Problema 7.3
def sgp_for_plot(problem, numb_gen, pop_size, in_max_depth, max_len, prob_mut_node, prob_cross, mutation, crossover,
                 select_parent, select_survivors, seed_=False, plot=True):
    """
    Problem dependent data, i.e., terminal set, function set and fitness cases are kept in a file.
    Could be implemented as a class object...

    problem = file name where the data for the problem are stored
    num_gen = number of generations
    pop_size = population size
    in_max_depth = max depth for initial individuals
    max_len = maximum number of nodes
    prob_mut_node = mutation probability for node mutation
    prob_cross = crossover probability (mutation probability = 1 - prob_cross)
    t_size = tournament size
    seed = define seed for the random generator. Default = False.

    """
    # initialize the random numbers generator
    if seed_:
        seed(123456789)
    # Extract information about the problem.  problem is the name of
    # the file where that information is stored
    # Fitness Cases = [[X1,...,Xn Y], ...]
    header, fit_cases = get_data(problem)
    # Header = Numb_Input_Vars, Function_Set
    numb_vars, function_set = header
    vars_set = generate_vars(numb_vars)
    ephemeral_constant = 'uniform(MIN_RND,MAX_RND)'
    const_set = [ephemeral_constant]
    terminal_set = vars_set + const_set

    # Define initial population
    chromosomes = ramped_half_and_half(function_set, terminal_set, pop_size, in_max_depth)
    # Evaluate population
    population = [[chromo, evaluate(chromo, fit_cases)] for chromo in chromosomes]
    best_indiv, best_fitness = best_indiv_population(population)
    print('Best at Generation %d:\n%s\nFitness: %s\n----------------' % (0, best_indiv, best_fitness))
    # Statistics
    if plot:
        best_fit_gener = [best_fitness]
        average = sum([fit for chromo, fit in population]) / len(population)
        ave_fit_gener = [average]
    # Evolve
    for i in range(numb_gen):
        # offspring after variation
        offspring = []
        for j in range(pop_size):
            if random() < prob_cross:
                # subtree crossover
                parent_1 = select_parent(population)
                parent_2 = select_parent(population)
                new_offspring = crossover(parent_1, parent_2)
            else:  # prob mutation = 1 - prob crossover!
                # mutation
                parent = select_parent(population)
                new_offspring = mutation(parent, prob_mut_node, function_set, vars_set, const_set)
            offspring.append(new_offspring)
        # Evaluate new population (offspring)
        offspring = [[chromo, evaluate(chromo, fit_cases)] for chromo in offspring]
        # Merge parents and offspring
        population = select_survivors(population, offspring)
        # Statistics
        best_indiv, best_fitness = best_indiv_population(population)
        print('Best at Generation %d:\n%s\nFitness: %s\n----------------' % (i + 1, best_indiv, best_fitness))
        # Statistics
        if plot:
            best_fit_gener.append(best_fitness)
            average = sum([fit for chromo, fit in population]) / len(population)
            ave_fit_gener.append(average)

    print('\n\nFINAL BEST\n%s\nFitness ---> %f\n\n' % (best_indiv, best_fitness))
    if plot:
        return best_fit_gener, ave_fit_gener
    return best_fitness


# display
def display_one_run(best, average_pop, titulo):
    # Show plot
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.title(titulo)
    p1 = plt.plot(best, 'r-o', label="Pop Best")
    p2 = plt.plot(average_pop, 'g-s', label="Pop Average")
    plt.legend(loc='best')
    plt.show()


#  Problem 7.4
def run_for_plot(num_runs, target, problem, numb_gen, pop_size, in_max_depth, max_len, prob_mut_node, prob_cross,
                 mutation, crossover, select_parent, select_survivors, seed=False):
    # Colect data
    print('Wait, please ')
    estatistica_total = [
        sgp_for_plot(problem, numb_gen, pop_size, in_max_depth, max_len, prob_mut_node, prob_cross, mutation, crossover,
                     select_parent, select_survivors, seed, True) for i in range(num_runs)]
    print("That's it!")
    best_total = [best for best, average_pop in estatistica_total]
    average_total = [average_pop for best, average_pop in estatistica_total]
    # Process Data: best and average
    best_genera = list(zip(*best_total))
    best = [max(genera) for genera in best_genera]
    average_genera = list(zip(*average_total))
    average = [sum(genera) / num_runs for genera in average_genera]
    # Show plot
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    titulo = f'Target: {target} Runs: {num_runs}, Prob. Mutation Node: {prob_mut_node:0.2f}, Crossover: {prob_cross:0.2f}'
    plt.title(titulo)
    p1 = plt.plot(best, 'r-o', label="Best")
    p2 = plt.plot(average, 'g-s', label="Average")

    plt.legend(loc='best')
    plt.show()


# Prtoblem 7.5

def survivors_best(population, offspring):
    """ join both and select the best."""
    new_pop = population + offspring
    new_pop.sort(key=itemgetter(1), reverse=True)
    return new_pop[:len(population)]


def survivors_elite(elite_size):
    def survivors(pop, off):
        size = math.ceil(elite_size * len(pop))
        pop.sort(key=itemgetter(1), reverse=True)
        off.sort(key=itemgetter(1), reverse=True)
        return pop[:size] + off[:-size]

    return survivors


# ----------------------------- Function Set Wrappers
def add_w(x, y):
    return x + y


def mult_w(x, y):
    return x * y


def sub_w(x, y):
    return x - y


def div_prot_w(x, y):
    if abs(y) <= 1e-3:
        return 1
    else:
        return x / y


if __name__ == '__main__':
    # my file prefix
    prefix = '/Users/ernestocosta/tmp/'
    filename_1 = 'data_symb.txt'
    file_name_2 = 'data_sin.txt'
    file_name_3 = 'data_sphere.txt'
    # parameters
    count = 0
    numb_runs = 10
    problem = prefix + filename_1
    numb_gen = 100
    pop_size = 30
    in_max_depth = 6
    max_len = 1000
    prob_mut_node = 0.01
    prob_cross = 0.9  # p_mutation = 1 - p_crossover
    tour_size = 3
    elite_size = 0.1
    my_seed = False

    mutation = point_mutation
    crossover = subtree_crossover
    select_parent = tournament(tour_size)
    # select_survivors = survivors_generational
    select_survivors = survivors_elite(elite_size)
    # select_survivors = survivors_best

    # best_g, average_pop_g = sgp_for_plot(problem,numb_gen,pop_size, in_max_depth, max_len,prob_mut_node, prob_cross, mutation, crossover, select_parent, select_survivors,False, True)
    # isplay_one_run(best_g, average_pop_g,'Symbolic Regression')
    run_for_plot(numb_runs, 'Symbolic Regression', problem, numb_gen, pop_size, in_max_depth, max_len, prob_mut_node,
                 prob_cross, mutation, crossover, select_parent, select_survivors)
    # run_for_plot_survivors(numb_runs,'Simbolic Regression',problem,numb_gen,pop_size,in_max_depth,max_len,prob_mut_node,prob_cross,tour_size, survivors,seed)
    # sgp(problem,numb_gen,pop_size,in_max_depth,max_len,prob_mut_node,prob_cross, mutation, crossover, select_parent,select_survivors,my_seed)



