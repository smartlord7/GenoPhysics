import matplotlib
from matplotlib import pyplot as plt

from genetic_programming.genetic_programming_algorithm import GeneticProgrammingAlgorithm

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    filename_1 = 'solar_system.txt'

    gp = GeneticProgrammingAlgorithm(filename_1, num_runs=30, num_generations=100, use_multiprocessing=True)
    gp.start()
