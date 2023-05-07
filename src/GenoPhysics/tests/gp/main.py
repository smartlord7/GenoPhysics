from matplotlib import pyplot as plt

from genetic_programming.genetic_programming_algorithm import GeneticProgrammingAlgorithm

if __name__ == '__main__':
    filename_1 = 'data_sphere.txt'

    gp = GeneticProgrammingAlgorithm(filename_1)
    gp.start()

    plt.show()
