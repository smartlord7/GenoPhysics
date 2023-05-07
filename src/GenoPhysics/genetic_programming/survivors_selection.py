import math
from operator import itemgetter


def survivors_generational(population, offspring):
    """Change all population with the new individuals."""
    return offspring


def survivors_best(population, offspring):
    """ join both and select the best."""
    new_pop = population + offspring
    new_pop.sort(key=itemgetter(1), reverse=True)

    return new_pop[:len(population)]


def survivors_elite(elite_size):
    def survivors(population, offspring):
        size = math.ceil(elite_size * len(population))
        population.sort(key=itemgetter(1), reverse=True)
        offspring.sort(key=itemgetter(1), reverse=True)

        return pop[:size] + offspring[:-size]

    return survivors