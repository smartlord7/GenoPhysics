from operator import itemgetter
from random import sample


def tournament(population, size):
    pool = sample(population, size)
    pool.sort(key=itemgetter(1), reverse=False)

    return pool[0]
