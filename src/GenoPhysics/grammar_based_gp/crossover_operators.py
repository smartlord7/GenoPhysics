from random import choice, randint


def one_point_crossover(p1, p2):
    at = randint(0, len(p1[0]) - 1)

    genotype = [i < at and p1[0][i] or p2[0][i] for i in range(len(p1[0]))]

    return [genotype, None]


def uniform_crossover(p1, p2):
    mask = [choice([True, False]) for i in range(len(p1[0]))]

    genotype = [mask[i] and p1[0][i] or p2[0][i] for i in range(len(p1[0]))]

    return [genotype, None]
