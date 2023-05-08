def one_point_crossover(p1, p2):
    at = randint(0, len(p1['genotype']) - 1)
    genotype = [i < at and p1['genotype'][i] or p2['genotype'][i] for i in range(len(p1['genotype']))]
    return [genotype, None]


def uniform_crossover(p1, p2):
    mask = [random.choice([True, False]) for i in range(len(p1['genotype']))]
    genotype = [mask[i] and p1['genotype'][i] or p2['genotype'][i] for i in range(len(p1['genotype']))]
    return [genotype, None]