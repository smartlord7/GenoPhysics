"""
------------GenoPhysics: Kepler's Third Law of Planetary Motion------------
 University of Coimbra
 Masters in Intelligent Systems
 Evolutionary Computation
 1st year, 2nd semester
 Authors:
 Sancho Amaral Simões, 2019217590, uc2019217590@student.uc.pt
 Tiago Filipe Santa Ventura, 2019243695, uc2019243695@student.uc.pt
 Credits to:
 Ernesto Costa
 João Macedo
 Coimbra, 12th May 2023
 ---------------------------------------------------------------------------
"""
from operator import itemgetter
from random import sample


def tournament(population, size):
    """
        Function to perform tournament selection on a population of individuals.

        Parameters:
        -----------
        population: list
            A list of tuples representing individuals in the population.
            Each tuple should contain the individual's genetic information
            and fitness score.
        size: int
            The size of the tournament pool.

        Returns:
        --------
        tuple
            A tuple representing the selected individual with the highest fitness score.
            The tuple should contain the individual's genetic information and fitness score.
    """
    pool = sample(population, size)
    pool.sort(key=itemgetter(1), reverse=False)

    return pool[0]
