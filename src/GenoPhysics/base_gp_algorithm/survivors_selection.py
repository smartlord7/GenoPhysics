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
import math
from operator import itemgetter


def survivors_generational(population, offspring):
    """
    Replaces all individuals in the population with the new offspring.

    Parameters:
    -----------
    population: list
        List of individuals representing the current population.
    offspring: list
        List of individuals representing the offspring generated in the current generation.

    Returns:
    --------
    list
        List of individuals representing the updated population.
    """
    return offspring


def survivors_best(population, offspring):
    """
    Combines the current population with the offspring and selects the best individuals.

    Parameters:
    -----------
    population: list
        List of individuals representing the current population.
    offspring: list
        List of individuals representing the offspring generated in the current generation.

    Returns:
    --------
    list
        List of individuals representing the updated population, where only the best individuals are kept.
    """
    new_pop = population + offspring
    new_pop.sort(key=itemgetter(1), reverse=False)

    return new_pop[:len(population)]


def survivors_elite(elite_size):
    """
       Returns a function that selects the best individuals from both the current population and the offspring,
       while keeping a certain number of elite individuals from the previous generation.

       Parameters:
       -----------
       elite_size: float
           Proportion of individuals from the current population to keep as elite individuals.

       Returns:
       --------
       function
           Function that takes in the current population and offspring, and returns the updated population
           with the best individuals from both populations, and the specified number of elite individuals from the
           previous generation.
    """
    def survivors(population, offspring):
        """
            Replaces the current population with the offspring, keeping only the best individuals.

            Parameters:
            -----------
            population: list
                List of individuals representing the current population.
            offspring: list
                List of individuals representing the offspring generated in the current generation.

            Returns:
            --------
            list
                List of individuals representing the updated population, where only the best individuals are kept.
        """
        size = math.ceil(elite_size * len(population))
        population.sort(key=itemgetter(1), reverse=False)
        offspring.sort(key=itemgetter(1), reverse=False)

        return population[:size] + offspring[:-size]

    return survivors