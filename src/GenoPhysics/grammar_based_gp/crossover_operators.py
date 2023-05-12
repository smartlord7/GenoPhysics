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
from random import choice, randint


def one_point_crossover(p1, p2):
    """
        Function to perform one-point crossover between two parents.

        Parameters:
        -----------
        p1: tuple
            A tuple representing the first parent's genotype, where the first element is a list of values.
        p2: tuple
            A tuple representing the second parent's genotype, where the first element is a list of values.

        Returns:
        --------
        list
            A list representing the child's genotype resulting from one-point crossover between the two parents.
            The second element in the returned list is None.
    """
    at = randint(0, len(p1[0]) - 1)

    genotype = [i < at and p1[0][i] or p2[0][i] for i in range(len(p1[0]))]

    return [genotype, None]


def uniform_crossover(p1, p2):
    """
        Function to perform uniform crossover between two parents.

        Parameters:
        -----------
        p1: tuple
            A tuple representing the first parent's genotype, where the first element is a list of values.
        p2: tuple
            A tuple representing the second parent's genotype, where the first element is a list of values.

        Returns:
        --------
        list
            A list representing the child's genotype resulting from uniform crossover between the two parents.
            The second element in the returned list is None.
    """
    mask = [choice([True, False]) for i in range(len(p1[0]))]

    genotype = [mask[i] and p1[0][i] or p2[0][i] for i in range(len(p1[0]))]

    return [genotype, None]
