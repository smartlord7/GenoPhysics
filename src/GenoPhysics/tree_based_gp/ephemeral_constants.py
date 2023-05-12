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
from random import uniform


def uniform_ephemeral(mn: float = -0.5, mx: float = 0.5):
    """
    Function to generate an ephemeral uniform random number generator.

    Parameters:
    -----------
    mn : float, optional
        Lower bound of the uniform distribution (default=-0.5).
    mx : float, optional
        Upper bound of the uniform distribution (default=0.5).

    Returns:
    --------
    callable
        A function that generates an ephemeral uniform random number on each call.
    """
    def uniform_ephemeral_():
        """
        Generates an ephemeral uniform random number.

        Returns:
        --------
        float
            A random number generated from a uniform distribution.
        """
        return uniform(mn, mx)

    return uniform_ephemeral_

