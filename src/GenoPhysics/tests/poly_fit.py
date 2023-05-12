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
import sympy as sp
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def main():
    """
       Function to fit a polynomial to data describing the Kepler's third law of the solar system and plot the results.

       Parameters:
       -----------
       None

       Returns:
       --------
       None
    """
    matplotlib.use('TkAgg')
    FILE_PATH = '../../../data/solar_system_raw.txt'
    # Load data from file
    data = np.loadtxt(FILE_PATH, delimiter=' ')

    # Extract the two columns of data
    x = data[:, 0]
    y = data[:, 1]

    # Choose the degree of the polynomial fit
    degree = 3

    # Fit the data with a polynomial of the chosen degree
    coefficients = np.polyfit(x, y, degree)

    m = list(coefficients[:len(coefficients) - 1])
    b = coefficients[-1]

    sym = sp.symbols('x')
    poly = sp.poly_from_expr(sum(c * sym ** i for i, c in enumerate(m[::-1])) + b)[0]
    print("Polynomial:", poly)

    # Convert the polynomial to a string representation
    poly_str = sp.poly(poly, sym).as_expr()
    print("Polynomial expression:", poly_str)

    # Calculate the predicted values of y for each x
    y_fit = np.polyval(coefficients, x)

    # Calculate the SSE of the fit
    SSE = np.sum((y - y_fit) ** 2)

    # Plot the original data and the predicted data
    plt.scatter(x, y, label='Real')
    plt.plot(x, y_fit,  '--',  label='Predicted')
    plt.legend()
    plt.xlabel('Distance (10^10 m)')
    plt.ylabel('Period (s)')
    plt.title('Kepler 3rd Law for the Solar System - Polynomial Fit with Degree {}'.format(degree))
    plt.show()

    print("SSE of the polynomial fit:", SSE)


if __name__ == '__main__':
    main()