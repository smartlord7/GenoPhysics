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
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def main():
    """
        Main function to calculate and plot the Kepler 3rd Law for the Solar System.

        Returns:
        --------
        None
    """
    matplotlib.use('TkAgg')
    FILE_PATH = '../../../data/solar_system_raw.txt'

    # Constants
    G = 6.67430e-11 * (365.25 * 24 * 60 * 60) ** 2 * (1e-10) ** 3  # Gravitational constant in (10^10 m)^3 kg^-1 yr^-2
    M_sun = 1.9885e30  # Mass of the sun in kg

    # Load data from file
    data = np.loadtxt(FILE_PATH, delimiter=' ')

    # Convert distance to meters
    distance = data[:, 0] * 1e10

    # Convert period to seconds
    period = data[:, 1] * 365.25 * 24 * 60 * 60

    # Calculate semi-major axis
    a = distance / (1 - (period ** 2 * G * M_sun) / (4 * np.pi ** 2 * distance ** 3)) ** (1 / 2)

    # Calculate SSE
    SSE = np.sum((a - distance) ** 2)

    # Plot results
    plt.scatter(distance, period, label='Real')
    plt.plot(a, period, '--', label='Predicted')
    plt.xlabel('Distance (10^10 m)')
    plt.ylabel('Period (s)')
    plt.title('Kepler 3rd Law for the Solar System')
    plt.legend()
    plt.show()

    print("SSE of the Kepler Law fit:", SSE)


if __name__ == '__main__':
    main()