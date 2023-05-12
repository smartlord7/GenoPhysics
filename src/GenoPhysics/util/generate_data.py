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

import matplotlib.pyplot as plt
import numpy as np



def data_creation(file_name, function, domain, numb_cases, *function_set):
    """
    Generate fitness cases for a given function and save them to a file.

    Parameters:
    -----------
    - file_name: str
        The name of the file to save the fitness cases to.
    - function: callable
        The function to generate fitness cases for.
    - domain: list of tuples
        The domain of the function. Each tuple contains the starting and ending values of a variable.
    - numb_cases: int
        The number of fitness cases to generate.
    - *function_set: list of tuples
        A list of tuples containing the name and arity of each function used to generate the fitness cases.

    Returns:
    -----------
        None
    """
    with open(file_name, 'w') as f_out:
        # Header
        header_base = str(len(domain)) + '\t'
        header = header_base + ''.join(
            [function_set[i][0] + '\t' + str(function_set[i][1]) + '\t' for i in range(len(function_set))])[:-1] + '\n'
        f_out.write(header)

        # generate fitness cases
        cases = []
        for i in range(len(domain)):
            init = domain[i][0]
            end = domain[i][1]
            step = (end - init) / numb_cases
            data_x = np.arange(init, end, step)
            cases.append(data_x)
        inputs = list(zip(*cases))
        outputs = [function(*in_vector) for in_vector in inputs]
        # save to file
        for i in range(numb_cases):
            line = ''.join([str(val) + '\t' for val in inputs[i]]) + str(outputs[i]) + '\n'
            f_out.write(line)
        f_out.close()

def plot_data(file_name, name, body, loc='best'):
    """
    Plot the data from a file containing fitness cases.

    Parameters:
    -----------
    - file_name: str
        The name of the file containing the fitness cases.
    - name: str
        The name of the plot.
    - body: str
        The label for the plot's data.
    - loc: str
        The location of the plot's legend.

    Returns:
    --------
        None
    """
    with open(file_name) as f_in:
        # read fitness cases
        data = f_in.readlines()[1:]
        # plot
        values_x = [float(line[:-1].split()[0]) for line in data]
        values_y = [float(line[:-1].split()[-1]) for line in data]
        plt.ylabel('Value')
        plt.xlabel('Input Values')
        plt.title(name)
        p = plt.plot(values_x, values_y, 'r-o', label=str(body))
        plt.legend(loc=loc)
        plt.show()

def simbolic_regression_2(x):
    """
    Function to calculate symbolic regression with 2 variables

    Parameters:
    -----------
    x: float
        Input value for the function

    Returns:
    --------
    float
        Output value of the function
    """
    return x ** 2 + x + 1


def sphere(x, y):
    """
    Function to calculate the sphere function with 2 variables

    Parameters:
    -----------
    x: float
        Input value for the first variable
    y: float
        Input value for the second variable

    Returns:
    --------
    float
        Output value of the function
    """
    return x ** 2 + y ** 2


def sin_w(x):
    """
    Function to calculate sin(x)

    Parameters:
    -----------
    x: float
        Input value for the function

    Returns:
    --------
    float
        Output value of the function
    """
    return np.sin(x)


def wave(x):
    """
    Function to calculate sin(x) + 5*cos(x) - 2*sin(2*x)

    Parameters:
    -----------
    x: float
        Input value for the function

    Returns:
    --------
    float
        Output value of the function
    """
    return np.sin(x) + 5 * np.cos(x) - 2 * np.sin(2 * x)


def main():
    """
    Main function to generate datasets and plots for different functions
    """
    # my file prefix
    prefix = ''
    data_creation(prefix + 'data_sin.txt', sin_w, [(2 * -3.14, 2 * 3.14)], 62, ('add_w', 2), ('sub_w', 2), ('mult_w', 2),
                  ('div_prot_w', 2))
    plot_data(prefix + 'data_sin.txt', 'sin', 'sin(x)', 'upper right')

    data_creation(prefix + 'data_wave.txt', wave, [(2 * -3.14, 2 * 3.14)], 62, ('add_w', 2), ('sub_w', 2), ('mult_w', 2),
                  ('div_prot_w', 2))
    plot_data(prefix + 'data_wave.txt', 'Wave', 'sin(x) + 5 * np.cos(x) - 2*sin(2*x)', 'upper right')

    data_creation(prefix + 'data_symb.txt', simbolic_regression_2, [(-1.0, 1.0)], 21, ('add_w', 2), ('sub_w', 2),
                  ('mult_w', 2), ('div_prot_w', 2))
    plot_data(prefix + 'data_symb.txt', 'Symbolic Regression', 'x**2 + x + 1', 'lower right')

    data_creation(prefix + 'data_sphere.txt', sphere, [[-5, 5], [-5, 5]], 22, ('add_w', 2), ('sub_w', 2), ('mult_w', 2),
                  ('div_prot_w', 2))
    plot_data(prefix + 'data_sphere.txt', 'Sphere', 'X**2 + Y**2', 'upper right')


if __name__ == '__main__':
    main()