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
def symbol(value):
    """
        Decorator that allows adding a symbol to a function's annotations dictionary.

        Parameters:
        -----------
        value : str
            A string representing the symbol to add to the function's annotations dictionary.

        Returns:
        --------
        Callable
            A decorator function that takes a function as an argument and adds the specified symbol to its annotations dictionary.
    """
    def decorator(func):
        func.__annotations__['symbol'] = value

        return func

    return decorator


@symbol('+')
def add_w(x, y):
    """
        Function to perform addition of two input values.

        Parameters:
        -----------
        x : float
            Input value for the first variable.
        y : float
            Input value for the second variable.

        Returns:
        --------
        float
            Output value of the addition operation.
    """
    return x + y


@symbol('*')
def mult_w(x, y):
    """
        Function to perform multiplication of two input values.

        Parameters:
        -----------
        x : float
            Input value for the first variable.
        y : float
            Input value for the second variable.

        Returns:
        --------
        float
            Output value of the multiplication operation.
    """
    return x * y


@symbol('-')
def sub_w(x, y):
    """
        Function to perform subtraction of two input values.

        Parameters:
        -----------
        x : float
            Input value for the first variable.
        y : float
            Input value for the second variable.

        Returns:
        --------
        float
            Output value of the subtraction operation.
    """
    return x - y


@symbol('/')
def div_prot_w(x, y):
    """
        Function to perform division of two input values, protecting against divide-by-zero errors.

        Parameters:
        -----------
        x : float
            Input value for the numerator.
        y : float
            Input value for the denominator.

        Returns:
        --------
        Union[float, int]
            Output value of the division operation. If the denominator is within a small tolerance of zero (1e-3),
            the function returns 1 to avoid divide-by-zero errors.
    """
    if abs(y) <= 1e-3:
        return 1
    else:
        return x / y



