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
from typing import Any
from types import FunctionType
from base_gp_algorithm import function_wrappers


def get_var_index(var):
    """
    Get the index of a variable name.

    Parameters:
    -----------
    var : str
        The name of the variable.

    Returns:
    --------
    int
        The index of the variable.
    """
    return int(var[1:])


def generate_vars(n_vars: int, suffix: str = 'X'):
    """
    Generate a list of variable names.

    Parameters:
    -----------
    n_vars : int
        The number of variables to generate.
    suffix : str, optional
        The suffix to append to the variable names (default='X').

    Returns:
    --------
    list
        A list of variable names.
    """
    vars_set = []

    for i in range(n_vars):
        vars_set.append(suffix + str(i))

    return vars_set


def is_var(name):
    """
    Check if a string is a variable name.

    Parameters:
    -----------
    name : str
        The string to check.

    Returns:
    --------
    bool
        True if the string is a variable name, False otherwise.
    """
    return isinstance(name, str) and (name[0] == 'X') and (name[1:].isdigit())


def individual_size(indiv):
    """
    Compute the number of nodes in an individual.

    Parameters:
    -----------
    indiv : any
        The individual to compute the size of.

    Returns:
    --------
    int
        The number of nodes in the individual.
    """
    if not isinstance(indiv, list):
        return 1
    else:
        return 1 + sum(map(individual_size, indiv[1:]))

# Interpreter. FGGP, algorithm 3.1 - pg.25
def interpreter(individual, variables):
    """
    Interprets a genetic programming tree and returns the resulting value.

    Parameters:
    -----------
    individual: list, float, int, or str
        The genetic programming tree to interpret. Can be a list of nested sub-trees, a float or int constant, a string
        variable name, or a string representing a terminal 0-ary function.
    variables: list of floats
        A list of values to bind to the variable names in the genetic programming tree.

    Returns:
    --------
    The value of the interpreted genetic programming tree.
    """
    value = None
    if isinstance(individual, list):
        try:
            func = eval(individual[0])
            if isinstance(func, FunctionType) and (len(individual) > 1):
                # Function: evaluate
                value = func(*[interpreter(arg, variables) for arg in individual[1:]])
            else:
                # Macro: don't evaluate arguments
                value = individual
        except Exception as e:
            print('Unknown error')

    elif isinstance(individual, (float, int)):
        # It's a constant
        value = individual
    elif is_var(individual):
        # It's a variable
        index = get_var_index(individual)
        value = variables[index]  # binding value
    else:
        # Terminal 0-ary function: execute
        value = individual()

    return value


def is_float(string):
    """
    Checks if a given string can be converted to a float.

    Parameters:
    -----------
    string: str
        The string to check.

    Returns:
    --------
    bool
        True if the string can be converted to a float, False otherwise.
    """
    try:
        float(string)

        return True
    except ValueError:

        return False


def tree_to_inline_expression(tree: list, decimal_places: int = 9) -> tuple[str, Any]:
    """
    Converts a genetic programming tree to a simplified symbolic expression.

    Parameters:
    -----------
    tree: list
        The genetic programming tree to convert.
    decimal_places: int, optional
        The number of decimal places to round constants to in the resulting expression. Default is 9.

    Returns:
    --------
    tuple of str and sympy.Expr
        The first element of the tuple is a string representation of the unsimplified expression. The second element
        is the resulting simplified symbolic expression.
    """
    non_simplified_expr = tree_to_inline_expression_(tree)
    expr = sp.sympify(non_simplified_expr)
    simplified_expr = sp.simplify(expr)
    simplified_expr = sp.factor(simplified_expr)
    simplified_expr = sp.N(simplified_expr, decimal_places)

    return non_simplified_expr, simplified_expr


def tree_to_inline_expression_(tree):
    """
    Convert a tree expression into an inline expression.

    Parameters:
    -----------
    tree: list
        Tree expression to convert.

    Returns:
    --------
    str
        Inline expression.
    """
    if isinstance(tree, list):
        operator = ' ' + eval(tree[0]).__annotations__['symbol'] + ' '
        operands = [tree_to_inline_expression_(subtree) for subtree in tree[1:]]

        return f'({operator.join(operands)})'
    else:
        return str(tree)


def simplify_expression(expr):
    """
    Simplify a math expression defined by functions and variables.

    Parameters:
    -----------
    expr: str
        Math expression to simplify.

    Returns:
    --------
    str
        Simplified math expression.

    Raises:
    -------
    ValueError
        If the function name in the expression is unknown.
    """
    # Split the expression into the function and the arguments
    func_name, args = expr.split('(', 1)
    args = args[:-1]  # Remove the closing parenthesis
    # Split the arguments into a list of arguments
    args_list = []
    arg = ''
    depth = 0
    for c in args:
        if c == ',' and depth == 0:
            args_list.append(arg)
            arg = ''
        else:
            arg += c
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
    args_list.append(arg)
    # Simplify the arguments recursively
    args_simp = [simplify_expression(a) if '(' in a else a for a in args_list]
    # Simplify the function
    if func_name == 'function_wrappers.add_w':
        return '({} + {})'.format(args_simp[0], args_simp[1])
    elif func_name == 'function_wrappers.sub_w':
        return '({} - {})'.format(args_simp[0], args_simp[1])
    elif func_name == 'function_wrappers.mult_w':
        return '({} * {})'.format(args_simp[0], args_simp[1])
    elif func_name == 'function_wrappers.div_prot_w':
        return '({} / {})'.format(args_simp[0], args_simp[1])
    elif func_name == 'function_wrappers.sin_w':
        return 'sin({})'.format(args_simp[0])
    elif func_name == 'function_wrappers.cos_w':
        return 'cos({})'.format(args_simp[0])
    elif func_name == 'function_wrappers.exp_w':
        return 'exp({})'.format(args_simp[0])
    elif func_name == 'x':
        return 'x{}'.format(int(args_simp[0]))
    else:
        raise ValueError('Unknown function: {}'.format(func_name))

