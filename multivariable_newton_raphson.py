from sympy import symbols
import sympy as sp
import numpy as np


# converting sympy matrix to numpy array
# this works for n X 1
def sympy_matrix_to_numpy_array(sym_matrix):
    vector = []
    for sym in sym_matrix:
        vector.append(float(sym.evalf()))
    numpy_array = np.array(vector)
    return numpy_array


# taking equations and returning the matrix form of the equations
def equations_to_matrix(equations_list):
    functions_as_matrix = sp.Matrix(equations_list)
    return functions_as_matrix


# inserting parameters to a symbolic-matrix
def insert_parameters_to_matrix_with_symbols(symbolic_matrix, symbols, parameters):
    symbols_list = list(symbols)
    dic = {}
    i = 0
    for sym in symbols_list:
        dic[sym] = parameters[i]
        i += 1
    numeric_matrix = symbolic_matrix.subs(dic)
    return numeric_matrix


# using the formula multivariable-newton-raphson and returning the result
# All parameters of type sympy matrix. For example: vector = sp.Matrix([0, 2])
def multivariable_newton_raphson_formula(guess_vector, jacob_inv_matrix_parameters, functions_as_matrix_parameters):
    result = guess_vector - jacob_inv_matrix_parameters * functions_as_matrix_parameters
    return result


# multivariable-newton-raphson
# printing all the calculations
# if there is a result then the function returns the result, otherwise - returns 0
def multivariable_newton_raphson(equations_list, symbols, initial_guess_vector, desired_epsilon, num_after_point=5,
                                 max_iterations=100):
    symbols_list = list(symbols)
    print(f'the symbols in use: {symbols_list}')
    functions_as_matrix = equations_to_matrix(equations_list)
    print(f'functions as matrix: {functions_as_matrix}')
    jacobian_matrix = functions_as_matrix.jacobian(symbols_list)
    print(f'jacobian matrix: {jacobian_matrix}')
    jacobian_matrix_inv = jacobian_matrix.inv()
    print(f'jacobian matrix inverted: {jacobian_matrix_inv}')

    print()

    x0 = initial_guess_vector
    parameters = []
    for parm in x0:
        parameters.append(parm)

    for iteration in range(max_iterations):
        print(f' ----------- n = {iteration} -----------')
        print(f'guess vector: {x0.evalf(num_after_point)}')
        jacobian_matrix_parameters = insert_parameters_to_matrix_with_symbols(jacobian_matrix, symbols_list, parameters)
        print(f'jacobian matrix with parameters: {jacobian_matrix_parameters.evalf(num_after_point)}')
        jacob_inv_matrix_parameters = insert_parameters_to_matrix_with_symbols(jacobian_matrix_inv, symbols_list,
                                                                               parameters)
        print(f'jacobian matrix inverted with parameters: {jacob_inv_matrix_parameters.evalf(num_after_point)}')
        functions_as_matrix_parameters = insert_parameters_to_matrix_with_symbols(functions_as_matrix, symbols_list,
                                                                                  parameters)
        print(f'functions as matrix parameters: {functions_as_matrix_parameters.evalf(num_after_point)}')
        x1 = multivariable_newton_raphson_formula(x0, jacob_inv_matrix_parameters, functions_as_matrix_parameters)
        print(f'result from multivariable-newton-raphson formula: {x1.evalf(num_after_point)}')
        delta = x1 - x0
        print(f'delta = x1 - x0: {delta.evalf(num_after_point)}')
        delta_numeric = sympy_matrix_to_numpy_array(delta)
        # print(f'delta_numeric: {delta_numeric}')
        norm = np.linalg.norm(delta_numeric)
        print(f'matrix norm of delta = x1 - x0: {norm}')
        formatted_number = "{:.{}g}".format(norm, num_after_point)
        print(f'rounded matrix norm of delta = x1 - x0: {formatted_number}')

        if norm < desired_epsilon:
            return x1

        x0 = x1
        parameters = []
        for parm in x0:
            parameters.append(parm)

        print()

    print("max iterations reached before the norm < epsilon")
    print("you can change the max iterations by adding an argument: 'max_iterations=[type an integer here]' to the "
          "function"" 'multivariable_newton_raphson'")
    return 0


if __name__ == '__main__':
    # ----- EXAMPLE -----
    x, y = sp.symbols('x y')  # initializing symbols
    eq1 = x ** 2 + 3 * y ** 2 - 10  # first equation
    eq2 = 5 * x - y ** 3 - 5  # second equation
    eps = 0.001  # epsilon
    equations_list = [eq1, eq2]  # putting the equations in a list
    symbols_list = [x, y]  # putting the symbols in a list
    initial_guess_vector = sp.Matrix([2, 2])  # initializing the first guess vector
    result = multivariable_newton_raphson(equations_list, symbols_list, initial_guess_vector, eps)  # THE METHOD
    # ----- END EXAMPLE -----
    '''
    if you want to change how much numbers after the point each number will show and/or increase/decrease the 
    number of iterations follow the comment bellow:

    The default values are num_after_point=5, max_iterations=100. in the line of code below I changed the default values 
    by adding the arguments: num_after_point=10, max_iterations=200

    result = multivariable_newton_raphson(equations_list, symbols_list, initial_guess_vector, eps, 
        num_after_point=10, max_iterations=200)  # THE METHOD 2
    '''
