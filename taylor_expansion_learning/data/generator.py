"""
Taylor Series Data Generator Module

This module contains functionality for generating mathematical functions
and their corresponding Taylor series expansions using SymPy.
"""

import json
import random
import sympy as sp


class TaylorDataGenerator:
    """
    Generates symbolic functions and their Taylor expansions using SymPy.
    
    This class handles the creation of a dataset containing mathematical
    functions and their corresponding Taylor expansions up to a specified order.
    
    Attributes:
        config (dict): Configuration parameters for data generation.
        x (sympy.Symbol): Symbol used for the variable in functions.
        functions (list): List of generated function strings.
        expansions (list): List of corresponding Taylor expansion strings.
    """
    
    def __init__(self, config):
        """
        Initialize the Taylor data generator.
        
        Args:
            config (dict): Configuration parameters including data generation settings.
        """
        self.config = config
        self.x = sp.Symbol('x')
        self.functions = []
        self.expansions = []

    def generate_valid_function(self):
        """
        Generate a random function that is valid for Taylor expansion.
        
        Creates functions by selecting from basic mathematical functions and
        applying various transformations.
        
        Returns:
            sympy.Expr: A symbolic mathematical function.
        """
        # Define a set of basic functions that work well with Taylor expansions
        basic_functions = [
            self.x,             # Identity function
            sp.sin(self.x),     # Sine
            sp.cos(self.x),     # Cosine
            sp.exp(self.x),     # Exponential
            sp.log(1 + self.x), # Logarithm (shifted to avoid singularity)
            1/(1 - self.x),     # Geometric series
            sp.sqrt(1 + self.x),# Square root (shifted to avoid branch cut)
            sp.tan(self.x),     # Tangent
            sp.atan(self.x),    # Arctangent
            sp.sinh(self.x),    # Hyperbolic sine
            sp.cosh(self.x),    # Hyperbolic cosine
        ]

        # Select a function from the basic set
        func = random.choice(basic_functions)

        # With some probability, apply some transformations
        if random.random() < 0.7:
            # Apply scaling
            a = random.randint(1, 5)
            if random.random() < 0.5:
                func = func.subs(self.x, a * self.x)  # Scale the input
            else:
                func = a * func  # Scale the output

        if random.random() < 0.5:
            # Simple additions/multiplications with polynomials
            degree = random.randint(0, 3)
            poly = sum(random.randint(-3, 3) * self.x**i for i in range(degree + 1))
            if random.random() < 0.5:
                func = func + poly  # Addition
            else:
                if random.random() < 0.7:  # Avoid division by zero issues
                    func = func * (poly + 1)  # Multiplication (with offset to avoid zeros)

        return func

    def compute_taylor_expansion(self, func, x0, order):
        """
        Compute Taylor expansion of the function around x0 up to given order.
        
        Args:
            func (sympy.Expr): Function to expand.
            x0 (int/float): Point around which to expand.
            order (int): Order of Taylor expansion.
            
        Returns:
            sympy.Expr: Taylor expansion or None if an error occurred.
        """
        try:
            expansion = sp.series(func, self.x, x0=x0, n=order+1).removeO()
            return expansion
        except (sp.core.sympify.SympifyError, TypeError, ValueError, ZeroDivisionError):
            return None

    def generate_dataset(self):
        """
        Generate a dataset of functions and their Taylor expansions.
        
        Returns:
            tuple: Lists of function strings and their corresponding Taylor expansions.
        """
        print("Generating dataset of functions with Taylor expansions...")

        successful_count = 0
        attempts = 0
        max_attempts = self.config['data_generation']['num_functions'] * 3

        while successful_count < self.config['data_generation']['num_functions'] and attempts < max_attempts:
            attempts += 1

            # Generate a function
            func = self.generate_valid_function()

            # Compute Taylor expansion
            expansion = self.compute_taylor_expansion(
                func,
                self.config['data_generation']['x0'],
                self.config['data_generation']['expansion_order']
            )

            if expansion is not None:
                self.functions.append(str(func))
                self.expansions.append(str(expansion))
                successful_count += 1

                if successful_count % 100 == 0:
                    print(f"Generated {successful_count} valid functions")

        print(f"Dataset generation complete. Generated {len(self.functions)} functions.")

        return self.functions, self.expansions

    def save_dataset(self, output_path):
        """
        Save the generated dataset to a JSON file.
        
        Args:
            output_path (str): Path to save the dataset.
        """
        data = {
            'functions': self.functions,
            'expansions': self.expansions,
            'config': self.config['data_generation']
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Dataset saved to {output_path}")
