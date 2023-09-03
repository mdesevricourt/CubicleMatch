"""Create an Agent class to store the information of each participant to the matching algorithm to solve their demand"""
import numpy as np
import random

class Agent:
    def __init__(self, name, U, budget):
        self.name = name
        U = np.array(U) 
        # check that U is a 2D square triangular numpy array
        assert isinstance(U, np.ndarray) # check that U is a numpy array
        assert U.ndim == 2 # check that U is 2D
        assert U.shape[0] == U.shape[1] # check that U is square
        assert np.allclose(U, np.triu(U)) # check that U is triangular

        self.U = np.array(U)
        self.budget = budget

    def utility(self, x):
        """Return the utility of a person for a given assignment across all half-days"""
        x = np.array(x)
        utility = np.matmul(np.matmul(x.T, self.U), x) 
        return utility
    
    def find_agent_demand(self, priced_bundles):
        """Return the assignment that maximizes the utility of the person
        
        Args:
            priced_bundles (list): A list of tupbles of the form (bundle, cubicle, price) where bundle is a numpy array of 0s and 1s and price is a float.
            
        Returns:
            tuple: A tuple of the form (bundle, cubicle, price) where bundle is a numpy array of 0s and 1s and price is a float that maximizes the utility of the person."""
        
        # remove all bundles that are too expensive
        feasible_bundles = [priced_bundle for priced_bundle in priced_bundles if priced_bundle[2] <= self.budget]

        # if there are no feasible bundles, return None
        if len(feasible_bundles) == 0:
            return None

        # get the bundle that maximizes the utility of the person
        bundle, cubicle, price = max(feasible_bundles, key=lambda x: self.utility(x[0]))

        return bundle, cubicle, price
        
        


        

if __name__ == "__main__":
    random.seed(123)
    # draw a random utility matrix that is 10 by 10 and triangular
    U = np.triu(np.random.randint(-10, 10, size=(10, 10)))

    Alice = Agent("Alice", U, 100)


    vectors = []

    # Loop over the numbers from 0 to 2^10 - 1
    for num in range(2**10):
        # Get the binary representation of the number as a string
        bin_str = np.binary_repr(num, width=10)
        # Convert the string into a numpy array of integers
        bin_arr = np.array([int(c) for c in bin_str])
        # Append the array to the list of vectors
        vectors.append(bin_arr)

    # Print the list of vectors
    #print(vectors)

    # create numpy array
    vectors = np.array(vectors)

    for i in range(50):
        best_assignment, utility = Alice.find_highest_utility_bundle(vectors)

    print(best_assignment, utility)

