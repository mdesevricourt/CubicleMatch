"""Create an Agent class to store the information of each participant to the matching algorithm to solve their demand"""
import random

import jax as jax
import jax.numpy as jnp
import numpy as np


class Agent:
    def __init__(self, name, U, budget, year=None):
        self.name = name
        U = np.array(U)
        # check that U is a 2D square triangular numpy array
        assert isinstance(U, np.ndarray)  # check that U is a numpy array
        assert U.ndim == 2  # check that U is 2D
        assert U.shape[0] == U.shape[1]  # check that U is square
        assert np.allclose(U, np.triu(U))  # check that U is triangular

        self.U = np.array(U)
        self.budget = budget
        self._current_assignment = None
        self._cubicle = None
        self.excess_budget = 0
        self.year = year

    @property
    def cubicle(self):
        return self._cubicle

    @cubicle.setter
    def cubicle(self, cubicle):
        self._cubicle = cubicle

    @property
    def current_assignment(self):
        return self._current_assignment

    @current_assignment.setter
    def current_assignment(self, assignment):
        self._current_assignment = assignment

    def utility(self, x):
        """Return the utility of a person for a given assignment across all half-days"""
        x = np.array(x)
        utility = np.matmul(np.matmul(x.T, self.U), x)
        return utility

    def find_agent_demand(self, priced_bundles, maxing_price=False):
        """Return the assignment that maximizes the utility of the person

        Args:
            priced_bundles (list): A list of tupbles of the form (bundle, cubicle, price) where bundle is a numpy array of 0s and 1s and price is a float.

        Returns:
            tuple: A tuple of the form (bundle, cubicle, price) where bundle is a numpy array of 0s and 1s and price is a float that maximizes the utility of the person.
        """

        # remove all bundles that are too expensive
        feasible_bundles = [
            priced_bundle
            for priced_bundle in priced_bundles
            if priced_bundle[2] <= self.budget
        ]

        # if there are no feasible bundles, return None
        if len(feasible_bundles) == 0:
            return None

        # compute utility of feasible bundles
        utility = [self.utility(priced_bundle[0]) for priced_bundle in feasible_bundles]

        # get the bundles that maximize the utility of the person
        max_utility = max(utility)
        maximizers = [
            feasible_bundles[i]
            for i in range(len(feasible_bundles))
            if utility[i] == max_utility
        ]

        # if there is only one bundle that maximizes the utility of the person, return it
        if len(maximizers) == 1:
            return maximizers[0]

        # if there are multiple bundles that maximize the utility of the person, return the one with the lowest price
        if maxing_price:
            max_price = max([priced_bundle[2] for priced_bundle in maximizers])
            demand = [
                priced_bundle
                for priced_bundle in maximizers
                if priced_bundle[2] == max_price
            ][0]
            return demand
        else:
            min_price = min([priced_bundle[2] for priced_bundle in maximizers])
            demand = [
                priced_bundle
                for priced_bundle in maximizers
                if priced_bundle[2] == min_price
            ][0]
            return demand

        return demand

    def find_best_extra_halfday(self, empty_slots):
        """This method finds the best extra half-day for the agent to take, given a list of empty slots.

        Args:
            empty_slots (list): A list of 0 and 1 indicating which slots are empty, where 1 means empty and 0 means full.

        Returns:
            int: The index of the best extra half-day for the agent to take.
        """

        # compute the utility of the agent for each extra half-day based on their current assignment
        utility = []
        for i, slot in enumerate(empty_slots):
            if slot == 1:
                # slot is empty so compute the utility of the agent for this extra half-day
                # find the alternative bundle
                alternative_assignment = self.current_assignment.copy()
                alternative_assignment[i] = 1
                # compute the utility of the agent for the alternative bundle
                utility.append(self.utility(alternative_assignment))

            if slot == 0:
                # slot is taken so utility is 0
                utility.append(0)

        # find the extra half-day that maximizes the utility of the agent
        max_utility = max(utility)
        best_extra_halfday = [
            i for i in range(len(utility)) if utility[i] == max_utility
        ][0]
        extra_utility = max_utility - self.utility(self.current_assignment)

        return best_extra_halfday, extra_utility


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
    # print(vectors)

    # create numpy array
    vectors = np.array(vectors)

    for i in range(50):
        best_assignment, utility = Alice.find_highest_utility_bundle(vectors)

    print(best_assignment, utility)
