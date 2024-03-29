import unittest

import numpy as np
from agent import Agent
from alter import SmallMarket
from cubicle import Cubicle
from market import Market


class TestAgent(unittest.TestCase):
    def setUp(self) -> None:
        U = np.zeros((2, 2))
        np.fill_diagonal(U, [2, 1])
        self.agent_Alice = Agent("Alice", U, 3)
        return

    def test_utility(self):
        # utility of Alice for [0, 0] should be 0
        self.assertEqual(self.agent_Alice.utility([0, 0]), 0)
        # utility of Alice for [1, 0] should be 2
        self.assertEqual(self.agent_Alice.utility([1, 0]), 2)
        # utility of Alice for [0, 1] should be 1
        self.assertEqual(self.agent_Alice.utility([0, 1]), 1)
        # utility of Alice for [1, 1] should be 3
        self.assertEqual(self.agent_Alice.utility([1, 1]), 3)
        return

    def test_find_agent_demand(self):
        # create a list of priced bundles
        priced_bundles = [
            ([0, 0], "1", 1),
            ([1, 0], "1", 2),
            ([0, 1], "2", 2),
            ([1, 1], "1", 3),
            ([1, 1], "2", 3),
        ]
        bundle, cublicle, price = self.agent_Alice.find_agent_demand(priced_bundles)
        # bundle should be [1, 1]
        self.assertEqual(bundle, [1, 1])
        # cublicle should be "1"
        self.assertEqual(cublicle, "1")
        # other priced_bundles
        priced_bundles = [
            ([0, 0], "1", 1),
            ([1, 0], "1", 2),
            ([0, 1], "2", 2),
            ([1, 1], "1", 5),
            ([1, 1], "2", 4),
        ]
        bundle, cublicle, price = self.agent_Alice.find_agent_demand(priced_bundles)
        # bundle should be [1,0]
        self.assertEqual(bundle, [1, 0])
        # cublicle should be "1"
        self.assertEqual(cublicle, "1")

        return

    def test_best_extra_half_day(self):
        """Test that the best extra half-day is the one that maximizes the utility of the agent."""

        # create a list of empty slots
        empty_slots = [1, 1]
        self.agent_Alice.current_assignment = [0, 0]
        # call the best_extra_half_day method
        best_extra_halfday, extra_utility = self.agent_Alice.find_best_extra_halfday(
            empty_slots
        )
        # check that the bundle is correct

        expected_index = 0
        actual_index = best_extra_halfday
        # use assertEqual to compare lists
        self.assertEqual(expected_index, actual_index)

        # check that the utility is correct
        expected_utility = 2
        actual_utility = extra_utility
        self.assertEqual(expected_utility, actual_utility)

        empty_slots = [0, 1]
        best_extra_halfday, extra_utility = self.agent_Alice.find_best_extra_halfday(
            empty_slots
        )
        # check that the bundle is correct

        expected_index = 1
        actual_index = best_extra_halfday
        # use assertEqual to compare lists
        self.assertEqual(expected_index, actual_index)

        # check that the utility is correct
        expected_utility = 1
        actual_utility = extra_utility
        self.assertEqual(expected_utility, actual_utility)

        return


class TestMarket(unittest.TestCase):
    # create a setUp method to initialize a Market instance with some sample data
    def setUp(self):
        # create some sample agents
        U_Alice = np.zeros((4, 4))
        np.fill_diagonal(U_Alice, [1, 0, 1, 0])
        U_Bob = np.zeros((4, 4))
        np.fill_diagonal(U_Bob, [0, 1, 0, 1])
        agents = [Agent("Alice", U_Alice, 100), Agent("Bob", U_Bob, 100)]
        # create some sample cubicles with different prices
        cubicles = [
            Cubicle("C1", prices=[10, 20, 30, 40]),
            Cubicle("C2", prices=[15, 25, 35, 45]),
        ]
        # create a Market instance with the agents and cubicles
        self.market = Market(agents, cubicles)

    # create a test method for each method of the Market class
    def test_prices_array(self):
        # test that the prices_array property returns a numpy array of the prices of the cubicles
        expected = np.array([[10, 20, 30, 40], [15, 25, 35, 45]])
        actual = self.market.prices_array
        # use assertArrayEqual to compare numpy arrays
        np.testing.assert_array_equal(actual, expected)

    def test_prices_array(self):
        # test that the update_prices_array method sets the prices of the cubicles correctly
        # create a new array of prices to update
        new_prices = np.array([[11, 21, 31, 41], [16, 26, 36, 46]])
        # call the update_prices_array method with the new prices
        self.market.prices_array = new_prices
        # check that the prices_array property reflects the changes
        expected = new_prices
        actual = self.market.prices_array
        # use assertArrayEqual to compare numpy arrays
        np.testing.assert_array_equal(actual, expected)

    def test_prices_vec(self):
        # test that the prices_vec property returns a numpy array of the prices of the cubicles flattened
        expected = np.array([10, 20, 30, 40, 15, 25, 35, 45])
        actual = self.market.prices_vec
        # use assertArrayEqual to compare numpy arrays
        np.testing.assert_array_equal(actual, expected)

    def test_prices_vec(self):
        # test that the update_prices_vec method sets the prices of the cubicles correctly
        # create a new array of prices to update (flattened)
        new_prices = np.array([11, 21, 31, 41, 16, 26, 36, 46])
        # call the update_prices_vec method with the new prices
        self.market.prices_vec = new_prices
        # check that the prices_vec property reflects the changes
        expected = new_prices
        actual = self.market.prices_vec
        # use assertArrayEqual to compare numpy arrays
        np.testing.assert_array_equal(actual, expected)

    def test_price_bundle(self):
        # test that the price_bundle method returns the lowest price of a bundle of half-days across all cubicles,
        # as well as the name of the cubicle that has the lowest price
        # create a sample bundle of half-days (a numpy array of zeros and ones)
        bundle = np.array([1, 0, 1, 0])
        # call the price_bundle method with the bundle
        price, cubicle = self.market.price_bundle(bundle)
        # check that the price is correct (the sum of the first and third elements of the lowest-priced cubicle)
        expected_price = 10 + 30  # C1 has the lowest price for this bundle
        actual_price = price
        # use assertEqual to compare scalars
        self.assertEqual(actual_price, expected_price)
        # check that the cubicle is correct
        expected_cubicle = "C1"
        actual_cubicle = cubicle
        # use assertEqual to compare strings
        self.assertEqual(actual_cubicle, expected_cubicle)

    def test_aggregate_demand(self):
        expected_aggregate_demand = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        actual_aggregate_demand = self.market.aggregate_demand()
        self.assertEqual(actual_aggregate_demand, expected_aggregate_demand.tolist())

    def test_excess_demand(self):
        expected_excess_demand = np.array([0, 0, 0, 0, -1, -1, -1, -1])
        actual_excess_demand = self.market.excess_demand()
        self.assertEqual(actual_excess_demand.tolist(), expected_excess_demand.tolist())

    def test_find_neighbors(self):
        expected_neighbor1 = np.array([10, 20, 30, 40, 15, 25, 35, 0])
        expected_neighbor2 = np.array([10, 20, 30, 40, 15, 25, 0, 45])
        actual_neighbor_list, _ = self.market.find_neighbors(
            [10, 20, 30, 40, 15, 25, 35, 45]
        )

        actual_neighbor_list = [neighbor[0] for neighbor in actual_neighbor_list]

        self.assertTrue(
            any(
                np.array_equal(expected_neighbor1, actual_neighbor)
                for actual_neighbor in actual_neighbor_list
            )
        )
        self.assertTrue(
            any(
                np.array_equal(expected_neighbor2, actual_neighbor)
                for actual_neighbor in actual_neighbor_list
            )
        )

    def test_filling_empty_slots(self):
        # create two agents Alice and Bob
        U_Alice = np.zeros((4, 4))
        np.fill_diagonal(U_Alice, [4, 3, 2, 1])
        U_Bob = np.zeros((4, 4))
        np.fill_diagonal(U_Bob, [4, 3, 2, 1])
        agents = [Agent("Alice", U_Alice, 100), Agent("Bob", U_Bob, 101)]
        # one cubicle with different prices
        agents[0].cubicle = "C1"
        agents[0].current_assignment = np.array([0, 0, 0, 0])
        agents[1].cubicle = "C1"
        agents[1].current_assignment = np.array([0, 0, 0, 0])
        cubicles = [Cubicle("C1", prices=[10, 20, 30, 40])]
        # create a Market instance with the agents and cubicles
        market = Market(agents, cubicles)
        market.filling_empty_slots(verbose=True)
        # check that the allocation is correct
        expected_bundle_Alice = np.array([1, 0, 1, 0])
        expected_bundle_Bob = np.array([0, 1, 0, 1])

        actual_bundle_Alice = market.agents[0].current_assignment
        print(f"actual_bundle_Alice: {actual_bundle_Alice}")
        actual_bundle_Bob = market.agents[1].current_assignment
        self.assertTrue(np.array_equal(expected_bundle_Alice, actual_bundle_Alice))
        self.assertTrue(np.array_equal(expected_bundle_Bob, actual_bundle_Bob))


class TestSmallMarket(unittest.TestCase):
    def setUp(self):
        # create some 3 types of agents
        U_Alice = np.zeros((4, 4))
        np.fill_diagonal(U_Alice, [1, 0, 1, 0])
        U_Bob = np.zeros((4, 4))
        np.fill_diagonal(U_Bob, [0, 1, 0, 1])
        U_Carol = np.zeros((4, 4))
        np.fill_diagonal(U_Carol, [1, 1, 1, 1])
        # create 4 agents of type Alice
        agents = [Agent("Alice ", U_Alice, 100) for i in range(4)]
        # create 4 agents of type Bob
        agents += [Agent("Bob", U_Bob, 101) for i in range(4)]
        # create 4 agents of type Carol
        agents += [Agent("Carol", U_Carol, 102) for i in range(4)]
        # create 4 cubicles with different prices
        cubicles = [
            Cubicle("C1", prices=[10, 20, 30, 40]),
            Cubicle("C2", prices=[15, 25, 35, 45]),
            Cubicle("C3", prices=[20, 30, 40, 50]),
            Cubicle("C4", prices=[25, 35, 45, 55]),
        ]
        # create a SmallMarket instance with the agents and cubicles
        self.market = SmallMarket(agents, cubicles)

    def test_assign_cubicles(self):
        # test that the assign_cubicles method assigns agents to cubicles
        # call the assign_cubicles method
        self.market.assign_cubicles()
        # check that each cubicle has 3 agents assigned to it
        expected = [3, 3, 3, 3]
        actual = [len(cubicle.assigned_agents) for cubicle in self.market.cubicles]
        self.assertEqual(actual, expected)
        # check that each agent in the same cubicle has a different name
        for cubicle in self.market.cubicles:
            names = [agent.name for agent in cubicle.assigned_agents]
            # check that the number of names is equal to the number of unique names
            self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()


# class TestMarket(unittest.TestCase):
