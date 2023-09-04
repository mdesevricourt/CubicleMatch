"""Create a market class to store the information of the market to the matching algorithm to solve their demand"""

import numpy as np
import random
import itertools

from agent import Agent
from cubicle import Cubicle

class Market:
    def __init__(self, agents, cublicles, maxing_price = False) -> None:
        self.agents = agents
        self.cublicles = cublicles
        self.cublicles_names = [cubicle.number for cubicle in cublicles]
        self.numberofagents = len(agents)
        self.numberofcublicles = len(cublicles)
        self.numberofhalfdays = cublicles[0].numberofhalfdays
        self.bundles = [np.array(t) for t in itertools.product([0, 1], repeat=self.numberofhalfdays)]
        self.max_budget = max([agent.budget for agent in agents])
        self.maxing_price = maxing_price
    @property
    def prices_array(self):
        """Returns the prices of the cublicles in a numpy array"""
        return np.array([cubicle.prices for cubicle in self.cublicles])
    
    @prices_array.setter
    def prices_array(self, prices):
        """Set the prices of the cublicles"""
        # check that prices is the right length
        assert len(prices) == self.numberofcublicles
        for cubicle, price in zip(self.cublicles, prices):
            cubicle.prices = price

    @property
    def prices_vec(self):
        """Returns the prices of the cublicles in a numpy array"""
        return np.array([cubicle.prices for cubicle in self.cublicles]).flatten()
    
    @prices_vec.setter
    def prices_vec(self, prices):
        """Set the prices of the cublicles"""
        
        # divide the prices into the cublicles
        prices = np.array(prices).reshape(self.numberofcublicles, self.numberofhalfdays)
        # use the update prices_array method
        self.prices_array = prices

    def print_allocation(self):
        """Print the allocation of the agents to the cublicles"""
        print(f"With prices: {self.prices_vec},")
        print(f"Aggregate demand: {self.aggregate_demand(verbose=True)},")
        print(f"and a clearing error of {self.clearing_error()[0]},")
        print(f"Agents are assigned to cublicles as follows:")
        for agent in self.agents:
            print(f"{agent.name} is assigned to {agent.cublicle} with bundle {agent.current_assignment}, with excess budget {agent.excess_budget}")
        print(f"Prices: {self.prices_vec}")
        

    def price_bundle(self, bundle):
        """Return the lowest price of a bundle of half-days across all cublicles, as well as the name of the cubicle that has the lowest price"""
        # create a dictionary of the prices of all the cublicles
        prices = {}
        for cubicle in self.cublicles:
            prices[cubicle.number] = cubicle.price_bundle(bundle)
        
        # get the number of cublicle that has the lowest price
        
        cubicle = min(prices, key=prices.get)
        # get the price of the cublicle that has the lowest price
        price = prices[cubicle]

        return price, cubicle

    @property
    def priced_bundles(self):
        """Return the prices of all the bundles, along with the cublicle that has the lowest price for that bundle"""
        
        bundles = self.bundles
        priced_bundles = []
        # if maxing_price = False, for each bundle, finds the cubicle that has the lowest price for that bundle, get its name and the price
        # create a list of triplet (bundle, cubicle, price)
        
        if self.maxing_price: 
            for bundle in bundles:
                for cubicle in self.cublicles:
                    price = cubicle.price_bundle(bundle)
                    priced_bundles.append((bundle, cubicle.number, price))
        else:
            for bundle in bundles:
                price, cubicle = self.price_bundle(bundle)
                priced_bundles.append((bundle, cubicle, price))

        # sort the list of triplet by price from lowest to highest
        priced_bundles = sorted(priced_bundles, key=lambda x: x[2])

        return priced_bundles
    
    def aggregate_demand(self, verbose = False):
        """Return the aggregate demand of all the agents, given the current prices of the cublicles"""
        # for each cublicle, create a list representing the demand for that cublicle
        demand_for_cublicles = [0] * self.numberofcublicles * self.numberofhalfdays
        
        maxing_price = self.maxing_price
        # for each agent, find the bundle that maximizes the utility of the agent
        for agent in self.agents:
            bundle, cubicle, _ = agent.find_agent_demand(self.priced_bundles, maxing_price=maxing_price)
            # update the current assignment of the agent
            agent.current_assignment = bundle
            # update the assigned cublicle of the agent
            agent.cublicle = cubicle
            # if the agent is assigned to a bundle
            if bundle is not None:
                # update the demand for that cublicle
                demand_for_cublicles[self.cublicles_names.index(cubicle) * self.numberofhalfdays: self.cublicles_names.index(cubicle) * self.numberofhalfdays + self.numberofhalfdays] += bundle
                agent.excess_budget = agent.budget - self.price_bundle(bundle)[0]

        
        return demand_for_cublicles
    
    def excess_demand(self):
        """Return the excess demand of all the agents, given the current prices of the cublicles"""
        # get the aggregate demand
        demand = self.aggregate_demand()
        # compute the excess demand
        excess_demand = np.array(demand) - 1
        
        return excess_demand
    
    @property
    def excess_budgets(self):
        """Return list of excess budgets for all agents"""
        excess_budgets = []
        for agent in self.agents:
            excess_budgets.append(agent.excess_budget)
        return excess_budgets
    
    def clearing_error(self, verbose = False):
        """Return the clearing error of the market, given the current prices of the cublicles
        
        Returns:
            float: The clearing error of the market
            numpy array: The clearing error for each item in the market
            numpy array: The excess demand for each item in the market
            """
        # get demand 
        excess_demand = self.excess_demand()
        prices = self.prices_vec
        

        # clearing error is excess demand if price is not 0, 0 otherwise
        z = np.array([excess_demand[i] if prices [i] != 0 else np.max([excess_demand[i]], 0) for i in range(len(prices))])
        alpha = np.sum(z**2)


        return alpha, z, excess_demand

    def find_neighbors(self, p):
        # create a list of neighbors
        self.price_vec = p # set the prices of the cublicles to p
        alpha, z, d = self.clearing_error() # compute the clearing error

        neighbors = []
        alpha_list = []
        demand_list = []
        neigbor_type = []
        # loop over the elements of p
        
        # find indexes of elements of z that are not 0
        indexes = [i for i in range(len(z)) if z[i] != 0]
        
        # individual adjustment neighbor
        for i in indexes: 
            # create a copy of p
            p_neighbor = np.array(p.copy())
            
            if z[i] > 0:
                # if there is excess demand for i, increase p[i] until at least one fewer agent demands the item
                d_neighbor_i = d[i]
                while d_neighbor_i >= d[i]:
                    neigbor_type.append("individual adjustment with excess demand")
                    p_neighbor[i] += min(self.excess_budgets) + 0.01
                    # update price
                    self.prices_vec = p_neighbor
                    alpha_neighbor, z_neighbor, d_neighbor = self.clearing_error()
                    d_neighbor_i = d_neighbor[i]
            else:
                # if there is excess supply for i, decrease p[i] until at least one more agent demands the item
                neigbor_type.append("individual adjustment with excess supply")
                p_neighbor[i] = 0
                # update price
                self.prices_vec = p_neighbor
                alpha_neighbor, z_neighbor, d_neighbor = self.clearing_error()
                d_neighbor_i = d_neighbor[i]

                    
    
            # append the neighbor to the list of neighbors
            neighbors.append(p_neighbor)
            alpha_list.append(alpha_neighbor)
            demand_list.append(d_neighbor)

        
        lambda_list = [0.1, 0.5, 1, 2,3,4,5, 6,7,8,9,10]
        for l in lambda_list:
            p_neighbor = p.copy()
            neigbor_type.append(f"gradient adjustment with lambda = {l}")
            p_neighbor = p_neighbor + l * z
            # keep the prices above 0
            p_neighbor = np.maximum(p_neighbor, 0)
            # update price
            self.prices_vec = p_neighbor
            alpha_neighbor, z_neighbor, d_neighbor = self.clearing_error()

            neighbors.append(p_neighbor)
            alpha_list.append(alpha_neighbor)
            demand_list.append(d_neighbor)

        # create tuples of the form (neighbor, alpha, excess demand)
        neighbors = list(zip(neighbors, alpha_list, demand_list, neigbor_type))
        # sort the neighbors by alpha
        neighbors = sorted(neighbors, key=lambda x: x[1])

        return neighbors, (alpha, z, d)
    

    def find_ACE(self, verbose = True):
        """Implements Approximate Competitive Equilibrium from Equal Incomes (ACEEI) algorithm to find an allocation of the cublicles to the agents
        
        Returns: 
            dict: A dictionary of the form {agent: cublicle} that represents the allocation of the cublicles to the agents
            numpy array: A numpy array of the form [price_cublicle_1_half_day_1, price_cublicle_1_half_day_2, ..., price_cublicle_2_half_day_1, ...] that represents the prices of the cublicles"
        """
        print("Running ACE algorithm")
        total_budget = sum([agent.budget for agent in self.agents])
        # initialize the prices of the haldayfs per cublicle to total budget/number of half-days

        besterror = np.inf
        pstar = None
        type_neighbor_selected = {}
        # run the algorithm 100 times
        for i in range(200):
            # max budget times random draw from uniform distribution
            p = self.max_budget * np.random.rand(len(self.prices_vec))
            self.prices_vec = p # set the prices of the cublicles to p
            # find the neighbors of p_0
            tabu_list = []
            c = 0
            searcherror = self.clearing_error()[0]
            if verbose:
                print(f"Running iteration {i}, best error so far is {besterror}, search error to beat is {searcherror}")
            while c < 5:
    
                neighbors, tuple = self.find_neighbors(p)
                alpha, z, d = tuple
                foundnextstep = False

                for neighbor in neighbors:
                    p_tilde = neighbor[0]
                    d = neighbor[2]
                    # check that d is not in the tabu list
                    d_in_tabu = np.any([np.allclose(d, d_tabu) for d_tabu in tabu_list])

                    if not d_in_tabu:
                        foundnextstep = True
                        type_neighbor_selected[neighbor[3]] = type_neighbor_selected.get(neighbor[3], 0) + 1
                        break
                if foundnextstep == False:
                    c = 5
                    
                else:
                    p = p_tilde
                    tabu_list.append(d)
                    currenterror = neighbor[1]
                    if currenterror < searcherror:
                        searcherror = currenterror
                        c = 0
                    else:
                        c += 1
                    if currenterror < besterror:
                        besterror = currenterror
                        pstar = p

            if besterror == 0:
                print(f"Found allocation with error 0 at iteration {i}")
                break

        self.pstar = pstar
        self.prices_vec = pstar
        self.neighbor_type_selected = type_neighbor_selected

    def pricing_out(self, verbose = False):
        """Implements the pricing out algorithm to get rid of excess demand."""

        print("Running pricing out algorithm")
        
        # get the excess demand
        excess_demand = self.excess_demand()
        # while there is excess demand
        i = 0
        while np.any(excess_demand > 0):
            if verbose:
                print(f"Iteration {i}, excess demand: {excess_demand}")
            # get the prices
            prices = self.prices_vec.copy()
            # find item with highest excess demand, in case of tie choose first one
            max_excess_demand = np.max(excess_demand)
            max_excess_demand_index = np.where(excess_demand == max_excess_demand)[0][0]
            # increase the price of the item with highest excess demand by the minimum excess budget plus 0.01
                       
            prices[max_excess_demand_index] += min(self.excess_budgets) + 0.01
            # update the prices
            self.prices_vec = prices
            # get the excess demand
            excess_demand = self.excess_demand()
            i += 1

    def filling_empty_slots(self):
        """This function fills the empty slots in each cubicle with agents that are assigned to have cubicle, in the order of excess budget"""

        for cubicle in self.cublicles:
            pass


def main(find_ACE = True, pricing_out = True, verbose = True, try_price = False, pricing_in = False):
    # generate agents 

    U_Alice = np.zeros((4, 4))
    np.fill_diagonal(U_Alice, [10, 9, 8, 7])
    U_Bob = np.zeros((4, 4))
    np.fill_diagonal(U_Bob, [10, 9, 8, 7])
    U_Charlie = np.zeros((4, 4))
    np.fill_diagonal(U_Charlie, [10, 8, 9, 7])
    U_David = np.zeros((4, 4))
    np.fill_diagonal(U_David, [8, 10, 7, 9])
    agents = [Agent("Alice", U_Alice, 100), Agent("Bob", U_Bob, 101), Agent("Charlie", U_Charlie, 102), Agent("David", U_David, 103)]
    # generate 2 cublicles
    cubicle_1 = Cubicle("C1", [10, 20, 30, 40])
    cubicle_2 = Cubicle("C2", [15, 25, 35, 45])
    cublicles = [cubicle_1, cubicle_2]
    # create a market instance
    market = Market(agents, cublicles, maxing_price=True)
    # print the prices of the cublicles
    print(market.prices_vec)
    # solve for the allocation
    if find_ACE:
        market.find_ACE()
        market.print_allocation()

    if pricing_out:
        market.pricing_out(verbose=True)
        market.print_allocation()

    if pricing_in:
        market.pricing_in(verbose=True)
        market.print_allocation()

    if try_price:
        p = np.array([50.5,50.5,50,50, 51, 51.5, 51, 51.5])
        market.prices_vec = p
        market.aggregate_demand(verbose=True)
        market.print_allocation()




if __name__ == "__main__":
    main(find_ACE=True, pricing_out=False, verbose=True, try_price=True, pricing_in=True)