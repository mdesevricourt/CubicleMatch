"""Create a market class to store the information of the market to the matching algorithm to solve their demand"""

import numpy as np
import random
import itertools
import pandas as pd
import os
import time

from agent import Agent

from cubicle import Cubicle

random.seed(123)
np.random.seed(123)
class Market:
    def __init__(self, agents, cubicles, maxing_price = False) -> None:
        self.agents = agents
        self.cubicles = cubicles
        self.cubicles_names = [cubicle.number for cubicle in cubicles]
        self.numberofagents = len(agents)
        self.numberofcubicles = len(cubicles)
        self.numberofhalfdays = cubicles[0].numberofhalfdays
        self.bundles = [np.array(t) for t in itertools.product([0, 1], repeat=self.numberofhalfdays)]
        self.max_budget = max([agent.budget for agent in agents])
        self.maxing_price = maxing_price
        self.CC_found = False
        self.best_error = np.inf
        self.pstar_type = None
        self.pstar = None

    def guess_prices(self):
        """Returns a guess for the prices of the cubicles close to the average price of the cubicles if everybody exhausted their budget"""
        total_budget = sum([agent.budget for agent in self.agents])
        # beta_list = [0.1, 0.5, 1, 2,3,4,5, 6,7,8,9,10]
        # randomly choose a beta
        beta = 1
        # print(f"beta is {beta}")
        
        self.prices_vec = total_budget / (self.numberofhalfdays * self.numberofcubicles) + beta*np.random.uniform(-1, 1, len(self.prices_vec))
    
    @property
    def prices_array(self):
        """Returns the prices of the cubicles in a numpy array"""
        return np.array([cubicle.prices for cubicle in self.cubicles])
    
    @prices_array.setter
    def prices_array(self, prices):
        """Set the prices of the cubicles"""
        # check that prices is the right length
        assert len(prices) == self.numberofcubicles
        for cubicle, price in zip(self.cubicles, prices):
            cubicle.prices = price

    @property
    def prices_vec(self):
        """Returns the prices of the cubicles in a numpy array"""
        return np.array([cubicle.prices for cubicle in self.cubicles]).flatten()
    
    @prices_vec.setter
    def prices_vec(self, prices):
        """Set the prices of the cubicles"""
        
        # divide the prices into the cubicles
        prices = np.array(prices).reshape(self.numberofcubicles, self.numberofhalfdays)
        # use the update prices_array method
        self.prices_array = prices
    def print_ACE(self):
        """Print the ACE found - careful, this will change the allocation of the agents to the cubicles"""
        print(f"With prices: {self.prices_vec},")
        print(f"Aggregate demand: {self.aggregate_demand(verbose=True)},")
        print(f"and a clearing error of {self.clearing_error()[0]},")
        self.print_allocation()

    def print_allocation(self):
        """Print the allocation of the agents to the cubicles"""

        print(f"Agents are assigned to cubicles as follows:")
        for agent in self.agents:
            print(f"{agent.name} is assigned to {agent.cubicle} with bundle {agent.current_assignment}, with excess budget {agent.excess_budget}")
        

    def price_bundle(self, bundle):
        """Return the lowest price of a bundle of half-days across all cubicles, as well as the name of the cubicle that has the lowest price"""
        # create a dictionary of the prices of all the cubicles
        prices = {}
        for cubicle in self.cubicles:
            prices[cubicle.number] = cubicle.price_bundle(bundle)
        
        # get the number of cubicle that has the lowest price
        
        cubicle = min(prices, key=prices.get)
        # get the price of the cubicle that has the lowest price
        price = prices[cubicle]

        return price, cubicle

    @property
    def priced_bundles(self):
        """Return the prices of all the bundles, along with the cubicle that has the lowest price for that bundle"""
        
        bundles = self.bundles
        priced_bundles = []
        # if maxing_price = False, for each bundle, finds the cubicle that has the lowest price for that bundle, get its name and the price
        # create a list of triplet (bundle, cubicle, price)
        
        if self.maxing_price: 
            for bundle in bundles:
                for cubicle in self.cubicles:
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
        """Return the aggregate demand of all the agents, given the current prices of the cubicles"""
        # for each cubicle, create a list representing the demand for that cubicle
        demand_for_cubicles = [0] * self.numberofcubicles * self.numberofhalfdays
        
        maxing_price = self.maxing_price
        # for each agent, find the bundle that maximizes the utility of the agent
        for agent in self.agents:
            bundle, cubicle, _ = agent.find_agent_demand(self.priced_bundles, maxing_price=maxing_price)
            # update the current assignment of the agent
            agent.current_assignment = bundle
            # update the assigned cubicle of the agent
            agent.cubicle = cubicle
            # if the agent is assigned to a bundle
            if bundle is not None:
                # update the demand for that cubicle
                demand_for_cubicles[self.cubicles_names.index(cubicle) * self.numberofhalfdays: self.cubicles_names.index(cubicle) * self.numberofhalfdays + self.numberofhalfdays] += bundle
                agent.excess_budget = agent.budget - self.price_bundle(bundle)[0]

        
        return demand_for_cubicles
    
    def excess_demand(self):
        """Return the excess demand of all the agents, given the current prices of the cubicles"""
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
        """Return the clearing error of the market, given the current prices of the cubicles
        
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
        self.price_vec = p # set the prices of the cubicles to p
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
                    neigbor_type.append("individual adjustment neighbor with excess demand")
                    p_neighbor[i] += min(self.excess_budgets) + 0.01
                    # update price
                    self.prices_vec = p_neighbor
                    alpha_neighbor, z_neighbor, d_neighbor = self.clearing_error()
                    d_neighbor_i = d_neighbor[i]
            else:
                # if there is excess supply for i, decrease p[i] until at least one more agent demands the item
                neigbor_type.append("individual adjustment neighbor with excess supply")
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
            neigbor_type.append(f"gradient adjustment neighbor with lambda = {l}")
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
    


    def find_ACE(self, N = 1000, time_allowed = 0, verbose = True):
        """Implements Approximate Competitive Equilibrium from Equal Incomes (ACEEI) algorithm to find an allocation of the cubicles to the agents
        
        Returns: 
            dict: A dictionary of the form {agent: cubicle} that represents the allocation of the cubicles to the agents
            numpy array: A numpy array of the form [price_cubicle_1_half_day_1, price_cubicle_1_half_day_2, ..., price_cubicle_2_half_day_1, ...] that represents the prices of the cubicles"
        """
        print("Running ACE algorithm")
        total_budget = sum([agent.budget for agent in self.agents])
        # initialize the prices of the haldayfs per cubicle to total budget/number of half-days

        besterror = np.inf
        best_number_excess_demand = np.inf
        pstar = None
        type_neighbor_selected = {}
        # run the algorithm 100 times
        i = 0
        # start timer
        start = time.time()
        time_elapsed = 0
        number_selected_uniform_price = 0

        while i < N or time_elapsed < time_allowed:
            i += 1
            # with probability 0.5, initialize the prices of the half-days per cubicle to max budget * random number between 0 and 1
            if i == 1: # for first iteration, check current guess
                p = self.prices_vec 
                p_type = "initial guess"  

            elif np.random.rand() < 0.5 :
                p = self.max_budget * np.random.rand(len(self.prices_vec))
                p_type = "random"
            else:
                # error drawn from uniform distribution between -1 and 1
                beta =  number_selected_uniform_price // 19 + 1
                number_selected_uniform_price += 1
                p = total_budget / (self.numberofhalfdays * self.numberofcubicles) + beta*np.random.uniform(-1, 1, len(self.prices_vec))
                p_type = f"uniform with beta = {beta}"
            # p = total budget / (number of half-days * number of cubicles) + random error
            
            self.prices_vec = p # set the prices of the cubicles to p
            # find the neighbors of p_0
            tabu_list = []
            c = 0
            searcherror = self.clearing_error()[0]
            search_number_excess_demand = np.sum(self.excess_demand() > 0)
            # check if search error is better than best error
            if searcherror < besterror or (searcherror == besterror and search_number_excess_demand < best_number_excess_demand):
                besterror = searcherror
                best_number_excess_demand = search_number_excess_demand
                pstar = p
                pstar_type = p_type
                

            if verbose:
                print1 = f"Running iteration {i} \n"
                print2 = f"\tBest error: {besterror}, \n\tBest number of excessively demanded half_days: {best_number_excess_demand}\n"
                print3 = f"\tSearch error: {searcherror}, \n\tnumber of excessively demanded half_days to beat is {search_number_excess_demand}\n"

                # Print with color and indentation
                print('\033[34m' + print1 + '\033[0m', end='')
                print(print2, end='')
                print(print3, end='')

            while c < 5:
                if verbose:
                    print(f"\t\tc= {c}")
                neighbors, tuple = self.find_neighbors(p)
                if verbose:
                    print(f"\t\t\tFound {len(neighbors)} neighbors")
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
                        p_neighbor_type = neighbor[3] + " of " + p_type
                        break

                
                if foundnextstep == False:
                    if verbose:
                        print(f"\t\t\tNo neighbor found that is not in the tabu list")
                    c = 5 
                else:
                    self.price_vec = p_tilde
                    tabu_list.append(d)
                    currenterror = neighbor[1]
                    if currenterror < searcherror:
                        searcherror = currenterror
                        c = 0
                        if verbose:
                            print(f"\t\t\tFound a neighbor that is not in the tabu list, with smaller error: {currenterror}")
                            print(f"\t\t\tSetting c to 0")
                    else:
                        if verbose:
                            print(f"\t\t\tFound a neighbor that is not in the tabu list, with larger error: {currenterror}")
                        c += 1
                    if currenterror < besterror or (currenterror == besterror and np.sum(z > 0) < best_number_excess_demand):
                        besterror = currenterror
                        pstar = self.price_vec.copy()
                        pstar_type = p_neighbor_type
                        if currenterror == 0:
                            break

            time_elapsed = time.time() - start
            if verbose:
                print(f"\tTime elapsed: {time_elapsed}")
            if besterror == 0:
                print(f"Found allocation with error 0 at iteration {i}")
                break

        self.pstar = pstar
        self.prices_vec = pstar
        self.price_type = pstar_type
        self.excess_demand() # make sure ACE is computed
        alpha, z, d = self.clearing_error()
        if alpha == 0:
            self.CC_found = True
        self.neighbor_type_selected = type_neighbor_selected

    # add argument to specify how often print statements should be printed
    def pricing_out(self, verbose = False, print_every = 100):
        """Implements the pricing out algorithm to get rid of excess demand."""
        print("Running pricing out algorithm")
        
        # get the excess demand
        excess_demand = self.excess_demand()
        # while there is excess demand
        i = 0
        while np.any(excess_demand > 0):
            if verbose and i % print_every == 0:
                print(f"Iteration {i}, total excess demand: {excess_demand}")
                print(f"Current prices: {self.prices_vec}")
            # get the prices
            prices = self.prices_vec.copy()
            # find item with highest excess demand, in case of tie choose first one
            max_excess_demand = np.max(excess_demand)
            max_excess_demand_index = np.where(excess_demand == max_excess_demand)[0][0]
            # increase the price of the item with highest excess demand by the minimum excess budget plus 0.01
            # find the corresponding cubicle
            cubicle = self.cubicles[max_excess_demand_index // self.numberofhalfdays]
            # item demanded
            item_demanded = max_excess_demand_index % self.numberofhalfdays
            # among agents in that cubicle, find lowest excess budget
            agents_in_cubicle = [agent for agent in self.agents if agent.cubicle == cubicle.number and agent.current_assignment[item_demanded] == 1]
            min_excess_budget = min([agent.excess_budget for agent in agents_in_cubicle])
                       
            prices[max_excess_demand_index] += max([min_excess_budget, 0.01]) + 0.01
            # update the prices
            self.prices_vec = prices
            # get the excess demand
            excess_demand = self.excess_demand()
            i += 1
    
    def pricing_in(self, verbose = False):
        """Implements a pricing in algorithm to get rid of excess supply."""
        print("Running pricing in algorithm")
        # get the excess demand
        excess_demand = self.excess_demand()
        i = 0
        # get clearing error 
        alpha_0, z, d = self.clearing_error()
        prices_0 = self.prices_vec.copy()
        # if there is excess demand, return
        if np.any(excess_demand > 0):
            print("There is excess demand, cannot run pricing in algorithm")
            return
        # if there is excess supply, decrease the price of the item with the highest excess supply so it becomes just affordable for the agent with the higest excess budget
        
        while np.any(excess_demand < 0):
            if verbose:
                print(f"Iteration {i}, excess demand: {excess_demand}")
            # get the prices
            prices = self.prices_vec.copy()
            # find item with highest excess demand, in case of tie choose first one
            min_excess_demand = np.min(excess_demand)
            min_excess_demand_index = np.where(excess_demand == min_excess_demand)[0][0]
            # find the corresponding cubicle
            cubicle = self.cubicles[min_excess_demand_index // self.numberofhalfdays]
            # among agents in that cubicle, find higest excess budget
            agents_in_cubicle = [agent for agent in self.agents if agent.cubicle == cubicle.number]
            max_excess_budget = max([agent.excess_budget for agent in agents_in_cubicle])
            # decrease the price of the item with highest excess demand by exactly the highest excess budget
            prices[min_excess_demand_index] -= max_excess_budget
            # update the prices
            self.prices_vec = prices
            # if new clearing error is higher than old clearing error, revert to old prices
            alpha, z, d = self.clearing_error()
            if alpha > alpha_0:
                if verbose:
                    print(f"New clearing error {alpha} is higher than old clearing error {alpha_0}, reverting to old prices")
                self.prices_vec = prices_0
                break
            else:
                if verbose:
                    print(f"New clearing error {alpha} is lower than old clearing error {alpha_0}, keeping new prices")
                alpha_0 = alpha
                prices_0 = prices
                if alpha == 0:
                    break
            i += 1

    def filling_empty_slots(self, verbose = False):
        """This function fills the empty slots in each cubicle with agents that are assigned to have cubicle, in the order of excess budget"""
        print("Running filling empty slots algorithm")
        for cubicle in self.cubicles:
            
            # get the agents that are assigned to the cubicle
            agents_in_cubicle = [agent.name for agent in self.agents if agent.cubicle == cubicle.number]
            # get agent's budget
            budgets = [agent.budget for agent in self.agents if agent.cubicle == cubicle.number]
            # get number of halfdays assigned
            n_halfdays = [np.sum(agent.current_assignment) for agent in self.agents if agent.cubicle == cubicle.number]
            agent_tuples = list(zip(agents_in_cubicle, budgets, n_halfdays))
            # sort by ascending n_halfdays, and then by ascending budget
            agent_tuples = sorted(agent_tuples, key=lambda x: (x[2], x[1]))
            # get the agents in the right order
            agents_in_cubicle = [agent_tuple[0] for agent_tuple in agent_tuples]

            number_of_agents_in_cubicle = len(agents_in_cubicle)
            # demand for the cubicle - add the demand of the agents that are assigned to the cubicle
            demand = np.array([0] * self.numberofhalfdays)

            for agent in agents_in_cubicle:
                demand += self.agents[[i for i in range(len(self.agents)) if self.agents[i].name == agent][0]].current_assignment
            
            # if there is an empty slot in the cubicle
            # get the number of empty slots
            extra_utility_agents = 1
            while np.any(demand == 0) and extra_utility_agents > 0:
                if verbose:
                    print(f"Agents in cubicle {cubicle.number} are {agents_in_cubicle}")
                    print(f"Demand for cubicle {cubicle.number} is {demand}")
                
                extra_utility_agents = 0
                for agent_name in agents_in_cubicle:
                    empty_slots = [1 if demand[i] == 0 else 0 for i in range(len(demand))]
                    # get position of agent in list of agents
                    agent_index = [i for i in range(len(self.agents)) if self.agents[i].name == agent_name][0]
                    best_extra_halfday, extra_utility = self.agents[agent_index].find_best_extra_halfday(empty_slots)
                    
                    extra_utility_agents += extra_utility
                    if extra_utility > 0:
                        if verbose:
                            print(f"Agent {self.agents[agent_index].name} takes half-day {best_extra_halfday}")
                        # update the demand
                        demand[best_extra_halfday] = 1
                        # update the current assignment of the agent
                        assignment = self.agents[agent_index].current_assignment.copy()
                        assignment[best_extra_halfday] = 1
                        self.agents[agent_index].current_assignment = assignment
                        if verbose:
                            print(f"Agent {self.agents[agent_index].name} is now assigned to {self.agents[agent_index].current_assignment}")
                        
                    # if there is no more empty slot, break
                    if np.all(demand == 1):
                        if verbose:
                            print(f"No more empty slots for cubicle {cubicle.number}")
                        break
                    elif extra_utility_agents == 0:
                        if verbose:
                            print(f"Agents do not care about empty slots in cubicle {cubicle.number} anymore.")
                        break

    def export_allocation(self, filename):
        """Export the allocation of the agents to a pandas dataframe. Each row is a cubicle, each column a half-day and each cell contains the name of the agent assigned to that half-day in that cubicle.
        
        Args:
            filename (str): The name of the file to export the allocation to
        """

        # create a dictionary of the form {cubicle: [half-day 1, half-day 2, ...]}
        allocation = {}
        for cubicle in self.cubicles:
            # get the agents assigned to the cubicle
            agents_in_cubicle = [agent for agent in self.agents if agent.cubicle == cubicle.number]
            # get the half-days assigned to the agents
            halfdays = [agent.current_assignment for agent in agents_in_cubicle]
            # get the names of the agents
            names = [agent.name for agent in agents_in_cubicle]
            allocation_cubicle = []
            for halfday_index in range(self.numberofhalfdays):
                # get the names of the agents assigned to the half-day as a string
                names_halfday = ", ".join([names[i] for i in range(len(names)) if halfdays[i][halfday_index] == 1])
                allocation_cubicle.append(names_halfday)

            # add the list to the dictionary
            allocation[cubicle.number] = allocation_cubicle
        
        # create a pandas dataframe from the dictionary
        df = pd.DataFrame.from_dict(allocation, orient="index")
        # export the dataframe to a csv file in the results folder
        # set the working directory to the root of the project
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        print(f"Exporting allocation to results/{filename}.csv")
        df.to_csv(f"results/{filename}.csv")


def main(find_ACE = False, pricing_out = False, verbose = True, try_price = False, pricing_in = False, filling_empty_slots = False):
    # generate agents 
    # set random seed
    

    U_Alice = np.zeros((4, 4))
    np.fill_diagonal(U_Alice, [10, 5, 5, 4])
    U_Bob = np.zeros((4, 4))
    np.fill_diagonal(U_Bob, [10, 9, 8, 7])
    U_Charlie = np.zeros((4, 4))
    np.fill_diagonal(U_Charlie, [10, 8, 9, 7])
    U_David = np.zeros((4, 4))
    np.fill_diagonal(U_David, [8, 10, 7, 9])
    agents = [Agent("Alice", U_Alice, 100), Agent("Bob", U_Bob, 101), Agent("Charlie", U_Charlie, 102), Agent("David", U_David, 103)]
    # generate 2 cubicles
    cubicle_1 = Cubicle("C1", [10, 20, 30, 40])
    cubicle_2 = Cubicle("C2", [15, 25, 35, 45])
    cubicles = [cubicle_1, cubicle_2]
    # create a market instance
    market = Market(agents, cubicles, maxing_price=True)
    # print the prices of the cubicles
    print(market.prices_vec)
    # solve for the allocation
    if find_ACE:
        market.find_ACE(N = 10)
        market.print_ACE()

    if pricing_out:
        market.pricing_out(verbose=True)
        market.print_ACE()

    if pricing_in:
        market.pricing_in(verbose=True)
        market.print_ACE()
    
    if filling_empty_slots:
        market.filling_empty_slots(verbose = True)
        market.print_allocation()

    if try_price:
        p = np.array([50.5,50.5,50,50, 51, 51.5, 51, 51.5])
        market.prices_vec = p
        market.aggregate_demand(verbose=True)
        market.print_allocation()


    market.export_allocation("test")




if __name__ == "__main__":
    main(find_ACE=True,  verbose=True, try_price=False, filling_empty_slots=True)