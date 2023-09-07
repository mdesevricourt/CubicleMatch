""" Alternative version of the algorithm where agents are first assigned to a cubicle and then the algorithm is run again to assign the agents to half-days."""


import numpy as np
import pandas as pd

from cubicle import Cubicle
from agent import Agent
from market import Market
import os
# set random seed

np.random.seed(123)

class SmallMarket():

    def __init__(self, agents, cubicles, maxing_price = False):
        self.agents = agents
        self.cubicles = cubicles
        self.maxing_price = maxing_price
        self._priced_bundles = None
        self._assignments = None
        self._unassigned_agents = None
        self.numberofhalfdays = cubicles[0].numberofhalfdays
    
    def assign_cubicles(self, byyear = False, verbose = False):
        """Assign agents to cubicles: assign agents to the same cubicles if their preferences are the most distanced from each other."""

        # sort agents by budget in ascending order
        agents_by_budget = sorted(self.agents, key = lambda x: x.budget, reverse = False)

        # assign agents to cubicles
        for agent in agents_by_budget:
            # find cubicle with the least number of agents
            if agent.year == "3" and byyear:
                cubicle_list = self.cubicles[0:2]
            else: 
                cubicle_list = self.cubicles
            least_agents = min([len(cubicle.assigned_agents) for cubicle in cubicle_list])
            least_agents_cubicles = [cubicle for cubicle in cubicle_list if len(cubicle.assigned_agents) == least_agents]
            # if there is only one cubicle with the least number of agents, or if all cubicles are empty, assign the agent to the first one in the list
            if len(least_agents_cubicles) == 1 or least_agents == 0:
                # assign agent to that cubicle
                cubicle = least_agents_cubicles[0]
                agent.cubicle = cubicle
                # print(f"Assigning {agent} to {cubicle}")
                cubicle.assigned_agents.append(agent)
                continue
            
            # among the cubicles with the least number of agents, find the cubicle with the largest minimum distance between the agent and the U of all the agents in the cubicle
            min_U = [np.min([np.linalg.norm(agent.U - agent2.U) for agent2 in cubicle.assigned_agents]) for cubicle in least_agents_cubicles]
            cubicle = least_agents_cubicles[np.argmax(min_U)]
            
            agent.cubicle = cubicle
            cubicle.assigned_agents.append(agent)
            # print(f"Assigning {agent} to {cubicle}")

        # highest number of agents in a cubicle
        max_agents = max([len(cubicle.assigned_agents) for cubicle in self.cubicles])
        print(f"Max number of agents in a cubicle: {max_agents}")
        # lowest number of agents in a cubicle
        min_agents = min([len(cubicle.assigned_agents) for cubicle in self.cubicles])
        print(f"Min number of agents in a cubicle: {min_agents}")
        # assert that the difference is at most 1
        assert max_agents - min_agents <= 1
        if verbose: # print agents in each cubicle
            for cubicle in self.cubicles:
                names = [agent.name for agent in cubicle.assigned_agents]
                print(f"{cubicle}: {names}")




    def assign_halfdays(self, N = 100):
        """For each cubicle, assign agents to half-days: assign agents to half-days using the market class."""

        for cubicle in self.cubicles:
            print(f"Assigning agents to half-days for {cubicle}")
            # create a market instance with the agents in the cubicle and the cubicle
            market = Market(cubicle.assigned_agents, [cubicle], maxing_price = self.maxing_price)
            market.find_ACE(N = N, verbose = True)
            market.pricing_out(verbose=True)
            market.filling_empty_slots(verbose=True)
            # make cubicles remember
            cubicle.pstar = market.pstar
            cubicle.CC_found = market.CC_found
            cubicle.price_type = market.price_type
    
    def export_allocation(self, filename):
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
            
            # append CC_found
            allocation_cubicle.append(cubicle.CC_found)
            # append pstar
            allocation_cubicle.append(cubicle.pstar)
            # append price_type
            allocation_cubicle.append(cubicle.price_type)

            # add the list to the dictionary
            allocation[cubicle.number] = allocation_cubicle
        
        # create a pandas dataframe from the dictionary
        df = pd.DataFrame.from_dict(allocation, orient="index")
        # export the dataframe to a csv file in the results folder
        # set the working directory to the root of the project
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        print(f"Exporting allocation to results/{filename}.csv")
        df.to_csv(f"results/{filename}.csv")




