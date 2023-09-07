
import numpy as np
import pandas as pd
import os
import time
import datetime

import cubicle
import agent
import market

# set random seed

np.random.seed(123)

def cubicle_builder(number_of_cubicles, number_of_half_days):
    """This function creates a list of cubicles, where each cubicle is a Cubicle object.
    
    Args:
        number_of_cubicles (int): The number of cubicles.
        number_of_half_days (int): The number of half-days.
    
    Returns:
        dict: A dictionary of cubicles.
    """

    # create a list of cubicles
    cubicles = [cubicle.Cubicle("C" + str(i), number_of_half_days) for i in range(1, number_of_cubicles+1)]


    return cubicles

def agent_builder(filename, verbose = False):
    """This function creates a list of agents, where each agent is an Agent object based on csv file specified by filename.
    
    Args:
        filename (str): The name of the csv file.
        
    Returns:
        list: A list of agents.
    """
    
    # read the csv file from the data folder
    # change current working directory to the data folder, print current working directory
    print(os.getcwd())
    os.chdir("data")
    # skip the first row of the csv file

    df = pd.read_csv(filename, skiprows=1, index_col=0)
    # print 
    print(df.index)
    print(df.columns)
    # for each agent in the dataframe
    agents = []
    for name in df.index:
        # get list of values for the agent for the halfdays Mon AM to Fri PM
        halfday_names = [f"Mon AM", f"Mon PM", f"Tues AM", f"Tues PM", f"Weds AM", f"Weds PM", f"Thurs AM", f"Thurs PM", f"Fri AM", f"Fri PM"]
        
        preferences = df.loc[name, halfday_names].values
        U = np.zeros((len(preferences), len(preferences)))
        np.fill_diagonal(U, preferences)
        # add 0.5 if two halfdays of the same day have been selected (i.e. 1 and 2, 3 and 4, 5 and 6, 7 and 8, 9 and 10)
        for i in range(0, len(preferences), 2):
            if U[i, i] > 0 and U[i+1, i+1] > 0:
                U[i, i+1] = 0.5
        year = str(df.loc[name, 'Year'])
        # budget is 100 + year - 3 + error term from uniform distribution on [0, 1]
        budget = 100 + float(year) - 3  + np.random.uniform(0, 1)
        # create an agent
        if verbose:
            print(f"Creating agent {name} with preferences {U} and budget {budget}")
        agents.append(agent.Agent(name, U, budget, year=year))
    return agents

    
cubicles = cubicle_builder(6, 10)
agents = agent_builder("Cubicles_results.csv", verbose = True)
market = Market(agents, cubicles, maxing_price = True)
#market.find_ACE(N = 2, verbose = True)
backup = False
long_run = False
alternative_algorithm = True
# time allowed is until tomorrow 8am
current_time = datetime.datetime.now()
tomorrow = current_time + datetime.timedelta(days=1)
new_time = datetime.datetime(year=tomorrow.year, month=tomorrow.month, day=tomorrow.day, hour=8, minute=0, second=0)
# Subtract the current time from the new time to get the difference in seconds
time_allowed = new_time - current_time
print(f"Time allowed is {time_allowed} seconds")


if alternative_algorithm:
    # create a small market

    small_market = SmallMarket(agents, cubicles, maxing_price = False)
    # assign agents to cubicles
    small_market.assign_cubicles(byyear = True, verbose = True)
    # assign agents to half-days

    small_market.assign_halfdays(N = 1000)
    # export allocation
    small_market.export_allocation("allocation_small_market")

if backup:
    market.guess_prices()
    market.pricing_out(verbose=True)
    market.export_allocation("allocation_pricing_out")
    market.filling_empty_slots(verbose=True)
    market.export_allocation("allocation_backup")
if long_run:
    market.find_ACE(N = 1000, verbose = True, time_allowed=time_allowed)
    market.export_allocation("allocation_ACE")
    market.pricing_out(verbose=True)
    market.export_allocation("allocation_pricing_out")
    market.filling_empty_slots(verbose=True)
    market.export_allocation("allocation_backup")

