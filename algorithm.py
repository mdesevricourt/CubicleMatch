from data import df
import numpy as np
import pandas as pd


number_of_cubicles = 6
# get the list of half-days from the dataframe
half_days = list(df.columns)

number_of_half_days = len(half_days) - 2
print(number_of_half_days)

# which index is the "No Match" option?
no_match_index = half_days.index("No Match")

# create names of cubicles
cubicles = ['C' + str(i) for i in range(1, number_of_cubicles+1)]

# create dictionary of cubicles
# each cubicle is a key
# each value is a list of zeros of length equal to the number of half-days
matching = {}
for cubicle in cubicles:
    matching[cubicle] = [0] * number_of_half_days

def findithpreferredhalfday(i, preferences):
    """This function returns the ith preferred half-day of a person, given their preferences.
    
    Args:
        i (int): The rank of the half-day.
        preferences (list): A list of the person's preferences over half-days. It specifies the rank of each half day for the person."""
    

    # get the index of the ith preferred half-day
    index = preferences.index(i)
    return index

print(findithpreferredhalfday(1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))



def match_person(name, preferences, matching):
    """This function returns a new match where a person is assigned to a cubicle-half-day combination, given their preferences. This function assumes that the person is not already assigned to a cubicle-half-day combination, 
    and that those already assigned have higher ranks than the person being matched.

    Args:
        name (str): The name of the person being matched.
        preferences (list): A list of the person's preferences over half-days. It specifies the rank of each half day for the person.
        matching (dict): A dictionary of the current matching of people to cubicles.
    
    Returns:
        dict: A dictionary of the new matching of people to cubicles.
    """

    for i in range(len(preferences)):
        # find the half day that is ranked i+1
        # get index of half day
        index = preferences.index(i+1)

        # if the preferred option is no match
        if index == no_match_index:
            # return the current matching
            print(f"{name} prefers not to be assigned to a cubicle-half-day combination given current matching")
            return matching
        
        count = count_people(matching)
        # get list of keys of count from lowest to highest
        sorted_cubicles = sorted(count, key=count.get)

        for cubicle in sorted_cubicles:
            # if the half day is available
            if matching[cubicle][index] == 0:
                # assign the person to the cubicle-half-day combination
                matching[cubicle][index] = name
                # return the new matching
                return matching
        
    # if the person is not assigned to a cubicle-half-day combination
    print(f"{name} couldn't be not assigned to a cubicle-half-day combination")

    return matching

def count_people(matching):
    '''Count the number of people assigned to each cubicle.
    
    Args:
        matching (dict): A dictionary of the current matching of people to cubicles.
    
    Returns:
        dict: A dictionary of the number of people assigned to each cubicle.
    '''

    # create a dictionary of the number of people assigned to each cubicle
    count = {}
    for cubicle in matching:
        count[cubicle] = 0
        # get list of people assigned to the cubicle
        people = matching[cubicle]
        # count the number of people assigned to the cubicle
        for person in people:
            if person != 0:
                count[cubicle] += 1
        
    return count


def match_people(df, matching):
    '''Match people to cubicles based on their preferences.
    
    Args:
        df (dataframe): The dataframe of people, their preferences and their ranks.
        matching (dict): A dictionary of the current matching of people to cubicles.
        
    Returns:
        dict: A dictionary of the new matching of people to cubicles.
    '''

    df_copy = df.copy()

    # sort the dataframe by rank
    df_copy.sort_values(by=['rank'], inplace=True)

    # for each person in the dataframe
    for name in df_copy.index:
        # get the preferences of the person
        preferences = df_copy.loc[name, half_days].tolist()
        # match the person to a cubicle-half-day combination
        matching = match_person(name, preferences, matching)

    return matching

def timetable(matching, half_days):
    '''Create a timetable from the matching given the name of the half days.
    
    Args:
        matching (dict): A dictionary of the current matching of people to cubicles.
        half_days (list): A list of the half days.
    
    Returns:
        dataframe: A dataframe of the timetable.
    '''

    # create a dictionary of the timetable
    timetable = {}
    
    for cubicle in matching:
        # get the list of people assigned to the cubicle
        people = matching[cubicle]
        # add the cubicle to the timetable
        timetable[cubicle] = people
    
    # create a dataframe from the timetable
    timetable = pd.DataFrame.from_dict(timetable, orient='index', columns=half_days[:-2])

    return timetable

if __name__ == "__main__":
    print(matching)
    matching = match_people(df, matching)
    print(matching)
    timetable = timetable(matching, half_days)
    print(timetable)
    print(count_people(matching))