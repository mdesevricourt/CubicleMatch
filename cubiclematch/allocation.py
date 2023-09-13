"""Defines the allocation class for the cubiclematch package."""


class Allocation:
    def __init__(self):
        pass

    def get_assignment(self, agent):
        """Return the assignment of the agent.

        Args:
            agent (Agent): An agent.

        Returns:
            tuple: A tuple of the form (bundle, cubicle where bundle is a numpy array of 0s and 1s.
        """
        pass

    def get_utility(self, agent):
        """Return the utility of the agent.

        Args:
            agent (Agent): An agent.

        Returns:
            float: The utility of the agent.
        """
        pass

    def agents(self):
        """Return the agents in the allocation.

        Returns:
            list: A list of agents.
        """
        pass


    
