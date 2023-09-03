import numpy as np

# Import the itertools module
import itertools

# Define a function that takes an integer n as an argument
def generate_binary_vectors(n):
  # Use itertools.product to get all combinations of 0 and 1 with length n
  # Convert each tuple to a list and return a list of lists
  return [list(t) for t in itertools.product([0, 1], repeat=n)]

# Test the function with n = 10
n = 10
vectors = generate_binary_vectors(n)

# Print the result
print(vectors)
