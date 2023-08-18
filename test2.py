import numpy as np

# Create an empty list to store the vectors
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
print(vectors)

# create numpy array
vectors = np.array(vectors)

# vector of prices
p = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# compute the price of each vector
prices = np.matmul(vectors, p)
print(prices)