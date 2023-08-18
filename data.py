# create fake data to test the algorithm

# Import pandas library for data manipulation
import pandas as pd

# Create a list of 60 fake names
names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Harry", "Irene", "Jack", "Kelly", "Leo", "Mia", "Noah", "Olivia", "Peter", "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zack", "Anna", "Ben", "Cathy", "Dan", "Ella", "Fred", "Gina", "Henry", "Isabel", "Jake", "Kim", "Liam", "Maya", "Nate", "Oscar", "Penny", "Quentin", "Ruby", "Steve", "Tina", "Umar", "Vanessa", "Will","Zoe", "Alex", "Bella", "Chris", "Diana", "Eric", "Fiona", "George", "Hannah", "Ivan", "Julia"]
print(len(names))
# Create a list of half-days in the week
options = ["Monday am","Monday pm","Tuesday am","Tuesday pm","Wednesday am","Wednesday pm","Thursday am","Thursday pm","Friday am","Friday pm", 'No Match']

# Create an empty dataframe with names as rows and half_days as columns
df = pd.DataFrame(index=names, columns=options)

# Fill the dataframe with shuffled numbers from 1 to 10
import random
# set seed
random.seed(123)
for name in names:
    # Create a list of numbers from 1 to 10
    numbers = list(range(1, len(options)+1))
    # Shuffle the list randomly
    random.shuffle(numbers)
    # Assign the shuffled numbers to each half-day for each person
    for i, half_day in enumerate(options):
        df.loc[name, half_day] = numbers[i]

rank = list(range(1, len(names)+1))
random.shuffle(rank)
df['rank'] = rank
# Print the dataframe
if __name__ == "__main__":
    print(df)

