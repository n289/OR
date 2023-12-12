#!/usr/bin/env python
# coding: utf-8

# In[160]:


import pandas as pd
from gurobipy import Model, GRB, quicksum

# Assuming the CSV file is on your desktop
file_path = "Alabama.csv"

# Read the data into a Pandas DataFrame
census_df = pd.read_csv(file_path)


print(census_df.columns)


# In[143]:


import pandas as pd
from gurobipy import Model, GRB, quicksum

# Assuming the CSV file is on your desktop
file_path = "Alabama.csv"

# Read the data into a Pandas DataFrame
census_df = pd.read_csv(file_path)


print(census_df.columns)d

# Assuming the column name for total population is 'Alabama'
total_population_str = census_df.loc[census_df['Label (Grouping)'].str.strip() == 'Total:', 'Alabama'].values[0]

# Convert total population to an integer
total_population = int(total_population_str.replace(',', ''))


racial_composition = 'White alone'


white_alone_row = census_df[census_df['Label (Grouping)'].str.strip() == racial_composition]

if not white_alone_row.empty:
    # Use the corresponding row as the racial composition data
    racial_data = white_alone_row['Alabama'].values[0]
else:
    raise ValueError(f"Row related to '{racial_composition}' not found.")

# Create a new model
model = Model("congressional_redistricting")

# Decision variables
num_precincts = len(census_df)
districts = range(1, 8)  # Assuming 7 districts
precincts = range(1, num_precincts + 1)

x = {}  # Binary variable: precinct i is assigned to district j
for i in precincts:
    for j in districts:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# Objective function: Minimize population divergence among districts
model.setObjective(
    quicksum((x[i, j] * int(census_df.loc[i - 1, 'Alabama'].replace(',', ''))) for i in precincts for j in districts),
    GRB.MINIMIZE
)

# Constraint 1: Each precinct must be assigned to exactly one district
for i in precincts:
    model.addConstr(sum(x[i, j] for j in districts) == 1, f"Assign_Precinct_{i}_to_One_District")

# Constraint 2: Ensure that the total population in each district is approximately equal
target_population_per_district = total_population / len(districts)
for j in districts:
    model.addConstr(
        quicksum(x[i, j] * int(census_df.loc[i - 1, 'Alabama'].replace(',', '')) for i in precincts) >= 0.0  # Relaxing the lower bound
    )
    model.addConstr(
        quicksum(x[i, j] * int(census_df.loc[i - 1, 'Alabama'].replace(',', '')) for i in precincts) <= total_population  # Relaxing the upper bound
    )

# Solve the model
model.optimize()

# Check the optimization status
if model.status == GRB.OPTIMAL:
    print("Optimal solution found")
else:
    print(f"Optimization terminated with status {model.status}")

# Extract the solution and visualize the results
for i in precincts:
    for j in districts:
        if x[i, j].x > 0.5:  # Check if the variable is assigned
            print(f"Precinct {i} is assigned to District {j}")


district_populations = {}
for j in districts:
    population = sum(x[i, j].x * int(census_df.loc[i - 1, 'Alabama'].replace(',', '')) for i in precincts)
    district_populations[f"District {j}"] = population

print("District Populations:")
print(district_populations)


print(f"Total Population: {total_population}")


# In[150]:


import pandas as pd
from gurobipy import Model, GRB, quicksum
import networkx as nx
import matplotlib.pyplot as plt

# Assuming the CSV file is on your desktop
file_path = "Alabama.csv"

# Read the data into a Pandas DataFrame
census_df = pd.read_csv(file_path)

# Assuming the column name for total population is 'Alabama'
total_population_str = census_df.loc[census_df['Label (Grouping)'].str.strip() == 'Total:', 'Alabama'].values[0]

# Convert total population to an integer
total_population = int(total_population_str.replace(',', ''))

# Assuming 'White alone' is a value in the 'Label (Grouping)' column
racial_composition = 'White alone'

# Find the row index where 'White alone' appears in the 'Label (Grouping)' column
white_alone_row = census_df[census_df['Label (Grouping)'].str.strip() == racial_composition]

if not white_alone_row.empty:
    # Use the corresponding row as the racial composition data
    racial_data = white_alone_row['Alabama'].values[0]
else:
    raise ValueError(f"Row related to '{racial_composition}' not found.")

# Create a new model
model = Model("congressional_redistricting")

# Decision variables
num_precincts = len(census_df)
districts = range(1, 8)  # Assuming 7 districts
precincts = range(1, num_precincts + 1)

x = {}  # Binary variable: precinct i is assigned to district j
for i in precincts:
    for j in districts:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# Objective function: Minimize population divergence among districts
model.setObjective(
    quicksum((x[i, j] * int(census_df.loc[i - 1, 'Alabama'].replace(',', ''))) for i in precincts for j in districts),
    GRB.MINIMIZE
)

# Constraints: Add constraints based on federal and state criteria
# Constraint 1: Each precinct must be assigned to exactly one district
for i in precincts:
    model.addConstr(sum(x[i, j] for j in districts) == 1, f"Assign_Precinct_{i}_to_One_District")

# Constraint 2: Ensure that the total population in each district is approximately equal
target_population_per_district = total_population / len(districts)
for j in districts:
    model.addConstr(
        quicksum(x[i, j] * int(census_df.loc[i - 1, 'Alabama'].replace(',', '')) for i in precincts) >= 0.0  # Relaxing the lower bound
    )
    model.addConstr(
        quicksum(x[i, j] * int(census_df.loc[i - 1, 'Alabama'].replace(',', '')) for i in precincts) <= total_population  # Relaxing the upper bound
    )

# Solve the model
model.optimize()

# Check the optimization status
if model.status == GRB.OPTIMAL:
    print("Optimal solution found")
else:
    print(f"Optimization terminated with status {model.status}")

# Create a graph to visualize the districts and precincts
G = nx.Graph()

# Add nodes for precincts
for i in precincts:
    G.add_node(f"Precinct {i}", color='skyblue', population=int(census_df.loc[i - 1, 'Alabama'].replace(',', '')))

# Add nodes for districts
for j in districts:
    G.add_node(f"District {j}", color='lightcoral')

# Add edges for assigned districts
for i in precincts:
    for j in districts:
        if x[i, j].x > 0.5:  # Check if the variable is assigned
            G.add_edge(f"Precinct {i}", f"District {j}")

# Set node colors based on assignment
node_colors = [G.nodes[node]['color'] for node in G.nodes]

# Draw the graph with Kamada-Kawai layout
pos = nx.kamada_kawai_layout(G)
nx.draw(G, pos, with_labels=True, node_color=node_colors, font_color='black', font_size=8, node_size=800, cmap=plt.cm.Blues, alpha=0.8, edge_color='gray')

# Display population size as node labels
labels = {node: G.nodes[node].get('population', '') for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=8)

# Display the plot
plt.title("Congressional Redistricting")
plt.show()


# In[167]:


import matplotlib.pyplot as plt

# Check the optimization status
if model.status == GRB.OPTIMAL:
    print("Optimal solution found")

    # Create a bar chart for district populations
    district_names = [f"District {j}" for j in districts]
    district_populations = [district_populations[d] for d in district_names]

    plt.bar(district_names, district_populations)
    plt.xlabel('Districts')
    plt.ylabel('Population')
    plt.title('Population Distribution Among Districts')
    plt.show()

    # Create a bar chart for precinct assignments
    precinct_assignments = {i: None for i in precincts}
    for i in precincts:
        for j in districts:
            if x[i, j].x > 0.5:
                precinct_assignments[i] = j

    precinct_names = [f"Precinct {i}" for i in precincts]
    precinct_districts = [precinct_assignments[i] for i in precincts]

    plt.bar(precinct_names, precinct_districts)
    plt.xlabel('Precincts')
    plt.ylabel('Assigned Districts')
    plt.title('Precinct Assignments to Districts')
    plt.show()

else:
    print(f"Optimization terminated with status {model.status}")


# In[ ]:




