from gene import Gene, GeneChain
from genetic import Genetic
from selector import TopNSelector
from util import print_row, print_header
from loader import load_data, load_params
from crosser import RandomGeneCrosser
import numpy as np

# Load data
data = load()
params = load_params()

# Init base gene chain
base_gene_chain = GeneChain()
for column in data:
    dtype = data[column].dtype
    min_value = min(data[column])
    max_value = max(data[column])
    base_gene_chain.add(Gene(dtype.type, max_value, min_value, -1))

for index, row in params.iterrows():
    dtype = np.dtype(row["TYPE"])
    min_value = dtype.type(row["MIN"])
    max_value = dtype.type(row["MAX"])
    base_gene_chain.add(Gene(dtype.type, max_value, min_value, -1))\

# Init genetic algorithm
genetic = Genetic(
    max_generations=20,
    max_individuals=10,
    base_gene_chain=base_gene_chain,
    crosser=RandomGeneCrosser(),
    selector=TopNSelector(3))

# Run
genetic.run()

for i, gene in enumerate(genetic.best()):
    print(i, gene)