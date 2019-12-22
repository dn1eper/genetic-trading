from gene import Gene, GeneChain
from genetic import Genetic
from selector import TopNSelector
from util import print_row, print_header
from loader import load
from crosser import RandomGeneCrosser

# Load data
data = load()

# Init base gene chain
base_gene_chain = GeneChain()
for column in data:
    dtype = data[column].dtype
    min_value = min(data[column])
    max_value = max(data[column])
    base_gene_chain.add(Gene(dtype.type, max_value, min_value, -1))

# Init genetic algorithm
genetic = Genetic(
    max_generations=1,
    max_individuals=10,
    base_gene_chain=base_gene_chain,
    crosser=RandomGeneCrosser(),
    selector=TopNSelector(3))

# Run
genetic.run()

for individ in genetic._individuals:
    print(individ.score)