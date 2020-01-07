import sys
sys.path.append("C:\\Users\\Zadne\\AppData\\Roaming\\Python\\Python38\\site-packages")
from gene import Gene, GeneChain
# from evolution import Evolution
from selector import TopNSelector
from util import print_row, print_header
from loader import load_indivs, load_indivs_pandas, load_results, load_trading_params, load_nontrading_params
from crosser import RandomGeneCrosser, PolinomiaCrosser
from mutator import RandomGeneChainMutator, GausianMutator
import numpy as np

# Load data
# db_indivs = load_indivs()   # list, for fitness function; db individuals have only non-trading parameters
db_indivs = load_indivs_pandas()   # pandas object, for precomputation
db_results = load_results() # list, for fitness function
trading_params = load_trading_params()  # pandas object, for precomputation
nontrading_params = load_nontrading_params()    # pandas object, for precomputation

# Init base gene chain
base_gene_chain = GeneChain()
for index, row in nontrading_params.iterrows():
    is_trading = False
    dtype = np.dtype(row["TYPE"])
    print(index, dtype)
    print(type(db_indivs_pandas[index]))
    is_interval = True if dtype != bool else False
    min_range = dtype.type(row["MIN_RANGE"])
    max_value = dtype.type(max(db_indivs_pandas[index]))
    min_value = dtype.type(min(db_indivs_pandas[index]))
    base_gene_chain.add(Gene(dtype=dtype, is_interval=is_interval, is_trading=is_trading,
                             min_value=min_value, max_value=max_value, ordering=-1))

for index, row in trading_params.iterrows():
    is_trading = True
    is_interval = False
    dtype = np.dtype(row["TYPE"])
    min_value = dtype.type(row["MIN"])
    max_value = dtype.type(row["MAX"])
    base_gene_chain.add(Gene(dtype=dtype, is_interval=is_interval, is_trading=is_trading,
                             min_value=min_value, max_value=max_value, ordering=-1))


# Init genetic algorithm
evolution = Evolution(
    max_generations=100,
    start_population=10,
    base_gene_chain=base_gene_chain,
    crosser=PolinomiaCrosser(indpb=0.2),
    selector=TournamentSelector(3),
    mutator=GausianMutator(sigma=0.1, indpb=0.2))

# Run
genetic.run()

for i, gene in enumerate(genetic.best()):
    print(i, gene)