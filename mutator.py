from copy import deepcopy
from abc import ABC, abstractmethod

class Mutator(ABC):
    @abstractmethod
    def mutate(self) -> list:
        pass

class RandomGeneChainMutator(Mutator):
    def __init__(self, n:int):
        if n > 0:
            self._n = n
        else:
            raise ValueError("n must be greater than 0")

    def mutate(self, gene_chains:list):
        result = [deepcopy(gene_chains[0]) for i in range(self._n)]
        for gene_chain in result:
            gene_chain.random()
        return result