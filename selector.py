from abc import ABC, abstractmethod
from gene import GeneChain

class Selector(ABC):
    @abstractmethod
    def select(self, gene_chains:list) -> list:
        pass

class TopNSelector(Selector):
    def __init__(self, n:int):
        if n > 0:
            self._n = n
        else:
            raise ValueError("n must be greater than 0")

    def select(self, gene_chains:list):
        return gene_chains[:self._n]