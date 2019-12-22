from abc import ABC, abstractmethod
from gene import GeneChain

class Selector(ABC):
    @abstractmethod
    def select(self, gene_chains:list) -> list:
        pass

class TopNSelector(Selector):
    def __init__(self, n:int):
        self._n = n

    def select(self, gene_chains:list):
        return gene_chains[:self._n]