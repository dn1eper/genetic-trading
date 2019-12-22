from copy import deepcopy
from abc import ABC, abstractmethod
from gene import GeneChain

class Crosser(ABC):
    @abstractmethod
    def cross(self, gene_chains:GeneChain) -> list:
        pass

class RandomGeneCrosser(Crosser):
    def cross(self, gene_chains:list):
        result = []
        for gene_chain in gene_chains:
            for i in range(len(gene_chain)):
                new_gene_chain = deepcopy(gene_chain)
                new_gene_chain[i].random()
                result.append(new_gene_chain)

        return result