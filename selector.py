from abc import ABC, abstractmethod
from gene import GeneChain
from fitness import evaluate, Evaluation
from copy import deepcopy
import random


class Selector(ABC):
    @abstractmethod
    def select(self, gene_chains: list) -> list:
        pass


class TopNSelector(Selector):
    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be greater than 0")
        self._n = n

    def select(self, indivs: list, n=None):
        return indivs[:self._n]


# class KoliaSelection(Selector):
#     def __init__(self, def_tournsize=None):
#         self._def_tournsize = def_tournsize

def TournamentSelector(individuals, k, tournsize):
    """
    :param individuals: list of individuals to select from
    :param k: number of individuals to select
    :param tournsize: number of random individuals from witch the best one is selected each iteration
    """
    tournsize = int(tournsize)
    # if tournsize == None and self._def_tournsize == None:
    #     raise ValueError("Didn't specified 'tournsize' parameter")
    # if tournsize == None:
    #     tournsize = self._def_tournsize
    chosen = []
    for i in range(k):
        aspirants = list(random.choice(individuals) for idx in range(tournsize))
        chosen.append(deepcopy(max(aspirants, key=lambda ind: ind.score)))
    return chosen
