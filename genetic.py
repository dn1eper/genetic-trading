from copy import deepcopy
from gene import GeneChain
from selector import Selector
from crosser import Crosser
from fitness import fitness

class Genetic:
    def __init__(self, max_generations:int, max_individuals:int, base_gene_chain:GeneChain, crosser:Crosser, selector:Selector):
        self._selector = selector
        self._crosser = crosser
        self._max_generations = max_generations
        # Create random individuals
        self._individuals = [deepcopy(base_gene_chain) for i in range(max_individuals)]
        for individ in self._individuals:
            individ.random()

    def _next_generation(self):
        # cross individ
        self._individuals = self._crosser.cross(self._individuals)
        # calculate fitness function for each indidivid
        for individ in self._individuals:
            individ.score = fitness(individ)
        # sort individ by score
        self._individuals.sort(key=lambda i: i.score, reverse=True)
        # get best individ
        self._individuals = self._selector.select(self._individuals)

    def run(self):
        for i in range(self._max_generations):
            self._next_generation()