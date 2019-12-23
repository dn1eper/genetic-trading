from time import time
from copy import deepcopy
from gene import GeneChain
from selector import Selector
from crosser import Crosser
from fitness import Fitness
from mutator import Mutator

class Genetic:
    def __init__(self, max_generations:int, start_population:int, 
        base_gene_chain:GeneChain, crosser:Crosser, selector:Selector, mutator:Mutator, 
        verbose:bool = True):

        self._selector = selector
        self._crosser = crosser
        self._mutator = mutator
        self._fitness = Fitness()
        self._max_generations = max_generations
        self._verbose = verbose
        # Create random individuals
        self._individuals = [deepcopy(base_gene_chain) for i in range(start_population)]
        for individ in self._individuals:
            individ.random()

    def _next_generation(self):
        # cross individ
        self._individuals = self._crosser.cross(self._individuals)
        # add mutatation
        self._individuals += self._mutator.mutate(self._individuals)
        # calculate fitness function for each indidivid
        for individ in self._individuals:
            individ.score = self._fitness.calc(individ)
        # sort individ by score
        self._individuals.sort(key=lambda i: i.score, reverse=True)
        # get best individ
        self._individuals = self._selector.select(self._individuals)

    def run(self):
        start_time = time()

        for i in range(self._max_generations):
            self._next_generation()
            if self._verbose:
                print("Generation:", i+1, "| Fitness:", round(self._individuals[0].score), "(%ss.)" % round(time() - start_time))
                start_time = time()

    def best(self):
        return self._individuals[0]