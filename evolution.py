import pickle
import random
import math
from copy import deepcopy
from time import time
from datetime import datetime

from gene import GeneChain
from selector import Selector
from crosser import Crosser
from effects import Effects
from fitness import Fitness
from mutator import Mutator
from util import print_flush

class Evolution:
    def __init__(self, MAX_GENS: int, fitness: Fitness, base_indiv: GeneChain,
                 crosser: Crosser, mutator: Mutator=None, selector: Selector=None, effects: list=[],
                 verbose: bool=True, multithreaded=True):
        self._base_indiv = base_indiv
        self._fitness = fitness
        self._crosser = crosser
        self._mutator = mutator
        self._effects = effects
        self._selector = selector
        self._max_gens = MAX_GENS
        self._verbose = verbose

        self._gpu = False
        self._multithread = multithreaded

        self._start_time = None
        self._run_results = None

    def init_random_indivs(self, size, min_fitness=-math.inf, min_models=0, POOL_SIZE=10):
        if min_fitness == -math.inf and min_models == 0:
            self._individuals = [deepcopy(self._base_indiv) for i in range(size)]
            for indiv in self._individuals:
                indiv.set_random()
        else:
            self._start_time = time()
            indivs = []
            tries = 0

            self._individuals = [deepcopy(self._base_indiv) for i in range(POOL_SIZE)]
            while len(indivs) < size:
                for indiv in self._individuals:
                    indiv.set_random()

                self._compute_fitness(sort=False)

                for indiv in self._individuals:
                    if indiv.fitness.fitness >= min_fitness and indiv.fitness.models >= min_models:
                        indivs.append(indiv)
                
                tries += POOL_SIZE
                print_flush("Found {} indivs with {} tries ({}s.)".format(len(indivs), tries, round(time() - self._start_time)))

            self._individuals = indivs[:size]

    def load_indivs(self, filename, size=None):
        with open(filename, 'rb') as file:
            self._individuals = pickle.load(file)
            if size is not None:
                self._individuals = self._individuals[:size]

    def run(self, dump=None, gpu=False, multithreaded=True):
        self._gpu = gpu
        self._multithread = multithreaded
        self._run_results = []
        self._start_time = time()

        for i in range(self._max_gens):
            gen_time = time()
            self._next_generation()
            self._run_results.append({"fitness": self._individuals[0].fitness, "time": round(time() - gen_time)})
            self._run_verbose()

            if dump is not None:
                self.dump(dump)

    def _run_verbose(self):
        if self._verbose:
            print_flush("Generation: {} | Best fitness: {} | Best ratio: {} | Models: {} | Time: avg per gen = {}s. total = {}s.".format(
                len(self._run_results),
                round(self._run_results[-1]["fitness"].fitness),
                round(self._run_results[-1]["fitness"].ratio),
                self._run_results[-1]["fitness"].models, 
                self._run_results[-1]["time"], 
                round(time() - self._start_time)))

    def _next_generation(self):
        prev_gen = deepcopy(self._individuals)
        size = len(self._individuals)

        # cross individ
        self._individuals = self._crosser.cross(self._individuals)

        # add mutatation
        if self._mutator is not None:
            self._compute_fitness()
            self._individuals = self._mutator.mutate(self._individuals)

        # perform last computations
        for effect in self._effects:
            self._compute_fitness()
            effect.effect(self._individuals)
        
        # select best individuals
        self._compute_fitness()
        if self._selector is not None:
            self._individuals = self._selector.select(self._individuals, size)
        
        if len(self._individuals) != size:
            raise Exception("Population size at start and end of the generation is not equal")

    def best(self):
        return self._individuals[0]

    def _compute_fitness(self, sort=True):
        if self._individuals:
            self._fitness.calc([ indiv for indiv in self._individuals if indiv.fitness is None ], gpu=self._gpu, multithreaded=self._multithread)
            if sort:
                self._sort_indivs()

    def _sort_indivs(self):
        if self._individuals:
            self._individuals.sort(key=lambda i: i.fitness.fitness, reverse=True)

    def dump(self, filename, best: int=None):
        if self._individuals:
            with open(filename, 'wb+') as file:
                dump = self._individuals
                if best:
                    self._sort_indivs()
                    dump = self._individuals[:best]
                pickle.dump(dump, file)

    # PRINT
    def print_generation(self, message=None):
        if message is not None:
            print(message)
        for indiv in self._individuals:
            print(indiv.fitness)
