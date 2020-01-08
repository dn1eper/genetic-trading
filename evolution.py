import pickle
import random
import math
from multiprocessing import Pool
from copy import deepcopy
from time import time
from datetime import datetime

from gene import GeneChain
from selector import Selector
from crosser import Crosser
from effects import Effects
from fitness import Evaluation
from mutator import Mutator
from util import print_flush

class Evolution:
    def __init__(self, MAX_GENS: int, evaluation: Evaluation, base_indiv: GeneChain,
                 crosser: Crosser, mutator: Mutator=None, selector: Selector=None, effects: list=[],
                 verbose: bool=True):
        self.best_fitness = -math.inf
        self._base_indiv = base_indiv
        self._evaluation = Evaluation(spread=0.0001)
        self._crosser = crosser
        self._mutator = mutator
        self._effects = effects
        self._selector = selector
        self._max_gens = MAX_GENS
        self._verbose = verbose

    def init_random_indivs(self, size, min_fitness=-math.inf, min_models=-math.inf, MAX_TRIES = 1000):
        start_time = time()
        self._individuals = [deepcopy(self._base_indiv) for i in range(size)]
        count = 0
        for i, indiv in enumerate(self._individuals):
            indiv.set_random()

            if min_fitness != -math.inf or min_models != -math.inf:
                idx = 1
                count += 1
                self._evaluation.evaluate(indiv)
                while indiv.fitness.score < min_fitness or indiv.fitness.models < min_models:
                    print_flush("Found {} indivs with {} tries (fitness={}, models={})".format(i, count, indiv.fitness.score, indiv.fitness.models))
                    if idx > MAX_TRIES:
                        raise AttributeError("Failed to generate random individual with min fitness {} and min models {}. {} times"
                            .format(min_fitness, min_models, MAX_TRIES))

                    indiv.set_random()
                    self._evaluation.evaluate(indiv)
                    idx += 1
                    count += 1
                print_flush("Found {} indivs with {} tries ({}s.)".format(i+1, count, round(time() - start_time)))

    def load_indivs(self, filename, size=None):
        precomp_indivs = []
        with open(filename, 'rb') as file:
            precomp_indivs = pickle.load(file)
            self._individuals = precomp_indivs
            if size is not None:
                self._individuals = self._individuals[:size]
            self._sort_indivs()

    def run(self):
        start_time = time()

        for i in range(self._max_gens):
            gen_time = time()
            self._next_generation()
            if self._verbose:
                print_flush("Generation: {} | Best fitness: {} | Models: {} | Time: avg per gen = {}s. total = {}s.)"
                    .format(i+1, round(self._individuals[0].fitness.score), self._individuals[0].fitness.models, round(time() - gen_time), round(time() - start_time)))
                gen_time = time()

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

        self.dump_indivs('best_indiv.pkl', best=1)

    def best(self):
        return self._individuals[0]

    def _compute_fitness(self, sort=True):
        if self._individuals:
            
            #for individ in self._individuals:
            #    if individ.fitness is None:
            #        self._evaluation.evaluate(individ)
            indivs = [individ for individ in self._individuals if individ.fitness is None]
            pool = Pool(8)
            results = pool.map(self._evaluation.evaluate, indivs)
            pool.close()
            pool.join()

            for i, individ in enumerate(indivs):
                individ.fitness = results[i]

            if sort:
                self._sort_indivs()

    def _sort_indivs(self):
        if self._individuals:
            self._individuals.sort(key=lambda i: i.fitness.score, reverse=True)

    def dump_indivs(self, filename, best: int=None):
        if self._individuals:
            with open(filename, 'wb+') as file:
                dump = self._individuals
                if best:
                    self._sort_indivs()
                    dump = self._individuals[:best]
                pickle.dump(dump, file)

    # PRINT
    def print_indiv(self, indiv: GeneChain):
        print("fitness: " + str(indiv.fitness.score) + " models: " + str(indiv.fitness.models) + " max account value: " +
              str(indiv.fitness.max_account) + " max drawdown: " + str(indiv.fitness.max_drawdown))

    def print_generation(self, message=None):
        if message is not None:
            print(message)
        for indiv in self._individuals:
            self.print_indiv(indiv)
