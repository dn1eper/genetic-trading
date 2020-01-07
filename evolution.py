from time import time
from copy import deepcopy
from gene import GeneChain
from selector import Selector, TournamentSelector
from crosser import Crosser
from Effects import Effects
from fitness import Evaluation, evaluate
from mutator import Mutator
import pickle
import random
import math
from time import time
from datetime import datetime

class Evolution:
    def __init__(self, MAX_GENS: int, POP_SIZE: int, evaluation: Evaluation, base_gene_chain: GeneChain,
                 crosser: Crosser, mutator: Mutator, effects: Effects, start_indiv_min_fitness=None,
                 verbose: bool = True):
        self.best_fitness = -math.inf
        self._base_gen_chain = base_gene_chain
        self._evaluation = Evaluation(spread=0.0001)
        self._crosser = crosser
        self._mutator = mutator
        self._effects = effects
        self._max_generations = MAX_GENS
        self._verbose = verbose

        # Create random individuals
        # self.print_generation(message="Start generation fitnesses:")


        # print('creating start individuals...')
        precomp_indivs = []
        with open('valid_indivs.pkl', 'rb') as file:
            precomp_indivs = pickle.load(file)
            self._individuals = precomp_indivs
            self._individuals = TournamentSelector(precomp_indivs, POP_SIZE, len(precomp_indivs) / 15)
            self._sort_indivs()

        # self._individuals = list(random.choice(precomp_indivs) for idx in range(POP_SIZE))
        # self._individuals = [self.init_indiv(min_models=100) for i in range(POP_SIZE)]
        # self.dump_indivs()

        # self._sort_indivs()
        self.print_generation("Initial generation")


    def _next_generation(self):
        # self._sort_indivs()
        # prev_gen = deepcopy(self._individuals)
        # cross individ
        self._individuals = self._crosser.cross(self._individuals)
        # add mutatation
        self._compute_fitness_and_sort()
        self._individuals = self._mutator.mutate(self._individuals)
        # perform last computations
        self._compute_fitness_and_sort()
        # self.print_generation_fitnesses(message="generation fitnesses after crosser and mutator:")
        # self._individuals = self._effects.effect(prev_gen, self._individuals)
        # self._sort_indivs()
        # self.print_generation_fitnesses(message="generation fitnesses after effects:")
        self.dump_best_indiv()

    def test(self):
        print('start time =', datetime.now())
        while True:
            indiv = self.init_indiv()
            self._evaluation.evaluate(indiv)
            if indiv.models >= 100:
                self.print_indiv(indiv)

        self._sort_indivs()
        self.dump_indivs()

    def run(self):
        start_time = time()

        for i in range(self._max_generations):
            self._next_generation()
            if self._verbose:
                print("Generation:", i + 1, "| Best fitness:", round(self._individuals[0].score),
                      "Models:", self._individuals[0].models, "(%ss.)" % round(time() - start_time))
                self.print_generation("Fitnesses:")
                start_time = time()

    def best(self):
        return self._individuals[0]

    def print_indiv(self, indiv: GeneChain):
        print("fitness: " + str(indiv.score) + " models: " + str(indiv.models) + " max account value: " +
              str(indiv.max_account) + " max drawdown: " + str(indiv.max_drawdown))

    def print_generation(self, message=None):
        if message is not None:
            print(message)
        for indiv in self._individuals:
         self.print_indiv(indiv)

    def init_indiv(self, min_fitness=-math.inf, min_models=-math.inf):
        # print('generating random gene chain with min fitness =', min_fitness, 'iterations:')
        indiv = deepcopy(self._base_gen_chain)
        indiv.set_random()
        idx = 1
        if min_fitness != -math.inf or min_models != -math.inf:
            self._evaluation.evaluate(indiv)
            while indiv.score < min_fitness or indiv.models < min_models:
                if idx == 1000:
                    raise AttributeError("Failed to generate random individual with min fitness " +
                                         str(min_fitness) + " and min models " + min_models + " 1000 times")
                indiv.set_random()
                self._evaluation.evaluate(indiv)
                idx += 1

        # print('inited indiv')
        return indiv

    def _compute_fitness_and_sort(self, reverse=True):
        if self._individuals:
            for individ in self._individuals:
                individ.score = self._evaluation.evaluate(individ)
            self._individuals.sort(key=lambda i: i.score, reverse=reverse)

    def _sort_indivs(self, reverse=True):
        if self._individuals:
            self._individuals.sort(key=lambda i: i.score, reverse=reverse)

    def dump_indivs(self):
        with open('valid_indivs.pkl', 'wb+') as file:
            pickle.dump(self._individuals, file)

    def dump_best_indiv(self):
        with open('best_indiv.pkl', 'wb+') as file:
            pickle.dump(max(self._individuals, key=lambda indiv: indiv.score), file)
