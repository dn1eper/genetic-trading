from copy import deepcopy
from abc import ABC, abstractmethod
from gene import Gene, GeneChain
import selector
import random as rand
import numpy as np

class Crosser(ABC):
    @abstractmethod
    def cross(self, gene_chains:GeneChain) -> list:
        pass

class RandomGeneCrosser(Crosser):
    def cross(self, gene_chains: list):
        result = []
        for gene_chain in gene_chains:
            for i in range(len(gene_chain)):
                new_gene_chain = deepcopy(gene_chain)
                new_gene_chain[i].set_random()
                new_gene_chain.fitness = None
                result.append(new_gene_chain)

        return result


class Cross10to100(Crosser):
    """
    """
    def __init__(self, indpb, tournsize, save_part):
        self.indpb = indpb
        self.tournsize = tournsize
        self.save_part = save_part

    def cross(self, indivs: list):
        """'indivs' should be sorted by score"""
        if not indivs:
            raise ValueError("Empty 'indivs' parameter in crosser.Cross10to100")

        save_indivs = int(len(indivs) * self.save_part)
        save_indivs = save_indivs + 1 if save_indivs == 0 else save_indivs
        new_indivs = deepcopy(indivs[:save_indivs])

        # copying part of indivs
        copy_indivs = int(len(indivs) * self.save_part)
        if copy_indivs == 0:
            copy_indivs = 1
        copied_indivs = selector.TournamentSelector(individuals=indivs, k=copy_indivs, tournsize=self.tournsize)

        # copying selected individuals to fill up 'mating_pool'
        pool_len = len(indivs) - save_indivs
        mating_pool = []
        for idx in range(int(pool_len / copy_indivs)):
            mating_pool += deepcopy(copied_indivs)
        if len(mating_pool) != pool_len:
            mating_pool += deepcopy(copied_indivs[:len(pool_len) - len(mating_pool)])

        # mating individuals
        while mating_pool:
            parent1 = rand.choice(mating_pool)
            parent2 = rand.choice(mating_pool)
            # print('parent =', type(parent1))
            # print('parent =', type(parent1))
            if parent1 == parent2:
                new_indivs.append(parent1)
                mating_pool.remove(parent1)
            else:
                cross_polynomial(parent1, parent2, self.indpb)
                new_indivs.extend([parent1, parent2])
                mating_pool.remove(parent1)
                mating_pool.remove(parent2)

        return new_indivs



def cross_polynomial(ind1: GeneChain, ind2: GeneChain, indpb):
    if type(ind1) is not GeneChain or type(ind2) is not GeneChain:
        raise TypeError("ind1 or ind2 has wrong data type")
    for idx in range(len(ind1)):
        if rand.random() >= indpb:
            gen1_copy = deepcopy(ind1[idx])
            ind1[idx].set(ind2[idx])
            ind2[idx].set(gen1_copy)
