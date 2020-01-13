
from copy import deepcopy
from abc import ABC, abstractmethod
import random as rand
import math

class Mutator(ABC):
    @abstractmethod
    def mutate(self) -> list:
        pass

class RandomGeneChainMutator(Mutator):
    def __init__(self, n:int):
        if n > 0:
            self._n = n
        else:
            raise ValueError("n must be greater than 0")

    def mutate(self, gene_chains: list):
        result = [deepcopy(gene_chains[0]) for i in range(self._n)]
        for gene_chain in result:
            gene_chain.set_random()
        return gene_chains + result


class MutRandomIndivs(Mutator):
    def __init__(self, sigma, indiv_indpb, gene_indpb, save_part):
        self.sigma = sigma
        self.indiv_indpb = indiv_indpb
        self.gene_indpb = gene_indpb
        self.save_part = save_part

    def mutate(self, individuals: list):
        save_indivs = int(len(individuals) * self.save_part)
        save_indivs = save_indivs + 1 if save_indivs == 0 else save_indivs
        new_indivs = deepcopy(individuals[:save_indivs])

        for idx in range(len(individuals) - save_indivs):
            indiv = deepcopy(rand.choice(individuals))
            if rand.random() < self.indiv_indpb:
                GausianMutator(indiv, self.sigma, self.gene_indpb)
            new_indivs.append(indiv)

        return new_indivs


def GausianMutator(individual, sigma, indpb):
    """
    Mutate individual in place; sigma much higher that 0.25 can will leed to long execution time
    :param sigma: mutation strength, 0.05 < sigma < 0.25 recommended
    :param indpb: independent probability of each gene to mutate
    :returns new individual
    """
    for idx, gene in enumerate(individual):
        if rand.random() > indpb:
            dtype = gene.type
            if dtype == bool:
                gene.value(not gene.value())
                continue

            min_value, max_value = gene.min, gene.max

            if not gene.is_interval:
                sigma_v = sigma * (min_value - max_value)
                if dtype == int and sigma_v < 0.5:
                    sigma_v = 0.5
                result = math.inf
                i = 0
                while not min_value <= result <= max_value:
                    result = rand.gauss(gene.value(), sigma_v)
                    if dtype == int:
                        result = dif.floor(result)

                    if i > 10000:
                        raise ValueError("tried to mutate trading attribute over 10 000 times")
                    i += 1

                gene.value(result)

            else:
                # finding center for new range
                rng_srt, rng_end, rng_ctr = gene.range_start(), gene.range_end(), gene.range_center()
                min_rng = gene.min_range
                min_rad = min_rng / 2
                rng = rng_end - rng_srt
                rng_rad = rng / 2
                min_rng_ctr, max_rng_ctr = min_value + (min_rng / 2), max_value - (min_rng / 2)
                sigma_c = sigma * (max_rng_ctr - min_rng_ctr)
                if dtype == int and sigma_c < 0.5:  # to make int variables with small range be able to mutate
                    sigma_c = 0.5

                if dtype == int and (rng_srt % 1 != 0 or rng_end % 1 != 0):
                    raise ValueError("int attribute has floating point range\n" + gene)

                counter = 0
                new_rng_ctr = math.inf
                while new_rng_ctr > max_rng_ctr or new_rng_ctr < min_rng_ctr:
                    new_rng_ctr = rand.gauss(rng_ctr, sigma_c)
                    if dtype == int:
                        new_rng_ctr = dif.floor_to_05(new_rng_ctr)
                    if counter >= 10000:
                        print("min_rng_ctr =", min_rng_ctr, "max_rng_ctr =", max_rng_ctr, rng_ctr, sigma_c)
                        raise ValueError("tried to generate new range center over 10000  times")
                    counter += 1

                max_rad = min(new_rng_ctr - min_value, max_value - new_rng_ctr)
                sigma_r = sigma * (max_rad - (min_rng / 2))
                if dtype == int and sigma_r < 0.5:
                    sigma_r = 0.5
                mu = min(rng_rad, max_rad)

                new_rng_rad = math.inf
                counter = 0
                while new_rng_rad < min_rad or new_rng_rad > max_rad:
                    new_rng_rad = rand.gauss(mu, sigma_r)
                    if dtype == int and new_rng_ctr % 1 == 0.5:
                        new_rng_rad = dif.floor_to_05(new_rng_rad)
                        if new_rng_rad % 0.5 != 0:
                            new_rng_rad = math.inf
                    elif dtype == int and new_rng_ctr % 1 == 0:
                        new_rng_rad = dif.floor(new_rng_rad)

                    if (counter >= 100):
                        print(new_rng_ctr, min_rad, min_value, max_value, sigma_r, sigma)
                        raise ValueError("tried to generate new range radius over 100 times")
                    counter += 1

                gene._range_center = new_rng_ctr
                gene.radius(new_rng_rad)
    return []






