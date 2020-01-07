from abc import ABC, abstractmethod
from copy import deepcopy

class Effects(ABC):
    @abstractmethod
    def effect(self, gene_chains: list) -> list:
        pass


class KolyanEffects(Effects):
    def __init__(self, rep: float):
        self.rep = rep # part of the worst individuals being replaced by the best individuals from the prev generation

    def effect(self, prev_individuals: list, current_individuals: list):
        gen_size = len(current_individuals)
        replace_indivs = int(gen_size * self.rep)
        if replace_indivs == 0:
            replace_indivs = 1

        current_individuals[gen_size - replace_indivs:] = deepcopy(prev_individuals[:replace_indivs])

        return current_individuals
