from random import uniform, randint
from fitness import evaluate
from util import floor05
import numpy as np
import pandas as pd


class Gene:
    """
    One gene of individ with value and radius(for value range)
    """
    def __init__(self, is_interval: bool, dtype: type, max_value=0, min_value=0, min_radius=0, ordering: int = 0, tags=[]):
        self.type = dtype
        self.max = max_value
        self.min = min_value
        self.is_interval = is_interval
        self._ordering = ordering
        self._tags = tags
        self._radius = None
        self._value = None

        if is_interval:
            self.min_radius = min_radius
        if ordering < 0:
            self.min = round(self.min, -ordering)
            self.max = round(self.max, -ordering)

    def has_tag(self, tag: str):
        return tag in self._tags

    def __str__(self):
        string = ''
        string += 'gene params:\n'
        string += 'is interval: {}\nmin value = {}\nmax value = {}\n'.format(str(self.is_interval), str(self.min), str(self.max))
        if self.is_interval and self._value is not None:
            string += 'range start = {}\nrange center = {}\nrange end = {}\n'.format(str(self._range_start), str(self._range_center), str(self._range_end))
        elif not self.is_interval and self._value is not None:
            string += 'value = {}'.format(str(self._value))
        else:
            string += 'this gene has no value yet...'

        return string

    def value(self, value=None):
        if value is None:
            if self._value is None:
                raise ValueError("Access to Gene value before set")
            else:
                return self._value
        else:
            value = self.type(value)
            if self._is_interval:
                if (self._radius is not None and self.min + self._radius <= value <= self.max - self._radius) or \
                    (self._radius is None and self.min + self.min_radius <= value <= self.max - self.min_radius):
                    self._value = value
            else:
                if self.max >= value >= self.min:
                    self._value = value
                else:
                    raise ValueError("Wrong Gene value")

    def radius(self, radius=None):
        """
        Get or set Gene radious
        """
        if not self.is_interval:
            raise ValueError("Getting/setting radius in non-interval Gene")
        if radius is None:
            if self._radius is None:
                raise ValueError("Access Gene radius before set")
            return self._radius
        else:
            radius = self.type(radius)
            if (self._range_center is not None and self._range_center - radius >= self.min and self._range_center + radius <= self.max) or \
               (self._range_center is None and radius < self.min_radius):
                self._radius = radius
                # if self._range_center != None:
                #     self._range_start = self._range_center - self._radius
                #     self._range_end = self._range_center + self._radius
            else:
                raise ValueError("Wrong Gene radius")

    def set_random(self):
        """
        Set random Gen value
        """
        if np.issubdtype(self.type, np.integer):
            if self.is_interval:
                self._value = floor05(uniform(self.min + self.min_radius, self.max - self.min_radius))
                max_radius = min(self._range_center - self.min, self.max - self._range_center)
                radius = round(uniform(self.min_radius, max_radius))
                self.radius(radius)
            else:
                self.value(randint(self.min, self.max))

        elif np.issubdtype(self.type, np.floating):
            if self.is_interval:
                self._value = uniform(self.min + self.min_radius, self.max - self.min_radius)
                max_radius = min(self._range_center - self.min, self.max - self._range_center)
                self.radius(uniform(self.min_radius, max_radius))
            else:
                self.value(uniform(self.min, self.max))

        else:
            raise Exception("Unexpected Gene value type")

    def set(self, gene):
        if type(gene) is not type(self):
            raise ValueError('gene parameter has inappropriate type')
        if self.is_interval:
            self._radius = gene.radius()
            self.range_center(gene.range_center())
        else:
            self._value = gene.value()

    def include(self, value):
        if self._is_interval:
            return self.value() - self.radius() <= value <= self.value() + self.radius()
        else:
            return self.value() == value



class GeneChain:
    """
    Genes chain of individ
    """

    def __init__(self, *genes: Gene):
        self._genes = genes
        self.score = None
        self.models = None
        self.max_drawdown = None
        self.max_account = None

    def __len__(self):
        return len(self._genes)

    def __iter__(self):
        for gene in self._genes:
            yield gene

    def __getitem__(self, index: int):
        return self._genes[index]

    def add(self, gene: Gene):
        self._genes += gene,

    def set_random(self):
        """
        Set random all Genes in Chain
        """
        for gene in self._genes:
            gene.set_random()

    def set_rand_gene_chain(self, min_fitness, evaluation): # TODO: удалить нахрен
        print('generating random gene chain with min fitness =', min_fitness, 'iterations:')
        self.set_random()
        idx = 1
        while evaluation.evaluate(self) < min_fitness:
            if idx == 1000:
                raise AttributeError("Failed to generate random individual with min fitness " + \
                                     str(min_fitness) + " 1000 times")
            # print('iteration', idx)
            self.set_random()
            idx += 1

        print('exited')

    def get(self, tag):
        genes = []
        for gene in self._genes:
            if gene.has_tag(tag):
                genes.append(gene)
        return tuple(genes)

