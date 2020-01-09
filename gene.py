from random import uniform, randint
from fitness import evaluate
from util import round05
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
            self._min_radius = min_radius
        if ordering < 0:
            self.min = round(self.min, -ordering)
            self.max = round(self.max, -ordering)

    def has_tag(self, tag: str):
        return tag in self._tags

    def __str__(self):
        string = ''
        string += 'gene params:\n'
        string += 'is interval: {}\nmin = {}, max = {}\n'.format(self.is_interval, self.min, self.max)
        if self.is_interval and self._value is not None:
            string += 'center = {}, radius = {}'.format(self.value(), self.radius())
        elif not self.is_interval and self._value is not None:
            string += 'value = {}'.format(self._value)
        else:
            string += 'this gene has no value yet...'

        return string + '\n'

    def value(self, value=None):
        if value is None:
            if self._value is None:
                raise ValueError("Access to Gene value before set")
            else:
                return self._value
        else:
            value = self.type(value)
            if self.is_interval:
                if (self._radius is not None and self.min + self._radius <= value <= self.max - self._radius) or \
                    (self._radius is None and self.min + self._min_radius <= value <= self.max - self._min_radius):
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
        if radius is None:
            if self._radius is None:
                if not self.is_interval:
                    return 0
                else:
                    raise ValueError("Access Gene radius before set")
            return self._radius
        else:
            if not self.is_interval:
                raise ValueError("Seting radius in non-interval Gene")
            if (self._value is not None and self._value - radius >= self.min and self._value + radius <= self.max) or \
               (self._value is None and radius < self._min_radius):
                self._radius = self.type(radius)
            else:
                raise ValueError("Wrong Gene radius")

    def set_random(self):
        """
        Set random Gen value
        """
        if np.issubdtype(self.type, np.integer):
            if self.is_interval:
                self._value = round05(uniform(self.min + self._min_radius, self.max - self._min_radius))
                max_radius = min(self._value - self.min, self.max - self._value)
                radius = round(uniform(self._min_radius, max_radius))
                self.radius(radius)
            else:
                self.value(randint(self.min, self.max))

        elif np.issubdtype(self.type, np.floating):
            if self.is_interval:
                self._value = uniform(self.min + self._min_radius, self.max - self._min_radius)
                max_radius = min(self._value - self.min, self.max - self._value)
                self.radius(uniform(self._min_radius, max_radius))
            else:
                self.value(uniform(self.min, self.max))

        else:
            raise Exception("Unexpected Gene value type")

    def set(self, gene):
        if type(gene) is not type(self):
            raise ValueError('Gene parameter has inappropriate type')
        if self.is_interval:
            self._radius = gene.radius()
        self._value = gene.value()

    def include(self, value):
        if self.is_interval:
            return self.value() - self.radius() <= value <= self.value() + self.radius()
        else:
            return self.value() == value

class GeneChain:
    """
    Genes chain of individ
    """

    def __init__(self, *genes: Gene):
        self._genes = genes
        self.fitness = None

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
        self.fitness = None

    def get(self, tag):
        genes = []
        for gene in self._genes:
            if gene.has_tag(tag):
                genes.append(gene)
        return tuple(genes)
