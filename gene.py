from random import uniform, randint
import numpy as np

class Gene:
    """
    One gene of individ with value and radius (for value range)
    """
    def __init__(self, dtype:type, max_value, min_value = 0, ordering:int = 0, interval:bool = False):
        self._min = min_value
        self._max = max_value
        self._ordering = ordering
        self._type = dtype
        self._interval = interval
        self._value = None
        self._radius = 0
        if ordering < 0:
            self._min = round(self._min, -ordering)
            self._max = round(self._max, -ordering)

    def __str__(self):
        if self._interval:
            return str([self.min(), self.max()])
        else:
            return str(self.value())

    def value(self, value=None):
        """
        Get or set Gene value
        """
        if value == None:
            if self._value == None:
                raise ValueError("Access to Gene value before set")
            else:
                return self._value
        else:
            if value <= self._max and value >= self._min:
                self._value = value
            else:
                raise ValueError("Wrong Gene value")

    def radius(self, radius=None):
        """
        Get or set Gene raious
        """
        if radius == None:
            if self._radius == None:
                raise ValueError("Access to Gene radius before set")
            else:
                return self._radius
        else:
            if not self._interval:
                raise Exception("Can't set radius to non interval gene")
            if radius > 0:
                self._radius = radius
            else:
                raise ValueError("Wrong Gene radius")

    def min(self):
        return self._type(max(self._min, self.value() - self.radius()))

    def max(self):
        return self._type(min(self._max, self.value() + self.radius()))

    def random(self):
        """
        Set random Gen value
        """
        if np.issubdtype(self._type, np.integer):
            self._value = randint(self._min, self._max)
            if self._interval:
                self._radius = randint(0, int((self._max - self._min) / 2))

        elif np.issubdtype(self._type, np.floating):
            self._value = uniform(self._min, self._max)
            if self._ordering <= 0:
                self._value = round(self._value, -self._ordering)
            if self._interval:
                self._radius = uniform(0, (self._max - self._min) / 2)

        else:
            raise Exception("Unexpected Gene value type")

        if self._ordering > 0:
            self._value = self._value - (self._value % pow(10, self._ordering))

        self._radius = self._type(self._radius)
        self._value = self._type(self._value)

class GeneChain:
    """
    Genes chain of individ
    """
    def __init__(self, *genes:Gene):
        self._genes = genes
        self.score = None

    def __len__(self):
        return len(self._genes)

    def __iter__(self):
        for gene in self._genes:
            yield gene

    def __getitem__(self, index:int):
        return self._genes[index]

    def add(self, gene:Gene):
        self._genes += (gene, )

    def random(self):
        """
        Set random all Genes in Chain
        """
        for gene in self._genes:
            gene.random()
