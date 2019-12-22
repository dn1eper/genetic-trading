from gene import GeneChain
from Data.main import build_db, evaluate

class Fitness:
    def __init__(self):
        build_db()

    def calc(self, individ:GeneChain) -> float:
        return evaluate(self._individ_converter(individ))

    def _individ_converter(self, individ:GeneChain) -> list:
        result = list()
        for gene in individ:
            if not gene._interval:
                result.append(gene.value())
            else:
                result.append([gene.min(), gene.max()])
        
        return result
