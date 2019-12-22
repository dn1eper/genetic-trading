from gene import GeneChain
import random

def fitness(individ:GeneChain) -> int:
    return random.randint(1, 100)