#Classes Genetic algorithm : City and Fitness 

import numpy as np
import random, operator, pandas as pd, matplotlib.pyplot as plt

class City:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def distance(self, city):
        return np.sqrt((self.x - city.x) ** 2 + (self.y - city.y) ** 2)

    def __repr__(self):
        return f"({self.x},{self.y})"


class Fitness:
    def __init__(self, route):
        self.route, self.distance, self.fitness = route, 0, 0.0

    def routeDistance(self):
        if self.distance == 0:
            self.distance = sum(self.route[i].distance(self.route[i+1 if i+1 < len(self.route) else 0]) for i in range(len(self.route)))
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


def createRoute(cityList):
    return random.sample(cityList, len(cityList))


def initialPopulation(popSize, cityList):
    return [createRoute(cityList) for _ in range(popSize)]


def rankRoutes(population):
    return sorted({i: Fitness(route).routeFitness() for i, route in enumerate(population)}.items(), key=operator.itemgetter(1), reverse=True)


def selection(popRanked, eliteSize):
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'], df['cum_perc'] = df.Fitness.cumsum(), 100 * df.Fitness.cumsum() / df.Fitness.sum()
    selectionResults = [popRanked[i][0] for i in range(eliteSize)]
    for _ in range(len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    return [population[i] for i in selectionResults]


def breed(parent1, parent2):
    geneA, geneB = sorted([int(random.random() * len(parent1)) for _ in range(2)])
    return parent1[geneA:geneB] + [gene for gene in parent2 if gene not in parent1[geneA:geneB]]


def breedPopulation(matingpool, eliteSize):
    pool, children = random.sample(matingpool, len(matingpool)), matingpool[:eliteSize]
    children += [breed(pool[i], pool[len(matingpool)-i-1]) for i in range(len(matingpool) - eliteSize)]
    return children


def mutate(individual, mutationRate):
    for i in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))
            individual[i], individual[swapWith] = individual[swapWith], individual[i]
    return individual


def mutatePopulation(population, mutationRate):
    return [mutate(ind, mutationRate) for ind in population]


def nextGeneration(currentGen, eliteSize, mutationRate):
    ranked = rankRoutes(currentGen)
    selectionResults = selection(ranked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    return mutatePopulation(children, mutationRate)


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance:", 1 / rankRoutes(pop)[0][1])
    for _ in range(generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    print("Final distance:", 1 / rankRoutes(pop)[0][1])
    return pop[rankRoutes(pop)[0][0]]


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = [1 / rankRoutes(pop)[0][1]]
    for _ in range(generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


if __name__ == "__main__":
    cityList = [City(x=int(random.random() * 200), y=int(random.random() * 200)) for _ in range(25)]
    geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
