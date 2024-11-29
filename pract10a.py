#simple genetic algorithm

import random

POPULATION_SIZE = 100
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP
ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''
TARGET = "I love MCC"

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(cls):
        return random.choice(GENES)

    @classmethod
    def create_gnome(cls):
        return [cls.mutated_genes() for _ in range(len(TARGET))]

    def mate(self, partner):
        child_chromosome = [
            gp1 if random.random() < 0.45 else gp2 if random.random() < 0.90 else self.mutated_genes()
            for gp1, gp2 in zip(self.chromosome, partner.chromosome)
        ]
        return Individual(child_chromosome)

    def cal_fitness(self):
        return sum(1 for gs, gt in zip(self.chromosome, TARGET) if gs != gt)

def main():
    generation = 1
    population = [Individual(Individual.create_gnome()) for _ in range(POPULATION_SIZE)]

    while True:
        population = sorted(population, key=lambda x: x.fitness)
        if population[0].fitness == 0:
            break
        
        next_gen = population[:10]  # Elitism: keep top 10%
        next_gen += [random.choice(population[:50]).mate(random.choice(population[:50])) for _ in range(90)]
        population = next_gen
        
        print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")
        generation += 1

    print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")

if __name__ == '__main__':
    main()
