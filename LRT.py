import random
import math
from deap import base
from deap import creator
from deap import tools
import csv
import matplotlib.pyplot as plt
import statistics

data = []
min_fitness = []
max_fitness = []
mean_fitness = []
std_fitness = []


def readFile():
    with open('data/Dataset-GA-normalized.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


def genotypeToPhenotype(individual):
    phenotype_array = []
    for start_index in range(10):
        phenotype_array.append(int((individual[(start_index * 8):(start_index * 8 + 7)]), 2))
    print(phenotype_array)


def fitnessFunction(individual):
    fitness = 0
    last_x = int(data[0][0])
    last_y = int(data[0][1])
    length = []
    unique = []
    for cell in individual:
        if cell not in unique:
            unique.append(cell)
        distance = (math.sqrt(
            (int(data[cell][0]) - last_x) * (int(data[cell][0]) - last_x) + (int(data[cell][1]) - last_y) * (
                        int(data[cell][1]) - last_y)))
        length.append(distance)
        fitness = fitness + float(data[cell][2]) + float(data[cell][3]) + float(data[cell][4])
        last_x = int(data[cell][0])
        last_y = int(data[cell][1])

    panadura_to_first = math.sqrt((int(data[150][0]) - (int(data[individual[0]][0])) * (
                int(data[150][0]) - (int(data[individual[0]][0]))) + (
                                               int(data[150][1]) - int(data[individual[0]][1])) * (
                                               int(data[150][1]) - int(data[individual[0]][1]))))
    last_to_pettah = math.sqrt((int(data[0][0]) - (int(data[individual[-1]][0])) * (
                int(data[0][0]) - (int(data[individual[-1]][0]))) + (int(data[0][1]) - int(data[individual[-1]][1])) * (
                                            int(data[0][1]) - int(data[individual[-1]][1]))))
    length.append(panadura_to_first)
    length.append(last_to_pettah)

    # minimize track length
    fitness = fitness + 10 * (math.sqrt(8 * 8 + 28 * 28)) / sum(length)

    # try to have even gaps between stations
    deviation = statistics.stdev(length)
    mean = statistics.mean(length)
    if deviation == 0.0:
        deviation = 0.01
    fitness = fitness + 1 / deviation

    # avoiding repetitions
    fitness = fitness + len(unique)
    return fitness,


def visualize():
    plt.plot(mean_fitness, label='Fitness')
    plt.xlabel('Generation')
    plt.ylabel('fitness')
    plt.legend()
    plt.show()


def main():
    random.seed(64)
    pop = toolbox.population(n=1000)
    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while g < 50:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        min_fitness.append(min(fits))
        max_fitness.append(max(fits))
        mean_fitness.append(mean)
        std_fitness.append(std)

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


if __name__ == "__main__":
    data = readFile()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 150)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 10)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitnessFunction)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
    toolbox.register("select", tools.selTournament, tournsize=3)

    main()
    visualize()
