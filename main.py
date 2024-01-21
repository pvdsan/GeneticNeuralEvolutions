from Population import Population
from DNA import DNA

# Constants for the genetic algorithm
POPULATION_SIZE = 10
MUTATION_RATE = 0.01
MAX_GENERATIONS = 10


def main():
    # Initialize the population
    population = Population(MUTATION_RATE, POPULATION_SIZE)
    best_performer = DNA()

    # Run the genetic algorithm
    while not population.IsFinished() and population.getGenerations() < MAX_GENERATIONS:
        population.calcPopFitness()  # Calculate the fitness of the population
        avgFitness = population.AvgFitness()  # Calculate Average Fitness
        population.evaluate()  # Evaluate the current population

        bst = population.getBest()
    
        if population.bestSpecimen.fitness > best_performer.fitness:
            best_performer = population.bestSpecimen

        print("Generation:" + str(population.generations))
        print("Best Fitness: " + str(bst))
        print("Average Fitness:" + str(avgFitness))
        print("Best Training Accuracy:" + str(population.bestSpecimen.TrainingACC))
        print("Best Testing Accuracy:" + str(population.bestSpecimen.TestingACC))
        print("------------------------------------")

        population.naturalSelection()  # Perform natural selection
        population.generate()  # Generate a new population
       
     
    # After the loop, you can print out information about the best specimen
    print(f"Finished after {population.getGenerations()} generations")
    best = population.bestSpecimen
    print(f"Best Specimen Fitness: {best.fitness}")

if __name__ == "__main__":
    main()
