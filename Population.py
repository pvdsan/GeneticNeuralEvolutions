import random
import math
from DNA import DNA

class Population:
    def __init__(self, m, species_num):
        self.generations = 0
        self.poplist = [DNA() for _ in range(species_num)]
        self.matingPool = []
        self.mutationRate = m
        self.best = 0
        self.bestSpecimen = DNA()
        self.finished = False

    def calcPopFitness(self):
        for i in range(len(self.poplist)):
            self.poplist[i].setNN()
            self.poplist[i].CreateSamples()
            self.poplist[i].calcFitness()

    def naturalSelection(self):
        self.matingPool = []

        for i in range(len(self.poplist)):
            n = math.ceil(self.poplist[i].fitness * 10)

            for j in range(n):
                self.matingPool.append(self.poplist[i])

    def generate(self):
        for i in range(len(self.poplist)):
            a = math.floor(random.random() * len(self.matingPool))
            b = math.floor(random.random() * len(self.matingPool))
            partnerA = self.matingPool[a]
            partnerB = self.matingPool[b]
            child = partnerA.crossover(partnerB)
            # child.mutate(self.mutationRate)
            self.poplist[i] = child

        self.generations += 1

    def getBest(self):
        return self.best

    def evaluate(self):
        genRecord = 0
        index = 0

        for i in range(len(self.poplist)):
            if self.poplist[i].fitness > genRecord:
                genRecord = self.poplist[i].fitness
                index = i

        self.best = genRecord
        self.bestSpecimen = self.poplist[index]

    def IsFinished(self):
        return self.finished

    def getGenerations(self):
        return self.generations

    def AvgFitness(self):
        total = 0

        for i in range(len(self.poplist)):
            total += self.poplist[i].fitness

        return total / len(self.poplist)

# Other functionality as needed
