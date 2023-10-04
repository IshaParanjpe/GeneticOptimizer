import random 

def func(x,y,z):
    return 6*x**3 + 9*y**2 + 90*z - 25

def fitness(x,y,z):
    ans = func(x,y,z)
    return abs(ans)

# Random generate 1000 possible solutions having 3 values
solutions = []
for i in range(1000):
    solutions.append((random.uniform(0, 10000), # x
                      random.uniform(0, 10000), # y
                      random.uniform(0, 10000))) # z

# iterate for some number of iterations/epochs: 10000
for i in range(10000):
    rankedSolutions = []
    # Calculate Fitness
    for s in solutions:
        rankedSolutions.append((fitness(s[0], s[1], s[2]), s))
    rankedSolutions.sort()
    print(f"------- Gen {i} Best Solutions -------")
    print(rankedSolutions[0])

    # stopping criterion
    if rankedSolutions[0][0] < 0.001:
        print(6*rankedSolutions[0][1][0]**3 + 9*rankedSolutions[0][1][1]**2 + 90*rankedSolutions[0][1][2])
        break
    
    # Selection
    # otherwise (i.e. if the stopping criteria is not satisfied)
    # select the top 100 solutions
    bestSolutions = rankedSolutions[:100]

    # s = (x,y,z)
    elements = []
    for s in bestSolutions:
        elements.append(s[1][0]) #x
        elements.append(s[1][1]) #y
        elements.append(s[1][2]) #z

    # crossover and mutation
    newGen = []
    for _ in range(1000):
        # x
        e1 = random.choice(elements) * random.uniform(0.99, 1.01)
        # y
        e2 = random.choice(elements) * random.uniform(0.99, 1.01)
        # z
        e3 = random.choice(elements) * random.uniform(0.99, 1.01)

        newGen.append((e1,e2,e3))

    solutions = newGen


    
