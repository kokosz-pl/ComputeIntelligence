import pygad
import numpy as np
import time
import math


def knapsack_problem():

    def fitness_func(ga_instance, solution, solution_idx):
        sum1 = np.sum(solution * value)
        sum2 = np.sum(solution * weight)
        if sum2 <= 25:
            fitness = sum1
        else:
            fitness = 0
        return fitness

    value = [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300]
    weight = [7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15]
    gene_space = [0, 1]
    fitness_function = fitness_func
    sol_per_pop = 10
    num_genes = len(value)
    num_parents_mating = 5
    num_generations = 30
    keep_parents = 2
    parent_selection_type = "sss"
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 8

    ga_instance = pygad.GA(gene_space=gene_space,
                           num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           stop_criteria=["reach_1600"]
                           )

    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Parameters of the best solution : {solution}")

    prediction = np.sum(value*solution)
    print(f"Predicted output based on the best solution : {prediction}")
    print(f"Number of generations: {ga_instance.generations_completed}\n")
    # ga_instance.plot_fitness()

    time_sum = 0
    for i in range(10):
        start = time.time()
        ga_instance = pygad.GA(gene_space=gene_space,
                               num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_function,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes,
                               stop_criteria=["reach_1600"]
                               )

        ga_instance.run()
        stop = time.time()
        print(f"No.{i}: {(stop - start) * 1000} ms")
        time_sum += stop - start

    print(f"Avg time for solution: {(time_sum / 10) * 1000} ms")


def endurance(x, y, z, u, v, w):
    return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w)


def engineering_problem():

    def fit_func(ga_instance, solution, solution_idx):
        return endurance(*solution)

    gene_space = {"low": 0, "high": 1}
    fitness_function = fit_func
    sol_per_pop = 30
    num_genes = 6
    num_parents_mating = 5
    num_generations = 50
    keep_parents = 2
    parent_selection_type = "sss"
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 20

    ga_instance = pygad.GA(gene_space=gene_space,
                           num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes)

    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    np.set_printoptions(precision=6)
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Parameters of the best solution : {solution}")

    prediction = solution
    print(f"Predicted output based on the best solution : {prediction}")
    print(f"Number of generations: {ga_instance.generations_completed}\n")
    # ga_instance.plot_fitness()


def maze_problem():

    maze = np.array([[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
                     [9, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 9],
                     [9, 9, 9, 0, 0, 0, 9, 0, 9, 9, 0, 9],
                     [9, 0, 0, 0, 9, 0, 9, 0, 0, 0, 0, 9],
                     [9, 0, 9, 0, 9, 9, 0, 0, 9, 9, 0, 9],
                     [9, 0, 0, 9, 9, 0, 0, 0, 9, 0, 0, 9],
                     [9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 9, 9],
                     [9, 0, 9, 0, 0, 9, 9, 0, 9, 0, 0, 9],
                     [9, 0, 9, 9, 9, 0, 0, 0, 9, 9, 0, 9],
                     [9, 0, 9, 0, 9, 9, 0, 9, 0, 9, 0, 9],
                     [9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 9],
                     [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]])

    def fitness_func(ga_instance, solution, solution_idx):
        x, y = 1, 1
        x_end, y_end = 10, 10
        steps = 0
        for move in solution:
            if move == 1:    # left move
                x -= 1
            elif move == 2:  # right move
                x += 1
            elif move == 3:  # up move
                y -= 1
            elif move == 4:  # down move
                y += 1

            steps += 1
            if (x, y) == (x_end, y_end):
                return (1 / steps) + 1
            elif maze[y, x] == 9:
                return 1 / (abs(x - x_end) + abs(y - y_end))
        return (1 / (abs(x - x_end) + abs(y - y_end))) / 10

    gene_space = [1, 2, 3, 4]
    fitness_function = fitness_func
    sol_per_pop = 150
    num_genes = 30
    num_parents_mating = 5
    num_generations = 250
    keep_parents = 2
    parent_selection_type = "sss"
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 0.1

    ga_instance = pygad.GA(gene_space=gene_space,
                           num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           stop_criteria=["reach_1"]
                           )

    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Parameters of the best solution : {solution}")

    prediction = solution
    print(f"Predicted output based on the best solution : {prediction}")
    print(f"Number of generations: {ga_instance.generations_completed}\n")

    # ga_instance.plot_fitness()
    print_solution(maze, solution)

    time_sum = 0
    for i in range(10):
        start = time.time()
        ga_instance = pygad.GA(gene_space=gene_space,
                               num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_function,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes,
                               stop_criteria=["reach_1"]
                               )

        ga_instance.run()
        stop = time.time()
        print(f"No.{i}: {(stop - start) * 1000} ms")
        time_sum += stop - start

    print(f"Avg time for solution: {(time_sum / 10) * 1000} ms")


def print_solution(maze, sol):

    x, y = 1, 1
    maze[y, x] = 1

    for num in sol:
        if num == 1:    # left move
            x -= 1
        elif num == 2:  # right move
            x += 1
        elif num == 3:  # up move
            y -= 1
        elif num == 4:  # down move
            y += 1

        if maze[y, x] == 0:
            maze[y, x] = 1
        elif maze[y, x] == 9:
            break

    print(f"\nSolution path in the maze:\n{maze}")


if __name__ == '__main__':
    knapsack_problem()
    engineering_problem()
    maze_problem()
