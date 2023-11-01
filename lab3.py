import math
import numpy as np
from matplotlib import pyplot as plt
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
from aco import AntColony
import time


def endurance(arr):
    # x = arr[0]
    # y = arr[1]
    # z = arr[2]
    # u = arr[3]
    # v = arr[4]
    # w = arr[5]
    return -(math.exp(-2*(arr[1]-math.sin(arr[0]))**2) + math.sin(arr[2]*arr[3]) + math.cos(arr[4]*arr[5]))


def f(x):
    n_particles = x.shape[0]
    j = [endurance(x[i]) for i in range(n_particles)]
    return np.array(j)


def engineering_problem():
    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    x_min = np.zeros(6)
    x_max = np.ones(6)
    my_bounds = (x_min, x_max)

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(
        n_particles=10, dimensions=6, options=options, bounds=my_bounds)

    # Perform optimization
    cost, pos = optimizer.optimize(f, iters=1000)

    # Obtain cost history from optimizer instance
    cost_history = optimizer.cost_history

    plot_cost_history(cost_history)
    plt.show()


def plot_nodes(coords, w=12, h=8):
    for x, y in coords:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def TSP():

    plt.style.use("dark_background")

    # coords = np.random.randint(100, size=(15, 2))
    # coords = tuple(map(tuple, coords))

    COORDS = ((20, 52),
              (43, 50),
              (20, 84),
              (70, 65),
              (29, 90),
              (87, 83),
              (73, 23),
              (80, 63),
              (60, 49),
              (21, 23),
              (68, 32),
              (25, 70),
              (80, 90),
              (70, 27))

    plot_nodes(COORDS)

    start = time.time()
    

    ######################################################################################################################################
    # Początkowy czas: 22.771663904190063
    # Czas przy zmianie alfa: 22.83426833152771
    # Czas przy zmianie beta: 22.86630940437317
    # Czas przy zmianie ant_count: 45.022698640823364
    # Czas przy zmianie pheromone_evaporation_rate: 22.651716470718384 dla 0.8
    # Czas przy zmianie pheromone_constant: 22.470106840133667 dla 2000
    # Parametry alfa oraz beta nie mają większego wpływu na czas wykonywania algorytmu. Przy zmianie wartości beta z 1.2 na 2.0 oraz 
    # alfa z 0.5 na 1.0 różnica w czasie wykonania alogrytmu była minimalna.
    # Znaczna różnica pojawia sie przy zmianie parametru ant_count. Czas wykonania algorytmu przy dwukrotnym zwiększeniu tej wartości wzrósł o 97.71%. 
    # Inna sytuacja pojawia się przy zwiększeniu parametrów pheromone_evaporation_rate oraz pheromone_constant. Zmiana obydwu tych wartości powoduje 
    # minimalnie krótszy czas wykonania algorytmu. 
    # 
    # colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2,
    #    pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
    #    iterations=300)
    # 
    ######################################################################################################################################
    # 
    colony = AntColony(COORDS, ant_count=300, alpha=0.9, beta=1.6,
                       pheromone_evaporation_rate=0.45, pheromone_constant=1300.0,
                       iterations=700)
    stop = time.time()

    print(f"Exec time: {stop - start}")

    optimal_nodes = colony.get_path()

    for i in range(len(optimal_nodes) - 1):
        plt.plot(
            (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
            (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
        )

    plt.show()


if __name__ == '__main__':
    engineering_problem()
    TSP()
