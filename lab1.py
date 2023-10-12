import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def prime(x):
    if x < 2:
        return True

    for i in np.arange(2, np.sqrt(x) + 1, 1):
        if x % i == 0:
            return False
    return True


def select_prime(arr):
    new_arr = []
    for i in range(len(arr)):
        if prime(arr[i]):
            new_arr.append(arr[i])
    return new_arr


def normalize_vec(x):
    z = ((x - min(x)) / (max(x) - (min(x))))
    return z


def standardize_vec(x):
    z = (x - np.average(x)) / np.std(x)
    return z


if __name__ == '__main__':

    # # Prime Nums

    # arr = [3, 6, 11, 25, 19]
    # print(f"Prime numbers: {select_prime(arr)}")

    # # Vectors

    # arr1 = np.array([3, 8, 9, 10, 12])
    # arr2 = np.array([8, 7, 7, 5, 6])

    # arr_sum = arr1 + arr2
    # arr_mul = arr1 * arr2

    # print(f"Vector Sum: {arr_sum}")
    # print(f"Vector Product: {arr_mul}")

    # dot_product = np.dot(arr1, arr2)
    # euclidan_dist = np.sqrt(np.power(arr1, 2) + np.power(arr2, 2))

    # print(f"Vector Dot Product: {dot_product}")

    # print(f"Euclidan Distance: {euclidan_dist}")

    # int_arr = np.random.randint(1, 100, 50)

    # print(f"Max value: {max(int_arr)}, Min value: {min(int_arr)}")
    # print(f"Average value: {np.average(int_arr)}")
    # print(f"Standard Deviation: {np.std(int_arr)}")

    # normalized_arr = normalize_vec(int_arr)

    # print(f"Before normaliztion: value {max(int_arr)}, index {np.argmax(int_arr)}")
    # print(f"After normaliztion: value {max(normalized_arr)}, index {np.argmax(normalized_arr)}")

    # standardized_arr = standardize_vec(int_arr)
    # print(f"Standardized vector: avg {np.average(standardized_arr)}, deviation {np.std(standardized_arr)}")

    # plot

    miasta = pd.read_csv("./miasta.csv")
    miasta.loc[len(miasta)] = [2010, 460, 555, 405]
    xi = list(range((len(miasta.values[:, 0]))))

    plt.plot(xi, miasta.values[:, 1], '-o', color='red', label='Gdansk')
    plt.plot(xi, miasta.values[:, 2], '-o', color='blue', label='Poznan')
    plt.plot(xi, miasta.values[:, 3], '-o', color='green', label='Szczecin')
    plt.xticks(xi, miasta.values[:, 0])
    
    plt.xlabel("Lata")
    plt.ylabel("Liczba ludnosci [w tys.]")
    plt.title("Ludnosc w miastach Polski")
    plt.legend()
    plt.show()
    # print(miasta.values[:, 1])