import math as m
import numpy as np
import matplotlib.pyplot as plt


def f(i, alpha):
    return alpha ** i


def main():
    t = float(input("Enter t: "))
    last_x0 = 0
    fuck = np.arange(0, 30, 0.005)
    fig, ax = plt.subplots()
    s = 0
    root = [[0], [0]]
    alpha = float(input("Enter alpha: "))

    for i in range(7):
        for el in fuck:
            if np.round(np.absolute(np.cos(el + s) - t), decimals=2) == 0 and el > last_x0:
                if int(el) != int(root[0][-1]):
                    root[0].append(el)
                    root[1].append(fuck.tolist().index(el)) # Optimize this

        first = fuck[root[1][1]:root[1][2]]
        last_x0 = root[0][2]
        specific_const = np.absolute(f(i, alpha) * np.cos(first[0] + s) - t)

        ax.plot(first, np.absolute(f(i, alpha) * np.cos(first + s) - t) - specific_const, 'o', color="black", markersize=2)

        if (t / f(i, alpha) > 1):
            break
        else:
            s = np.arccos(t / f(i, alpha)) - root[0][2]
        
        root = [[0], [0]]

    ax.set_xlabel('$228$')
    ax.set_ylabel('$1488$')
    ax.grid()
    plt.show()


if __name__ == '__main__':
    main()
