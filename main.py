import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sp
import matplotlib.image as mpimg

def squarization(matrix):

    squar = int(np.sqrt(len(matrix)))
    matrix = matrix.reshape((squar,squar))

    return matrix

def initialization(image):

    for i in range(len(image)):
        if (image[i] == 0):
            image[i] = -1
        else:
            image[i] = +1

    return image

def chromosomeProcess(matrix):

    chromosome = np.zeros(len(matrix))
    spectrum = [-1,1]
    for i in range(len(matrix)):
        chromosome[i] = np.random.choice(spectrum)

    return chromosome

def fitness(source, test):

    score = 0
    for i in range(len(source)):
        if ((source[i] - test[i]) == 0):
            score += 1

    score = int(score/np.sqrt(len(source)))
    return score

def mutation(matrix):

    for i in range(len(matrix)):

        if (np.random.normal(0.0, 1) >= 0.1):

            matrix[i] = -1 * matrix[i]


#image = mpimg.imread('')
