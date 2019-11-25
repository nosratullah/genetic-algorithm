import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sp
import matplotlib.image as mpimg

image = mpimg.imread('bunny.png')

#plt.imshow(image);

#plt.imshow(squarization(image));
#plt.imshow(squarization(imaginary));

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

    return matrix

image = image.flatten()
image = initialization(image)
imaginary = chromosomeProcess(image)
fitnessValue = fitness(image,imaginary)
population = 20
generation = 10000

for k in range(generation):

    if ( k == 0):
        populationMat = np.zeros((len(image)+1,population))
        np.shape(populationMat)
        for i in range(population):
            tempMat = chromosomeProcess(image)
            for j in range(len(image)+1):
                if (j==(len(image))):
                    populationMat[j][i] = fitness(image,tempMat)
                else:
                    populationMat[j][i] = tempMat[j]

        #np.shape(populationMat)
    else:
        for i in range(population):
            tempMat = mutation(firstChild)
            for j in range(len(image)+1):

                if (j==(len(image))):
                    populationMat[j][i] = fitness(image,tempMat)
                else:
                    populationMat[j][i] = tempMat[j]

    fitnessList = np.zeros(population)
    for i in range(population):
        fitnessList[i] = populationMat[-1][i]

    #finding 2 best matching chromosome:
    firstMaxMAt = np.zeros(len(image))
    secondMaxMat = np.zeros(len(image))
    for j in range(2):

        result = np.where(fitnessList == np.amax(fitnessList))
        fitnessList[result[0][0]] = -1 *fitnessList[result[0][0]]
        if (j == 0):
            for i in range(len(image)):
                firstMaxMAt[i] = populationMat[i][result[0][0]]
        else:
            for i in range(len(image)):
                secondMaxMat[i] = populationMat[i][result[0][0]]

    # Crossover :
    firstChild = np.zeros(len(image))
    secondChild = np.zeros(len(image))
    for i in range(len(image)):

        if(i<len(image)/2.0):

            firstChild[i] = firstMaxMAt[i]
            secondChild[i] = secondMaxMat[i]
        else:
            firstChild[i] = secondMaxMat[i]
            secondChild[i] = firstChild[i]
    if (k%100 == 0):
        plt.imshow(squarization(firstChild))
        plt.savefig('/png/figure{}'.format(k))
        print(np.max(fitnessList))
