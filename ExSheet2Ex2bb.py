"""
Created on Wed Apr 25 05:19:18 2018
@author: John Robert
"""

from mpi4py import MPI
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import cv2

from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sb

comm = MPI.COMM_WORLD 
rank=comm.Get_rank()
size = comm.Get_size()
root = 0
#To read the image into a matrix
#Load a Color image  
imgColor = cv2.imread('images.jpg',-1)

# img = cv2.imread('images.jpg',0)   load a color image in grayscale

#Print the matrix of the image
#print(imgColor)


#Get the dimension of the matrix 
matrixColorSize = imgColor.shape
#print("The size of matrix for Color image : {} \n".format(matrixColorSize))
NoRowColor = imgColor.shape[0]
#print("The number of rows in the matrix is : {} \n".format(NoRowColor))

t_start = 0
#To find the frequencey of a number in an array 
def frequency(listofNumber):
    freqencyResult = {}
    for i in listofNumber:
        if i in freqencyResult:
                freqencyResult[i] = freqencyResult.get(i)+1
        else:
                freqencyResult[i] =1
    return freqencyResult

dataToComputeColor = []

dataToScatterColor = []

TotalRdata = []
TotalGdata = []
TotalBdata = []

totalFrequencyTable ={}
ProcessedFlattenData = []
if rank == 0:
    t_start = MPI.Wtime()
    # No. of rows each process will work on
    dataSize4Worker = int(Decimal(NoRowColor / (size)).quantize(Decimal('1.'), rounding= ROUND_HALF_UP))
    #print("Size of data for each process is {} \n".format(dataSize4Worker))
     
    
    #Dividing  the initial matrix among the workers and sending it to them for computation
    for i in range(0,size):
        
        startSlice = dataSize4Worker * i
        
        #Checking if it is the last worker
        #Give remaining data to the last worker, in case all slice don't have equal size
        if i == size:
            endSlice = NoRowColor
        else:
            endSlice = startSlice + dataSize4Worker  
        dataToComputeColor.append(imgColor[startSlice:endSlice, :,:])
        
#send sliced data to other workers for computation 
#print("DataToCompute of RGB image: \n {} \n".format(dataToComputeColor))
dataToScatterColor = comm.scatter(dataToComputeColor, root = 0)        

#Convert the matrix of the picture to 2 dimension 
#So we can have R G B on each row 
RGBdataToScatterColor = dataToScatterColor.reshape(-1, dataToScatterColor.shape[-1])
MatrixRGBdataToScatterColor = RGBdataToScatterColor.shape
#print("The size of matrix for Color image in RGB : {} \n".format(MatrixRGBdataToScatterColor))

#Red values for each pixel in the image 
Rvalue4Data = RGBdataToScatterColor[:,0]
#print("Rvalue4Data Red for Rank {}: \n {} \n".format(rank, Rvalue4Data))
#Red values for each pixel in the image 
Gvalue4Data = RGBdataToScatterColor[:,1]
#print("Gvalue4Data Green for Rank {}: \n {} \n".format(rank, Gvalue4Data))
#Red values for each pixel in the image 
Bvalue4Data = RGBdataToScatterColor[:,2]
#print("Bvalue4Data Blue for Rank {}: \n {} \n".format(rank, Bvalue4Data))

#Combine the R values in the data of each process 
TotalRdata.append(Rvalue4Data)
TotalRdata = np.ravel(TotalRdata)
#print("The size of matrix for Color image in R : {} \n".format(len(TotalRdata)))
#Combine the G values in the data of each process 
TotalGdata.append(Gvalue4Data)
TotalGdata = np.ravel(TotalGdata)
#print("The size of matrix for Color image in G : {} \n".format(len(TotalGdata)))
#Combine the B values in the data of each process 
TotalBdata.append(Bvalue4Data)
TotalBdata = np.ravel(TotalBdata)
#print("The size of matrix for Color image in B : {} \n".format(len(TotalBdata)))



#cv2.imshow('image',imgColor)
#To display the image for a particular number of seconds
#for a keyboard event 
#cv2.waitKey(0)
#TO close the windows of the images that were open
#cv2.destroyAllWindows()

if rank == 0:
    
    #Process frequency for B
    processfrequencyTableB = frequency(TotalBdata)
    #Process frequency for G
    processfrequencyTableG = frequency(TotalGdata)
    #Process frequency for R
    processfrequencyTableR = frequency(TotalRdata)
    

    plt.rcParams['figure.figsize'] = (15,15)
    sb.set_style ('whitegrid')

    # Draw the histogram for R 
#    plt.hist(TotalRdata,256,[0,256])
#    plt.title('Histogram for Colored picture for R',fontsize=18)
#    plt.xlabel('Values of 0 - 255', fontsize=18)
#    plt.ylabel('Frequency of occurrence of R', fontsize=18)
#    plt.legend()
#    plt.show()
    
     # Draw the histogram for G 
#    plt.hist(TotalGdata,256,[0,256])
#    plt.title('Histogram for Colored picture for G',fontsize=18)
#    plt.xlabel('Values of 0 - 255', fontsize=18)
#    plt.ylabel('Frequency of occurrence of G', fontsize=18)
#    plt.legend()
#    plt.show()
    
    # Draw the histogram for B 
#    plt.hist(TotalBdata,256,[0,256])
#    plt.title('Histogram for Colored picture for B',fontsize=18)
#    plt.xlabel('Values of 0 - 255', fontsize=18)
#    plt.ylabel('Frequency of occurrence of B', fontsize=18)
#    plt.legend()
#    plt.show()
    
    
    #Arrange the frequency distribution of R in asending other
    processfrequencyTableR = dict(sorted(processfrequencyTableR.items()))
    #print('The frequency distribution of R')
    print('value  Frequency')
    #for value, frequency in processfrequencyTableR.items():
        #print('{}   {}'.format(value, frequency))
        
    #Arrange the frequency distribution of G in asending other
    processfrequencyTableG = dict(sorted(processfrequencyTableG.items()))
    #print('The frequency distribution of G')
    #print('value  Frequency')
    #for value, frequency in processfrequencyTableG.items():
        #print('{}   {}'.format(value, frequency))
        
    #Arrange the frequency distribution of B in asending other
    processfrequencyTableB = dict(sorted(processfrequencyTableB.items()))
    #print('The frequency distribution of B')
    #print('value  Frequency')
    #for value, frequency in processfrequencyTableB.items():
        #print('{}   {}'.format(value, frequency))

if rank == 0:
    t_diff = MPI.Wtime() - t_start
    print("Total time sent {} \n".format(t_diff))



