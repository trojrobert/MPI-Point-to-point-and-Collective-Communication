# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:10:30 2018
@author: John Robert
"""

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
processorName = MPI.Get_processor_name()
root = 0

#Initailize the size of the array
N = 100

#Generating array with random numbers
arrayA = np.random.randint(1,100, size=(1,N))
print("Array A: \n {} \n".format(arrayA))

#Calculating the destination rank for each process
def destA(rank):
    return  (2 *rank) + 1

t_start = 0
#Since only rank 0 send data without receiving data 
if rank == 0:
    t_start = MPI.Wtime()
    #Checking if the rank to receive the data exist before sending it 
    if destA(rank) < size:
        comm.send(arrayA,dest=1) 
        print("Sending to rank: {} \n".format(destA(rank)))
    if  (destA(rank) + 1) < size :
        comm.send(arrayA,dest=2) 
        print("Sending to rank: {} \n".format(destA(rank) + 1))
    comm.barrier()
   
if rank != 0:
    
    #Calculate the rank to receive the data
    recvProc = int((rank-1)/2)
    
    #Receiving the data from the sender 
    recvArrayA= comm.recv(source = recvProc)
    print("Receiving from rank: {} \n".format(recvProc))
     #Checking if the rank to receive the data exist before sending it  
    if destA(rank) < size:
        comm.send(arrayA,dest=destA(rank)) 
        print("Sending to rank: {} \n".format(destA(rank)))
    if (destA(rank) + 1) < size:
        comm.send(arrayA,dest=(destA(rank) +1))
        print("Sending to rank: {} \n".format(destA(rank) + 1))
    comm.barrier()
if rank == 0:
    t_diff = MPI.Wtime() - t_start
    print("Total time sent {} \n".format(t_diff))