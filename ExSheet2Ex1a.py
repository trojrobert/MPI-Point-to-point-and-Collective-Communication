"""
Created on Mon Apr 23 09:36:26 2018
@author: John Robert
"""
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
processorName = MPI.Get_processor_name()
root = 0

t_start = 0
#Initailize the size of the array
N = 100

# the number of process must be greater than 1 
if size == 1:
    
    print("The number of Process is 1, Process 1 have no other process to send data to")
if size > 1:   
    t_start = MPI.Wtime()
    #if is the Process 1 - the root process
    if rank == 0:
        
        #Generating array with random numbers
        arrayA = np.random.randint(1,100, size=(1,N))
        print("Array A: \n {} \n".format(arrayA))
        
        #Send Array A to all process P-1
        for i in range(1,size):        
        
            #Sending array A to all process 
            comm.send(arrayA,dest=i)
         
        comm.Barrier()
        
    if rank != 0:
        #receive array A from root process
        recDataA = comm.recv(source= 0)
        #print("Data was received from the root process")
        comm.Barrier()
if rank == 0:
    t_diff = MPI.Wtime() - t_start
    print("Total time sent {} \n".format(t_diff))     