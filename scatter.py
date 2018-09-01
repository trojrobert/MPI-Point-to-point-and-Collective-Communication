from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("size = {}".format(size))
if rank == 0:
    variable_to_share = [(i + 1)**2 for i in range(size)]
else:
    variable_to_share = None
    
recv = comm.scatter(variable_to_share, root = 0)
print("process = {}, variable shared = {}".format(rank,recv))
