SHELL := /bin/bash
CC = gcc-7
MPICC = mpicc
MPIRUN = mpirun --oversubscribe -np 4

test_sequential:
	cd ../; make lib;
	cp ../lib/*.a ../inc/knnring.h ./;
	$(CC) tester.c knnring_sequential.a -o $@ -lm -lopenblas
	./test_sequential
	rm test_sequential


test_mpi:
	cd ..; make lib;
	cp ../lib/*.a ../inc/knnring.h ./;
	$(MPICC) tester_mpi.c knnring_mpi.a -o $@ -lm -lopenblas
	$(MPIRUN) ./test_mpi
	rm test_mpi
