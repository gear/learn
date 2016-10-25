CC=g++
CFLAGS=-c -Wno-write-strings -Wall 

all: callcc

callcc: call.cpp edge_data.h edat_read.cpp costfun.cpp
	$(CC) -lcv -lhighgui -lcxcore call.cpp edat_read.cpp costfun.cpp graphcut/graph.cpp graphcut/maxflow.cpp itBM.cpp -o callcc












