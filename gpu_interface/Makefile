#  THIS IS THE MAKE FILE FOR OpenCL INTERFACE
#  
#  This is a custom library to make OpenCL calls
#  from fortran.
#
#  written by: Dr. Terence Musho
#
#
#  uses gcc c compiler
#
CC = gcc -fPIC -Wwrite-strings
OPT = -O3 #-pg
LIBS = -L/opt/AMDAPP/include -L/usr/lib -lOpenCL 
OBJECTS = clinterface.o clinterface.a
INCLUDES = -I/usr/local/include -I/opt/AMDAPP/include -I/opt/cuda/include
CLIBS =  -lm 
CFLAG = $(OPT) $(INCLUDES)

all:    ${OBJECTS}

clinterface.o: clinterface.cpp
	$(CC) $(CFLAG) -c clinterface.cpp

clinterface.a: clinterface.o
	ar $(ARFLAGS) $@ $?
	ln -s clinterface.a libclinterface.a

clean:
	rm $(PROGRAM) $(OBJECTS) *.a

