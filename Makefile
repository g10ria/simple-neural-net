CC=gcc
CFLAGS=-I.
DEPS = network.c outputFunctions.c errorFunctions.c activationFunctions.c dibdump.c

%.o: %.c $(DEPS)
	$(CC) -o $@ $^ $(CFLAGS)

makenet: network.o outputFunctions.o errorFunctions.o activationFunctions.o dibdump.o