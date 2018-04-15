GSLINC = /opt/local/include
GSLLIB = /opt/local/lib

emu.exe: emu.c params.h
	gcc -g -L$(GSLLIB) -I$(GSLINC) emu.c -lgsl -lgslcblas -lm -o emu.exe

clean:
	/bin/rm -f *.o

