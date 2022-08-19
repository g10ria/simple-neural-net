/**
 * Gloria Zhu
 * Created 11/15/2019
 * This file contains the header files for dibdump-related functions. 
 * More specific documentation can be found in the source file.
 */

#ifndef dibdump_h
#define dibdump_h

#include <windows.h>

int *readBitmap(char *, char *);
void writePelsToTextFile(unsigned int *, int, char *);
void writeBitmap(char *, char *, char *);
void writeBitmapHelper(char *, unsigned int *, int, BITMAPFILEHEADER, BITMAPINFOHEADER);

#endif