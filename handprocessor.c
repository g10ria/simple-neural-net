#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "./headerfiles/outputFunctions.h"     // importing output,
#include "./headerfiles/activationFunctions.h" // activation, and
#include "./headerfiles/errorFunctions.h"      // error functions

#include "./headerfiles/dibdump.h" // importing dibdump functions

#define MAX_FILE_NAME_LENGTH 2048        // max characters in a file name

#define NUM_BITMAPS_TO_PROCESS 5

int main()
{
   char * pelsOutputFile = "./inputs/bitmapinputs.txt";

   FILE *outFile = fopen(pelsOutputFile, "a");
   fprintf(outFile, "5\n");

   readBitmap("./hands/Gloria1.bmp", pelsOutputFile);
   fprintf(outFile, "1\n0\n0\n0\n0\n");
   readBitmap("./hands/Gloria2.bmp", pelsOutputFile);
   fprintf(outFile, "0\n1\n0\n0\n0\n");
   readBitmap("./hands/Gloria3.bmp", pelsOutputFile);
   fprintf(outFile, "0\n0\n1\n0\n0\n");
   readBitmap("./hands/Gloria4.bmp", pelsOutputFile);
   fprintf(outFile, "0\n0\n0\n1\n0\n");
   readBitmap("./hands/Gloria5.bmp", pelsOutputFile);
   fprintf(outFile, "0\n0\n0\n0\n1\n");
}