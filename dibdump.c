/**
 * Gloria Zhu / Adapted from Dr. Nelson
 * Created 11/23/2019
 * This file is responsible for reading in 24-bit and 32-bit bitmaps
 * and converting them to activation values. It is also responsible 
 * for reconstructing bitmaps from a text file of values.
 * 
 * Functions in this file:
 * int* readBitmap(char* inFileName, char* pelsOutputFile)
 * void writePelsToTextFile(unsigned int *pels, int numPels, char *pelsOutputFile)
 * void writeBitmap(char *pelsOutputFile, char *originalDIBFile, char *outputDIBFile)
 * void writeBitmapHelper(char *outFileName, unsigned int *pels, int numPels, BITMAPFILEHEADER bmpFileHeader, BITMAPINFOHEADER bmpInfoHeader)
{
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "./headerfiles/dibdump.h"

/**
 * This function reads in a bitmap and writes out its pels to a text file.
 * Note: this will only run for 24-bit and 32-bit bitmaps (that don't have a color table)
 * 
 * @param inFileName the name of the input bitmap file
 * @param pelsOutputFile the file to output the pel values to
 * @return a pointer to the array of pels (in blue|green|red|reserved byte form)
 */
int *readBitmap(char *inFileName, char *pelsOutputFile)
{
   BITMAPFILEHEADER bmpFileHeader;
   BITMAPINFOHEADER bmpInfoHeader;

   char topDownDIB = 'n';

   unsigned int *pels; // using unsigned int to prevent overflow errors

   FILE *inFile = fopen(inFileName, "rb");

   if (inFile != NULL)
   {
      fread((char *)&bmpFileHeader, sizeof(BITMAPFILEHEADER), 1, inFile);

      printf("bfType=%2X\nbfSize=%lu\nbfReserved1=%u\nbfReserved2=%u\nbfOffBits=%lu\n",
             bmpFileHeader.bfType,
             bmpFileHeader.bfSize,
             bmpFileHeader.bfReserved1,
             bmpFileHeader.bfReserved2,
             bmpFileHeader.bfOffBits);

      fread((char *)&bmpInfoHeader, sizeof(BITMAPINFOHEADER), 1, inFile);

      printf("biSize=%lu\nbiWidth=%ld\nbiHeight=%ld\nbiPlanes=%u\nbiBitCount=%u\nbiCompression=%lu\nbiSizeImage=%lu\nbiXPelsPerMeter=%ld\nbiYPelsPerMeter=%ld\nbiClrUsed=%lu\nbiClrImportant=%lu\n\n",
             bmpInfoHeader.biSize,
             bmpInfoHeader.biWidth,
             bmpInfoHeader.biHeight,
             bmpInfoHeader.biPlanes,
             bmpInfoHeader.biBitCount,
             bmpInfoHeader.biCompression,
             bmpInfoHeader.biSizeImage,
             bmpInfoHeader.biXPelsPerMeter,
             bmpInfoHeader.biYPelsPerMeter,
             bmpInfoHeader.biClrUsed,
             bmpInfoHeader.biClrImportant);

      int biHeightAbs = bmpInfoHeader.biHeight;

      if (bmpInfoHeader.biHeight < 0)
      {
         topDownDIB = 'Y';
         biHeightAbs = -biHeightAbs;
      }

      // allocating space for the pels
      pels = malloc(bmpInfoHeader.biWidth * biHeightAbs * sizeof(int));

      int bytesPerPixel = bmpInfoHeader.biBitCount / 8;

      if (bytesPerPixel == 4) // 32-bit bitmap
      {
         fread(pels, bytesPerPixel, bmpInfoHeader.biWidth * biHeightAbs, inFile);

         for (int i = 0; i < bmpInfoHeader.biWidth * biHeightAbs; i++)
         {
            unsigned int pel = pels[i];

            // pels are read in reserved|red|green|blue order
            int blue = pel & 0x00FF;
            int green = (pel >> 8) & 0x00FF;
            int red = (pel >> 16) & 0x00FF;
            int reserved = (pel >> 24) & 0x00FF;
            // rearranging pels into blue|green|red|reserved order
            unsigned int newpel = blue << 24 | green << 16 | red << 8 | reserved;

            pels[i] = newpel;
         }

         writePelsToTextFile(pels, bmpInfoHeader.biWidth * biHeightAbs, pelsOutputFile);
      }
      else if (bytesPerPixel == 3) // 24-bit bitmap
      {
         fread(pels, bytesPerPixel, bmpInfoHeader.biWidth * biHeightAbs, inFile);

         writePelsToTextFile(pels, bmpInfoHeader.biWidth * biHeightAbs, pelsOutputFile);
      }
      else // if the bitmap is not 24 or 32 bit (bytesPerPixel is not 3 or 4)
      {
         fprintf(stderr, "ERROR: Bitmap must be 24-bit or 32-bit.");
      }
   }
   else // if inFile == NULL
   {
      fprintf(stderr, "INPUT ERROR: %s\n", strerror(errno));
   }

   return pels;
}

/**
 * This function writes an array of pels to a text file in hexadecimal form.
 * It concatenates a 1 to the front and two copies of the pels to fit
 * the input txt file format of the network.
 * 
 * @param pels array of the pel values
 * @param numPels the total number of pels
 * @param pelsOutputFile the file to write the pels to
 */
void writePelsToTextFile(unsigned int *pels, int numPels, char *pelsOutputFile)
{
   printf("Writing pels to text file...\n");
   FILE *outFile = fopen(pelsOutputFile, "a+b");

   // fprintf(outFile, "1\n"); uncomment me for original code submission

   for (int i = 0; i < numPels; i++)
   {
      fprintf(outFile, "%x\n", pels[i]);
   }

   for (int i = 0; i < numPels; i++)
   {
      fprintf(outFile, "%x\n", pels[i]);
   }

   fclose(outFile);

   printf("Finished writing %d pels to text file %s\n", numPels, pelsOutputFile);

   return;
}

/**
 * This function transfers from pel .txt file to a bitmap, using
 * the original bitmap file as a reference. It is responsible for
 * reading in the file/info headers of the original DIB file, and
 * then passing those values + the pels to a helper function.
 * 
 * @param pelsOutputFile the file to read pels in from
 * @param originalDIBFile the original bitmap file to reference for header/info values
 * @param outputDIBFile the bitmap file to output to
 */
void writeBitmap(char *pelsOutputFile, char *originalDIBFile, char *outputDIBFile)
{
   FILE *origFile = fopen(originalDIBFile, "rb");
   FILE *pelsFile = fopen(pelsOutputFile, "r");

   unsigned int *pels;

   if (origFile != NULL)
   {
      BITMAPFILEHEADER bmpFileHeader;
      BITMAPINFOHEADER bmpInfoHeader;

      fread((char *)&bmpFileHeader, sizeof(BITMAPFILEHEADER), 1, origFile);
      fread((char *)&bmpInfoHeader, sizeof(BITMAPINFOHEADER), 1, origFile);

      int biHeightAbs = bmpInfoHeader.biHeight;
      char topDownDIB = 'n';

      if (bmpInfoHeader.biHeight < 0)
      {
         topDownDIB = 'Y';
         biHeightAbs = -biHeightAbs;
      }

      pels = malloc(bmpInfoHeader.biWidth * biHeightAbs * sizeof(int));

      unsigned int pel;

      for (int i = 0; i < bmpInfoHeader.biWidth * biHeightAbs; i++)
      {
         fscanf(pelsFile, "%x", &pel);
         pels[i] = pel;
      }

      writeBitmapHelper(outputDIBFile, pels, bmpInfoHeader.biWidth * biHeightAbs, bmpFileHeader, bmpInfoHeader);

      fclose(origFile);
      fclose(pelsFile);

      printf("Finished writing pels from %s to bitmap file %s based on original bitmap %s\n",
             pelsOutputFile,
             outputDIBFile,
             originalDIBFile);

   } // if (origFile == NULL)

   return;
}

/**
 * This function transfers from pel .txt file to a bitmap, having been
 * given the file/info headers of the original bitmap file.
 * 
 * @param outFileName the name of the output dibdump file
 * @param pels the array of pels
 * @param numPels the total number of pels
 * @param bmpFileHeader the bitmap file header to use
 * @param bmpInfoHeader the bitmap info header to use
 */
void writeBitmapHelper(
    char *outFileName,
    unsigned int *pels,
    int numPels,
    BITMAPFILEHEADER bmpFileHeader,
    BITMAPINFOHEADER bmpInfoHeader)
{

   FILE *outFile = fopen(outFileName, "wb");

   unsigned int pel;
   for (int i = 0; i < numPels; i++)
   {
      pel = pels[i];
      int reserved = pel & 0x00FF;
      int red = (pel >> 8) & 0x00FF;
      int green = (pel >> 16) & 0x00FF;
      int blue = (pel >> 24) & 0x00FF;
      // rearranging the pels back into their proper structure
      unsigned int newpel = reserved << 24 | red << 16 | green << 8 | blue;

      pels[i] = newpel;
   }

   if (outFile != NULL)
   {
      fwrite(&bmpFileHeader, sizeof(BITMAPFILEHEADER), 1, outFile);
      fwrite(&bmpInfoHeader, sizeof(BITMAPINFOHEADER), 1, outFile);
      fwrite(pels, 4, numPels, outFile);
      fclose(outFile);

      printf("Finished writing %d pels to bitmap output %s\n", numPels, outFileName);
   }
   else
   {
      fprintf(stderr, "OUTPUT ERROR: %s\n", strerror(errno));
   }

   return;
}
