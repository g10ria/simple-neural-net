/**
 * Gloria Zhu
 * Created 9/8/2019
 * This file holds a collection of error functions to be used
 * for evaluating error in a neural network. Note that every 
 * function here should take in as parameters two arrays of
 * float values of the same length, the first being the
 * expected output and the second being the actual output,
 * as well as an integer that is the length of the two arrays.
 * 
 * Functions in this file:
 * 
 * quadraticLoss
 */

#include <stdlib.h>
#include <math.h>

#include "./headerfiles/errorFunctions.h"

/**
 * The quadratic loss error function returns half the sum of the squares of the
 * differences between expected outputs and actual outputs.
 * 
 * @param expectedOutput array of the expected outputs
 * @param actualOutput array of the actual outputs
 * @param arrayLength the number of outputs to compare and use to calculate error
 */
double quadraticLoss(double expectedOutput[], double actualOutput[], int arrayLength)
{
   double error = 0.0;
   for (int i = 0; i < arrayLength; i++)
   {
      double deviation = expectedOutput[0] - actualOutput[0];
      error += deviation * deviation;
   }
   return 0.5 * error;
}

