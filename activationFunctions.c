/**
 * Gloria Zhu
 * Created 9/8/2019
 * This file holds a collection of activation functions to be used.
 * 
 * Functions in this file:
 * 
 * identity
 */

#include <stdlib.h>
#include <math.h>

#include "./headerfiles/activationFunctions.h"

/**
 * The identity function just returns the exact input.
 */
double identity(double input)
{
   return input;
}