/**
 * Gloria Zhu
 * Created 9/8/2019
 * This file holds a collection of output functions as well as their derivatives
 * to be used in a neural network. Note that every function here should
 * take in as a parameter a single float value, which it will
 * return the corresponding output or derivative for.
 * 
 * Functions in this file:
 * 
 * sigmoid & sigmoidDeriv
 * tanh & tanhDeriv
 * relu & reluDeriv
 */

#include "./headerfiles/outputFunctions.h"

#include <stdlib.h>
#include <math.h>

/**
 * The sigmoid function is defined as:
 * sigmoid(x) = 1/(1+e^-x)
 * It returns values close to 1 for large
 * values of x and values close to 
 * 0 for small values of x.
 */
double sigmoid(double value)
{
   return 1.0 / (1.0 + exp(-value));
}

/**
 * This function returns the derivative of the sigmoid function,
 * which is sigmoid(x) * (1-sigmoid(x))
 */ 
double sigmoidDeriv(double value)
{
   double sig = sigmoid(value);
   return sig * (1.0 - sig);
}

/**
 * The hyperbolic tangent function is defined as:
 * tanh(x) = sinh(x)/cosh(x)
 * where sinh(x) = ( e^x - e^-x )/2
 * and cosh(x) = ( e^x + e^-x )/2
 * It returns values close to 1 for large
 * values of x and values close to
 * -1 for small values of x.
 */
double tanh(double value)
{
   return tanh(value);
}

/**
 * This function returns the derivative of the tanh function,
 * which is sech(x)^2, or 1/cosh(x)^2
 */ 
double tanhDeriv(double value)
{
   return 1.0 / (cosh(value) * cosh(value));
}

/**
 * The ReLU (rectified linear activation unit) function is defined as:
 * relu(x) =   x if x>0
 *             0 if x<=0
 * It returns the value itself if the value is positive
 * and zero if the value is negative.
 */
double relu(double value)
{
   return fmax(0.0, value);
}

/**
 * This function returns the derivative of the relu function,
 * which is 0 if the value is negative and 1 if the value is positive.
 * 
 * Note that the derivative does not exist at x=0 since the left/right
 * derivatives are different; in this case, the function merely returns 1.
 * 
 * Please excuse the use of two return statements; this seemed like the most 
 * efficient way to write this function.
 */ 
double reluDeriv(double value)
{
   if (value>=0.0)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
   
}