/**
 * Gloria Zhu
 * Created 9/7/2019
 * This file defines and runs a network.
 * 
 * Functions in this file:
 * 
 * void parseConfig(void)
 * void takeDimensionInputs(void)
 * void takeTrainingSetsInputs(void)
 * void initializeWeightsFromFile(void)
 * void initializeWeightsRandomly(double, double)
 * double randomNumber(double, double)
 * void writeWeightsToFile(void)
 * void writeOutputsToFile(void)
 * void calculateNumNodesAndWeights(void)
 * void freeMemory(void)
 * 
 * void printWeights(void)
 * void printNetworkConfiguration(void)
 * void runNetwork(void)
 * 
 * double calculateError(void)
 * void runForAllTrainingSets(void);
 * void trainForAllTrainingSets(void);
 * void train(int, double);
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> // need this library to get unique seed (current unix time) for rng

#include "./headerfiles/outputFunctions.h"     // importing output,
#include "./headerfiles/activationFunctions.h" // activation, and
#include "./headerfiles/errorFunctions.h"      // error functions

#include "./headerfiles/dibdump.h" // importing dibdump functions

#define MAX_FILE_NAME_LENGTH 2048        // max characters in a file name
#define UNSIGNED_INT_SCALER 4294967295.0 // used for scaling the pels to [0,1]

/**
 * This function pointer refers to the output function
 * to be used in determining a node's output according
 * to its activation level. Output functions are defined
 * in ./outputFunctions.c and included in this file.
 */
double (*outputFunction)(double value) = &sigmoid; // set the output function here

/**
 * This function pointer refers to the output derivative function
 * to be used in determining partial derivatives. They are defined
 * in ./outputFunctions.c and included in this file.
 */
double (*outputDerivFunction)(double value) = &sigmoidDeriv; // set the output derivative function here

/**
 * This function pointer refers to the activation function
 * to be used in calculating the activation of a unit according 
 * to input. Activation functions are 
 * defined in ./activationFunctions.c and included in this file.
 */
double (*activationFunction)(double input) = &identity; // set the activation function here

/**
 * This function pointer refers to the error function
 * to be used in calculating error. Error functions are
 * defined in ./errorFunctions.c and included in this file.
 */
double (*errorFunction)(
    double expectedOutput[],
    double actualOutput[],
    int numNodes) = &quadraticLoss; // set the error function here

// function headers ----------------------

// functions that handle utility tasks like i/o and mem allocation
void parseConfig(void);
void takeDimensionInputs(void);
void takeTrainingSetsInputs(void);
void initializeWeightsFromFile(void);
void initializeWeightsRandomly(double, double);
double randomNumber(double, double);
void writeWeightsToFile(void);
void writeOutputsToFile(void);
void calculateNumNodesAndWeights(void);
void freeMemory(void);

// functions for printing and debugging
void printWeights(void);
void printNetworkConfiguration(void);

// functions that run/train the network
void runNetwork(void);
double calculateError(void);
void runForAllTrainingSets(void);   // does not train
void trainForAllTrainingSets(void); // helper function
void train(int, double);

// variable declarations ----------------------

// variables that describe the structure of the network
char configFilename[MAX_FILE_NAME_LENGTH];

int numLayers;
int numHiddenLayers;
int numInputNodes;
int *layerDimensions;
int numOutputNodes;

char useRandomWeights;

// arrays that hold the actual values of the network
double *nodes;
double *weights;
double *expectedOutputs;

// backprop arrays
double *thetas;
double *psis;

// calculated values related to the structure of the network
int totalWeights;
int maxNodesInALayer;
int maxWeightsInALayer;

// file paths for i/o files
char weightsFileInput[MAX_FILE_NAME_LENGTH];
char weightsFileOutput[MAX_FILE_NAME_LENGTH];

char nodesFileInput[MAX_FILE_NAME_LENGTH];
char nodesFileOutput[MAX_FILE_NAME_LENGTH];
int numTrainingSets;

char useBitmap;
char bitmapFileInput[MAX_FILE_NAME_LENGTH];
char bitmapFileOutput[MAX_FILE_NAME_LENGTH];

// values related to training
char trainNetwork;          // whether or not to train (Y for yes, anything else for no)
char printNetworkSpecifics; // whether or not to print the specific values of the network
char printDebugMessages;    // whether or not to print debug messages
char enableWeightRollback;  // whether or not to enable weight rollback

double *trainingSets; // stores training set values

double error;                // current error of network (set to some initial config value)
double learningFactor;       // current lambda value
double learningFactorScaler; // lambda scaler

double minLearningFactor; // lower bound for lambda value
double maxLearningFactor; // upper bound for lambda value

int dumpEveryIterations; // dump weights/outputs every _x_ iterations

int maxIterations;  // max number of iterations before stopping
double targetError; // training stops when error reaches this value

/**
 * The main function makes the actual calls that complete parts
 * of the process of running a neural network.
 */
int main()
{
    printf("What config file should I use? ");
    scanf("%s", &configFilename);
    parseConfig();

    printf("\nINITIAL NETWORK:\n");
    runForAllTrainingSets();

    clock_t CPU_time_1 = clock();

    if (trainNetwork == 'Y')
    {
        printf("AFTER TRAINING:\n");
        train(maxIterations, targetError);
   }

   clock_t CPU_time_2 = clock();

   writeWeightsToFile();
   writeOutputsToFile();

   if (useBitmap == 'Y')
   {
      writeBitmap(nodesFileOutput, bitmapFileInput, bitmapFileOutput);
   }

   freeMemory();

   printf("Time taken: %fms", ((double)(CPU_time_2 - CPU_time_1)) / CLOCKS_PER_SEC * 1000);

   return 0;
}

/**
 * This function parses in all of the network's options through the
 * config file (see the README for details on the options).
 */
void parseConfig()
{
   FILE *config = fopen(configFilename, "r");

   char dummy[MAX_FILE_NAME_LENGTH]; // dummy value for handling the descriptor strings in the config file

   fscanf(config, "%s", &dummy);
   fscanf(config, "%d", &numInputNodes); // reading in number of input nodes

   fscanf(config, "%s", &dummy);
   fscanf(config, "%d", &numHiddenLayers); // reading in number of hidden layers

   fscanf(config, "%s", &dummy);
   fscanf(config, "%d", &numOutputNodes); // reading in number of output nodes

   numLayers = numHiddenLayers + 2; // setting some network structure values
   layerDimensions = calloc(numLayers, sizeof(int));
   layerDimensions[0] = numInputNodes;
   layerDimensions[numLayers - 1] = numOutputNodes;

   for (int i = 0; i < numHiddenLayers; i++) // taking in hidden layer dimensions
   {
      fscanf(config, "%s", &dummy);
      fscanf(config, "%d", layerDimensions + i + 1);
   }

   fscanf(config, "%s", &dummy);
   fscanf(config, "%s", &trainNetwork); // whether to train or just run instead

   fscanf(config, "%s", &dummy);
   fscanf(config, "%s", &printNetworkSpecifics); // whether or not to print network specifics

   fscanf(config, "%s", &dummy);
   fscanf(config, "%s", &printDebugMessages); // whether or not to print debug messages

   calculateNumNodesAndWeights(); // calculating some useful values

   nodes = malloc(maxNodesInALayer * numLayers * sizeof(double)); // allocating memory
   if (nodes == NULL)
   {
      printf("There was an error allocating memory for nodes.\n");
   }
   weights = malloc(maxWeightsInALayer * numLayers * sizeof(double));
   if (weights == NULL)
   {
      printf("There was an error allocating memory for weights.\n");
   }
   expectedOutputs = malloc(numOutputNodes * sizeof(double));
   if (expectedOutputs == NULL)
   {
      printf("There was an error allocating memory for expected outputs.\n");
   }

   thetas = malloc(maxNodesInALayer * numLayers * sizeof(double));
   if (thetas == NULL)
   {
      printf("There was an error allocating memory for thetas.\n");
   }
   psis = malloc(maxNodesInALayer * numLayers * sizeof(double));
   if (psis == NULL)
   {
      printf("There was an error allocating memory for psis.\n");
   }

   fscanf(config, "%s", &dummy);
   fscanf(config, "%s", &useBitmap); // whether or not to use bitmaps
   printf("use bitmap? %c\n", useBitmap);

   fscanf(config, "%s", &dummy);
   fscanf(config, "%s", &bitmapFileInput); // input bitmap file
   printf("bitmap input: %s\n", bitmapFileInput);

   fscanf(config, "%s", &dummy);
   fscanf(config, "%s", &bitmapFileOutput); // outputted bitmap file
   printf("bitmap output: %s\n", bitmapFileOutput);

   fscanf(config, "%s", &dummy);
   fscanf(config, "%s", nodesFileInput); // reading in training sets
   printf("nodes input: %s\n", nodesFileInput);

   if (useBitmap == 'Y')
   {
      readBitmap(bitmapFileInput, nodesFileInput);
   }

   takeTrainingSetsInputs();

   fscanf(config, "%s", &dummy);
   fscanf(config, "%s", nodesFileOutput); // where it would dump output values
   printf("nodes output: %s\n", nodesFileOutput);

   fscanf(config, "%s", &dummy);
   fscanf(config, "%s", &useRandomWeights); // whether or not to randomize weights
   printf("use random weights? %c\n", useRandomWeights);

   double randomWeightsLowerBound;
   double randomWeightsUpperBound;

   fscanf(config, "%s", &dummy);
   fscanf(config, "%lf", &randomWeightsLowerBound); // reading in randomized weights' lower bound

   fscanf(config, "%s", &dummy);
   fscanf(config, "%lf", &randomWeightsUpperBound); // reading in randomized weights' lower bound

   fscanf(config, "%s", &dummy);
   fscanf(config, "%s", &weightsFileInput); // where it would read preset weights from
   printf("weights input: %s\n", weightsFileInput);

   fscanf(config, "%s", &dummy);
   fscanf(config, "%s", &weightsFileOutput); // where it would dump weights to
   printf("weights output: %s\n", weightsFileOutput);

   fscanf(config, "%s", &dummy);
   fscanf(config, "%d", &dumpEveryIterations); // where it would dump weights to

   if (useRandomWeights == 'Y')
   {
      initializeWeightsRandomly(randomWeightsLowerBound, randomWeightsUpperBound);
   }
   else
   {
      initializeWeightsFromFile();
   }

   fscanf(config, "%s", &dummy);
   fscanf(config, "%lf", &learningFactor); // reading in initial learning factor
   printf("learning factor: %lf\n", learningFactor);

   fscanf(config, "%s", &dummy);
   fscanf(config, "%lf", &learningFactorScaler); // reading in learning factor scaler
   printf("learning factor scaler: %lf\n", learningFactorScaler);

   fscanf(config, "%s", &dummy);
   fscanf(config, "%lf", &minLearningFactor); // reading in minimum allowed learning factor

   fscanf(config, "%s", &dummy);
   fscanf(config, "%lf", &maxLearningFactor); // reading in maximum allowed learning factor

   fscanf(config, "%s", &dummy);
   fscanf(config, "%s", &enableWeightRollback); // whether or not to enable weight rollback

   fscanf(config, "%s", &dummy);
   fscanf(config, "%d", &maxIterations); // reading in max iterations for training

   fscanf(config, "%s", &dummy);
   fscanf(config, "%lf", &error); // reading in initial error

   fscanf(config, "%s", &dummy);
   fscanf(config, "%lf", &targetError); // reading in target error

   fclose(config);
}

/**
 * This function allocates space for all the training sets 
 * according to the number of training sets (first line of
 * the input file) and the number of input and output nodes 
 * (set in the config file). 
 * It then reads in the values and stores them.
 */
void takeTrainingSetsInputs()
{
   if (useBitmap == 'Y') // take input from a bitmap
   {
      unsigned int node = 0.0;
      FILE *nodesFile = fopen(nodesFileInput, "r");

      fscanf(nodesFile, "%x", &numTrainingSets);

      printf("num training sets: %d\n", numTrainingSets);

      trainingSets = calloc(numTrainingSets * (numInputNodes + numOutputNodes), sizeof(double));

      for (int i = 0; i < numTrainingSets * (numInputNodes + numOutputNodes); i++)
      {
         fscanf(nodesFile, "%x", &node);
         // printf("%u divided by %lf is %lf\n", (unsigned int) node, UNSIGNED_INT_SCALER, ((double)node) / UNSIGNED_INT_SCALER);
         *(trainingSets + i) = ((double)node) / UNSIGNED_INT_SCALER;
      }

      fclose(nodesFile);
   }
   else // take input from a pre-setup file
   {
      double node = 0.0;
      FILE *nodesFile = fopen(nodesFileInput, "r");

      fscanf(nodesFile, "%x", &numTrainingSets);

      trainingSets = calloc(numTrainingSets * (numInputNodes + numOutputNodes), sizeof(double));

      for (int i = 0; i < numTrainingSets * (numInputNodes + numOutputNodes); i++)
      {
         fscanf(nodesFile, "%lf", &node);
         *(trainingSets + i) = node;
      }

      fclose(nodesFile);
   }

   return;
}

/**
 * This function initializes the weights to known values from
 * a file. Weights are stored in mkj order.
 */
void initializeWeightsFromFile()
{
   double weight = 0.0;
   FILE *weightsFile = fopen(weightsFileInput, "r");

   for (int i = 0; i < totalWeights; i++)
   {
      fscanf(weightsFile, "%lf", &weight);
      *(weights + i) = weight;
   }

   fclose(weightsFile);

   return;
}

/**
 * Initializes all weights randomly to values between given bounds.
 * Randomization uses the current time as its seed.
 * 
 * @param lowerBound the lower bound of the randomized weights
 * @param upperBound the upper bound of the randomized weights
 */
void initializeWeightsRandomly(double lowerBound, double upperBound)
{
   srand(time(0));
   for (int m = 0; m < numLayers - 1; m++)
   {
      for (int j = 0; j < layerDimensions[m]; j++)
      {
         for (int k = 0; k < layerDimensions[m + 1]; k++)
         {
            unsigned int index = m * maxWeightsInALayer + j * maxNodesInALayer + k;
            double randWeight = randomNumber(lowerBound, upperBound);

            weights[index] = randWeight;
         }
      }
   }
   printf("Finished initializing weights\n");
   return;
}

/**
 * @return a random number between a given lower and upper bound
 * 
 * @param lowerBound the lower bound of the randomized weights
 * @param upperBound the upper bound of the randomized weights
 */ 
double randomNumber(double lowerBound, double upperBound) {
   return (double)rand() / (double)RAND_MAX * (upperBound - lowerBound) + lowerBound;
}

/**
 * This function write the current weights to a file.
 * Weights are stored in mkj order.
 */
void writeWeightsToFile()
{
   FILE *weightsFile = fopen(weightsFileOutput, "w+");

   for (int i = 0; i < totalWeights; i++)
   {
      fprintf(weightsFile, "%lf\n", weights[i]);
   }

   fclose(weightsFile);

   return;
}

/**
 * This function writes all the current outputs to a specified file.
 */
void writeOutputsToFile()
{
   FILE *outFile = fopen(nodesFileOutput, "w");

   for (int i = 0; i < numOutputNodes; i++)
   {
      fprintf(outFile, "%x\n", (unsigned int)(nodes[maxNodesInALayer * (numLayers - 1) + i] * UNSIGNED_INT_SCALER));
   }

   fclose(outFile);

   return;
}

/**
 * This function is responsible for calculating the maximum nodes
 * and maximum weights in a layer. These values are used for
 * allocating space.
 */
void calculateNumNodesAndWeights()
{
   for (int layer = 0; layer < numLayers; layer++)
   {
      // maintaining maxNodesInALayer
      if (layerDimensions[layer] > maxNodesInALayer)
      {
         maxNodesInALayer = layerDimensions[layer];
      }
   }

   maxWeightsInALayer = maxNodesInALayer * maxNodesInALayer;
   totalWeights = maxWeightsInALayer * (numLayers - 1);

   return;
}

/**
 * This function actually runs the network (which is assumed
 * to have already been initialized with inputs and weights).
 * It does not do any error calculation or training and merely
 * propagates values throughout the nodes, while collecting
 * theta values to be used in backprop.
 */
void runNetwork()
{
   for (int m = 0; m < numLayers - 1; m++) // looping through connectivity layers
   {
      int numSourceNodes = layerDimensions[m];
      int numDestNodes = layerDimensions[m + 1];

      for (int j = 0; j < numDestNodes; j++) // looping through right layer
      {
         int destNodeIndex = (m + 1) * maxNodesInALayer + j;
         thetas[destNodeIndex] = 0.0;

         for (int k = 0; k < numSourceNodes; k++) // looping through left layer
         {
            int sourceNodeIndex = m * maxNodesInALayer + k;
            int currentWeightIndex = m * maxNodesInALayer * maxNodesInALayer + k * maxNodesInALayer + j;

            thetas[destNodeIndex] += activationFunction(weights[currentWeightIndex] * nodes[sourceNodeIndex]);
         } // for (int k = 0; k < numSourceNodes; k++)

         nodes[destNodeIndex] = outputFunction(thetas[destNodeIndex]);
      } // for (int j = 0; j < numDestNodes; j++)
   }    // for (int m = 0; m < numLayers - 1; m++)

   return;
}

/**
 * This function calculates the error of a network (that
 * should already have been run) according to the
 * error function defined above.
 */
double calculateError()
{
   double *actualOutputs = calloc(numOutputNodes, sizeof(double));

   // storing actual outputs into an array to pass into the error function
   int nodeIndex = (numLayers - 1) * maxNodesInALayer;
   for (int i = 0; i < numOutputNodes; i++)
   {
      actualOutputs[i] = nodes[nodeIndex + i];
   }

   double error = errorFunction(expectedOutputs, actualOutputs, numOutputNodes);

   free(actualOutputs);

   return error;
}

/**
 * This function is responsible for freeing memory after running/training the network.
 */
void freeMemory()
{
   free(layerDimensions);
   free(nodes);
   free(weights);
   free(expectedOutputs);
   free(thetas);
   free(psis);

   return;
}

/**
 * This function prints the current
 * weights of the neural network.
 */
void printWeights()
{
   printf("\nNOW PRINTING WEIGHTS\n");
   for (int i = 0; i < totalWeights; i++)
   {
      printf("%lf\n", weights[i]);
   }

   return;
}

/**
 * Trains the network once for all training sets, using backprop,
 * then calculates the new error.
 * 
 * Adaptive learning can be disabled by setting the learning
 * factor scaler to 1.0 in the config. Weight rollback can 
 * also be enabled/disabled.
 */
void trainForAllTrainingSets()
{
   double *oldWeights;
   // only enable weight rollback if adaptive learning is enabled as well
   if (enableWeightRollback == 'Y' && learningFactorScaler != 1.0)
   {
      oldWeights = calloc(totalWeights, sizeof(double));
      for (int i = 0; i < totalWeights; i++)
      {
         oldWeights[i] = weights[i]; // storing old weights
      }
   }

   double errorSum = 0.0;
   int index = 0;
   for (int t = 0; t < numTrainingSets; t++) // train on every training set
   {
      for (int k = 0; k < numInputNodes; k++) // setting correct input values
      {
         nodes[k] = trainingSets[index];
         index++;
      }
      for (int k = 0; k < numOutputNodes; k++) // setting correct expected output values
      {
         expectedOutputs[k] = trainingSets[index];
         index++;
      }

      runNetwork();

      // collecting/applying psi values in the rightmost layer
      for (int j = layerDimensions[numLayers - 2] - 1; j >= 0; j--) // last hidden layer
      {
         int sourceNodeIndex = maxNodesInALayer * (numLayers - 2) + j;

         for (int i = layerDimensions[numLayers - 1] - 1; i >= 0; i--) // output layer
         {
            int destNodeIndex = maxNodesInALayer * (numLayers - 1) + i;

            int weightJIIndex = maxWeightsInALayer * (numLayers - 2) + maxNodesInALayer * j + i;

            double w = nodes[destNodeIndex] - expectedOutputs[i];
            double theta = thetas[maxNodesInALayer * (numLayers - 1) + i];
            double psiI = w * outputDerivFunction(theta);

            psis[destNodeIndex] = psiI;
            psis[maxNodesInALayer * (numLayers - 2) + j] += psiI * weights[weightJIIndex];

            /**
             * A -= is used here instead of a += like the documentation states
             * because when the weights are calculated, they are not multiplied
             * by the the extra -1 in the calculation formula. This avoids
             * unnecessarily flipping signs two times, saving time.
             */ 
            weights[weightJIIndex] -= learningFactor * nodes[sourceNodeIndex] * psiI;

         } // for (int i = layerDimensions[numLayers - 1] - 1; i >= 0; i--)

         double thetaJ = thetas[maxNodesInALayer * (numLayers - 2) + j];
         psis[maxNodesInALayer * (numLayers - 2) + j] *= outputDerivFunction(thetaJ);
      } // for (int j = layerDimensions[numLayers - 2] - 1; j >= 0; j--)

      // collecting/applying values in the non-rightmost layers
      for (int m = numLayers - 3; m >= 0; m--) // looping backwards through connectivity layers
      {
         int numSourceNodes = layerDimensions[m];
         int numDestNodes = layerDimensions[m + 1];

         for (int j = numDestNodes - 1; j >= 0; j--) // looping through right layer
         {
            int destNodeIndex = maxNodesInALayer * (m + 1) + j;

            for (int k = numSourceNodes - 1; k >= 0; k--) // looping through left layer
            {
               int sourceNodeIndex = maxNodesInALayer * m + k;

               double psiJ = psis[destNodeIndex];
               int weightKJIndex = maxWeightsInALayer * m + maxNodesInALayer * k + j;

               weights[weightKJIndex] -= learningFactor * nodes[sourceNodeIndex] * psiJ;

            } // for (int k = numSourceNodes - 1; k >= 0; k--)
         }    // for (int j = numDestNodes - 1; j >= 0; j--)
      }       // for (int m = numLayers - 3; m >= 0; m--)

      double err = calculateError();

      errorSum += err * err;
   }          // for (int t = 0; t < numTrainingSets; t++)

   double newError = 0.5 * errorSum; // multiply by 0.5 according to the error function

   if (learningFactorScaler != 1.0) // enable adaptive learning
   {
      if (newError > error && learningFactor > minLearningFactor) // error went up and the learning factor has room to decrease
      {
         learningFactor /= learningFactorScaler;

         if (enableWeightRollback == 'Y')
         {
            for (int i = 0; i < totalWeights; i++)
            {
               weights[i] = oldWeights[i];
            }
            free(oldWeights);
         }
      }
      else if (newError < error) // error went down
      {
         error = newError;
         learningFactor *= learningFactorScaler;
      }

      if (learningFactor > maxLearningFactor)
         learningFactor = maxLearningFactor; // capping the learning factor
   }
   else // adaptive learning is disabled
   {
      error = newError;
   }

   return;
}

/**
 * This runs the network for all the training sets and prints
 * out the input nodes, output nodes, expected output nodes, and error.
 * It also prints out the total error over all training sets.
 * No training is done.
 */
void runForAllTrainingSets()
{
   int index = 0;
   double errorSum = 0.0;

   for (int i = 0; i < numTrainingSets; i++) // looping through all training sets
   {
      for (int k = 0; k < numInputNodes; k++)
      {

         if (printNetworkSpecifics == 'Y') // for debugging
         {
            printf("%lf ", trainingSets[index]);
         }

         nodes[k] = trainingSets[index];
         index++;
      }
      if (printNetworkSpecifics == 'Y')
      {
         printf(" --> (expected ");
      }

      for (int k = 0; k < numOutputNodes; k++)
      {
         
         if (printNetworkSpecifics == 'Y') // for debugging
         {
            printf(" %lf", trainingSets[index]);
         }

         expectedOutputs[k] = trainingSets[index];
         index++;
      }

      runNetwork();
      double err = calculateError();

      if (printNetworkSpecifics == 'Y') // for debugging
      {
         printf(") actual: ");
         for (int k = 0; k < numOutputNodes; k++)
         {
            printf(" %lf", nodes[maxNodesInALayer * (numLayers - 1) + k]);
         }
         printf("\n");
      }

      errorSum += err * err;
   }

   error = errorSum * 0.5; // multiply by 0.5 according to the error function

   printf("Total error: %.16lf\n\n", error);

   return;
}

/**
 * Trains the network.
 * 
 * @param numTimes the amount of times to train the network
 * @param targetError the error at which to stop training (if reached)
 */
void train(int numTimes, double targetError)
{
   int cycles = 0;
   while (cycles < numTimes && error > targetError)
   {
      trainForAllTrainingSets();
      cycles++;

      if (printDebugMessages == 'Y')
      {
         printf("DEBUG: iteration %d, error: %.16lf, lambda: %lf\n", cycles, error, learningFactor);
      }

      if (cycles % dumpEveryIterations == 0) // dumps values every _x_ iterations
      {
         writeWeightsToFile();
         writeOutputsToFile();
      }
   }

   runForAllTrainingSets();

   printf("lambda: %lf\n", learningFactor);
   printf("Stopped after %d cycles (max %d cycles)\n", cycles, numTimes);
   printf("Current error: %.16lf\n", error);

   // printing termination conditions that were or were not met
   if (cycles == numTimes - 1)
      printf("Stopped due to cycle amount\n");
   if (error <= targetError)
      printf("Stopped due to sufficiently low error (%.16lf < %.16lf)\n", error, targetError);
   else
      printf("Did not reach specified error successfully (%.16lf > %.16lf)\n", error, targetError);

   return;
}