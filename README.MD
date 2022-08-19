# Project Info
This is a set of files that runs a basic neural network. 
The main file is `network.c`, and the other files 
provide miscellaneous utilities:  
   `outputFunctions.c` - stores output functions for use in the network  
   `activationFunctions.c` - stores activation functions for use in the network  
   `errorFunctions.c` - stores error functions for use in the network  
   `dibdump.c` - stores utility functions for use with bitmap i/o  
  
Configuration values should be set in a .txt file whose path
is specified during runtime.

# Running the network

   ```
   $ gcc -o network network.c outputFunctions.c errorFunctions.c activationFunctions.c dibdump.c
   $ network.exe
   ```
to compile and run the network; enter the path to the config when prompted.

# Config Structure

```
num_input_nodes            2
num_hidden_layers          1
num_output_nodes           3

hidden_layer_1_size        10                   // add lines below this for more hidden layers

trainNetwork               Y                    // whether to train or just run
print_network_specifics    Y                    // whether to print the specific values of the network
print_debug_messages       Y                    // whether to print debug messages while running/training

use_bitmap                 Y                    // whether or not to train on a bitmap
original_bitmap_file       ./input.bmp          // if so, input bitmap file
output_bitmap_file         ./output.bmp         // if so, output bitmap file

training_sets_file         ./inputs.txt         // file to store input values (bitmap or not)
where_to_dump_outputs      ./outputs.txt        // where to dump final/periodic output values
randomize_weights          Y                    // whether to randomize weights
random_weights_lower       -0.5                 // if randomize: lower bound
random_weights_upper       0.5                  // if randomize: upper bound
preset_weights_file        ./weights.txt        // if not randomize: where to read weights from
where_to_dump_weights      ./weightsdump.txt    // where to dump final/periodic weights 
dump_every_x_iterations    100                  // dump weights/nodes every _x_ iterations

initial_learning_factor    0.1
learning_factor_scaler     2.0                  // set this to 1.0 to disable adaptive learning
min_learning_factor        0.001                // lower and upper bounds
max_learning_factor        2.0                  // for the learning factor
enable_weight_rollback     Y                    // whether or not to enable weight rollback

max_training_iterations    100000               // max # of iterations before stopping training
initial_error              1.0                  // what value to initialize the error at
target_training_error      0.00001              // target training error (to stop at)
```