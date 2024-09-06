# This is code for  the paper " On the Strong Convexity of PnP Regularization using Linear Denoisers" by A. Sinha and K. N. Chaudhury

# This includes codes for two experiments:

## Figure 1. It finds a lower bound to the strong convexity index of the objective. The code is hosted in ```Figure_1_Strong_Convexity_index.py```.

## Figure 2. Demonstrates the global convergence of PnP using a symmetric denoiser ```DSG-NLM``` and a nonsymmetric denoiser ```NLM```. The code is hosted in ```Figure_2_Global_Convergence_inpainting.py```.









# Instructions to Activate Conda Environment and Install Required Packages

## Step 1: Activate Conda Environment

1. Open your terminal or command prompt.

2. Create a new conda environment with Python version 3.8.18 (if you haven't already):

    ```sh
    conda create --name myenv python=3.8.18
    ```

3. Activate the newly created conda environment:

    ```sh
    conda activate myenv
    ```

## Step 2: Install Required Packages

1. Ensure you are in the root directory of your project where `dependencies.sh` is located.

2. Run the `dependencies.sh` script to install the required packages:

    ```sh
    sh dependencies.sh
    ```

That's it! Your conda environment should now be activated with Python 3.8.18, and the required packages should be installed.


