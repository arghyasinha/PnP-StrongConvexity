## Code for the Paper: "On the Strong Convexity of PnP Regularization using Linear Denoisers" by A. Sinha and K. N. Chaudhury

This repository contains the code for two experiments from the paper:

## Experiments

### Figure 1: Strong Convexity Index

- This experiment finds a lower bound for the strong convexity index of the objective function.
- The code for this experiment is in `Figure_1_Strong_Convexity_index.py`.
- There are three command line arguments: `forward_model`, `image_id`, and `sigma` in [0,255].
- where the `forward model` can be selected either `inpainting` or `deblurring`. The `image_id` referes to the name of the image from `images` folder.
- Example usage: `python Figure_1_Strong_Convexity_index.py inpainting 1 120`

### Figure 2: Global Convergence of PnP

- This experiment demonstrates the global convergence of PnP using two denoisers: a symmetric denoiser (`DSG-NLM`) and a nonsymmetric denoiser (`NLM`).
- The code for this experiment is in `Figure_2_Global_Convergence_inpainting.py`.

#### In addition, an example file has been provided (`Example_of_denoiser.ipynb`) on the denoiser `DSG-NLM`.

---






## Instructions to Activate Conda Environment and Install Required Packages

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


