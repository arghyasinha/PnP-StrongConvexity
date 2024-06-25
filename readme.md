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

