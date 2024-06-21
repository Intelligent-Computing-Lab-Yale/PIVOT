# PIVOT

## Prerequisites

Ensure you have Conda installed on your system. You can download and install Conda from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

## Setting Up the Environment

This guide will help you set up a Conda environment with the necessary dependencies for your project.
Follow these steps to set up your Conda environment:

1. **Create the Conda Environment**

    Open your terminal and run the following command to create a new Conda environment named `pivot_env` with Python 3.9 and the specified packages:

    ```sh
    conda create -y -n pivot_env python=3.9 cupy pkg-config libjpeg-turbo opencv cudatoolkit=11.3 numba -c pytorch -c conda-forge
    ```

2. **Activate the Conda Environment**

    Activate the newly created environment with the following command:

    ```sh
    conda activate pivot_env
    ```

3. **Install PyTorch with CUDA Enabled**

    With the environment activated, install PyTorch and related packages with CUDA support:

    ```sh
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

4. **Install Additional Python Packages**

    Install additional Python packages using pip:

    ```sh
    pip install ffcv
    pip install timm
    ```

Your Conda environment is now set up with all the necessary dependencies. You can start using it for your project.

## Troubleshooting

If you encounter any issues during the setup process, ensure that your Conda installation is up to date and that you have a stable internet connection. Refer to the [Conda documentation](https://docs.conda.io/) for further assistance.

