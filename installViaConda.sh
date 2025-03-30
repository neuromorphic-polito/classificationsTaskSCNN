#!/bin/bash

# Check if conda is installed
flagConda=false

if ! command -v conda &> /dev/null
then
    echo "It appears that CONDA is not installed"
    echo "Run the following commands to install it"
    echo ""
    echo "    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh"
    echo "    chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh"
    echo "    ./Miniconda3-py38_4.10.3-Linux-x86_64.sh"
    echo ""
    echo "once done, restart the terminal"
    echo ""
else
    flagConda=true
fi


if $flagConda
then
    # Installing enviroment via CONDA
    source /home/$USER/miniconda3/etc/profile.d/conda.sh
    conda create --name scnn python=3.13.2
    conda activate scnn

    # Installing package via CONDA
    conda install pip

    # Installing package via PIP
    export CUDA_PATH=/usr/local/cuda
    pip install pybind11==2.13.6
    pip install psutil==7.0.0
    pip install numpy==2.2.4
    tar -xzf genn-5.1.0.tar.gz
    cd genn-5.1.0
    python setup.py install
    pip install matplotlib==3.10.1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    pip install scipy==1.15.2
    pip install pandas==2.2.3
fi
