#!/bin/bash
# install latest version of anaconda
if !command -v conda 2>/dev/null; then
    anaconda_archive_path=https://repo.continuum.io/archive/
    anaconda_file=$(wget -q -O - $anaconda_archive_path | grep "Anaconda2-" | grep "Linux" | grep "x86_64" | head -n 1 | cut -d \" -f 2)
    wget -O ./anaconda.sh $anaconda_archive_path$anaconda_file
    bash ./anaconda.sh
fi
# save conda to ~/.bashrc
current_path=`pwd`
echo export PATH=$current_path/anaconda2/bin:\$PATH >> ~/.bashrc
source ~/.bashrc

# install tensorflow in anaconda environment
conda create -n tensorflow python=2.7
source activate tensorflow
# conda install -c conda-forge tensorflow
conda install tensorflow

# configure keras to use tensorflow as default backend
mkdir ~/.keras/
echo '{"image_dim_ordering": "tf", "epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow"}' > ~/.keras/keras.json

# install requirement packages
pip install -r requirements.txt

# deactivate tensorflow
source deactivate tensorflow
