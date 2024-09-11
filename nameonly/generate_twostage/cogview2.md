pip install icetk
pip install Pillow
pip install diffusers
pip install SwissArmyTransformer==0.2.8
pip install accelerator
pip install torchvision
pip install icetk # 0.0.7
pip install protobuf==3.20.0 # 3.18.3

/root/.sat_models/
/home/.sat_models

git clone https://github.com/Sleepychord/Image-Local-Attention
cd Image-Local-Attention
python setup.py install

# If error (cusolverDn.h) occurs, please include PATH to cuda and run again
export PATH=/usr/local/cuda/bin:$PATH

# Datacenter setup
1. sharef.tar scratch에 scp로 옮기고 ln -s
2. conda create -n cogview python=3.10.8
3. pip install icetk Pillow diffusers SwissArmyTransformer==0.2.8 accelerator torchvision
4. pip install icetk protobuf==3.20.0

# If numpy error occurs
pip install numpy==1.24.1
