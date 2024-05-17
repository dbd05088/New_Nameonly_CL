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

MASTER_PORT=12345 CUDA_VISIBLE_DEVICES=0 python get_image_onestage.py --config_path ./configs/cogview2.yaml > logs/cogview2_30_39.log 2>&1 &
