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

MASTER_PORT=12145 CUDA_VISIBLE_DEVICES=6 nohup python get_image_onestage.py --config_path ./configs/NICO_cogview2.yaml > logs/NICO_cogview2_0_29.log 2>&1 &
MASTER_PORT=12146 CUDA_VISIBLE_DEVICES=7 nohup python get_image_onestage.py --config_path ./configs/NICO_cogview2.yaml > logs/NICO_cogview2_30_59.log 2>&1 &
