# # setup python 3.7 environment
# # conda create -n rl_project python=3.7 -y 
# # conda deactivate 
# # conda deactivate 
# # conda activate rl_project
# # python --version

# # setup mujoco linux
# # https://www.chenshiyu.top/blog/2019/06/19/Tutorial-Installation-and-Configuration-of-MuJoCo-Gym-Baselines/
# # wget https://roboti.us/download/mjpro150_linux.zip
# # mkdir ~/.mujoco
# # cp mjpro150_linux.zip ~/.mujoco
# # unzip ~/.mujoco/mjpro150_linux.zip
# # mv ~/.mujoco/mjpro150_linux mjpro150
# # mv .mujoco/mjkey.txt ~/.mujoco
# # rm mjpro150_linux.zip

# # setup mujoco mac
# wget https://roboti.us/download/mjpro150_osx.zip
# mkdir ~/.mujoco
# cp mjpro150_osx.zip ~/.mujoco
# unzip ~/.mujoco/mjpro150_osx.zip -d ~/.mujoco
# cp .mujoco/mjkey.txt ~/.mujoco
# cp .mujoco/mjkey.txt ~/.mujoco/mjpro150/bin
# rm mjpro150_osx.zip


# # ubuntu
# # sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

# # mac os
# brew install cmake openmpi

# # tensorflow cpu release
# pip install tensorflow==1.15

# # tensorflow gpu release for ubuntu and windows
# # pip install --upgrade tensorflow-gpu==1.15

# # install python packages
# pip install -U 'mujoco-py<1.50.2,>=1.50.1'
# pip install gym[all]==0.15.3
# pip install baselines
# pip install mpi4py
# pip install atari-py

pip install gym-retro==0.7.0
pip install gym==0.10
python -m atari_py.import_roms ./content/ROM/
python3 -m retro.import ./content/_roms/
mpiexec -n 2 python run.py