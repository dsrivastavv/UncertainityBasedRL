pip install gym-retro==0.7.0
pip install gym==0.10
python -m atari_py.import_roms ./content/ROM/
python3 -m retro.import ./content/_roms/
mpiexec -n 2 python run.py