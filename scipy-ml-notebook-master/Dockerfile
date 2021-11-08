FROM jupyter/datascience-notebook:python-3.7.6

USER root

COPY /scripts /usr/share/datahub/scripts/
COPY /scripts/run_jupyter.sh /
COPY /scripts/jupyter_notebook_config.py /etc/jupyter/jupyter_notebook_config.py

RUN /usr/share/datahub/scripts/install-all.sh && \
	pip install pandas --upgrade && \
	pip install nltk && \
	cat /usr/share/datahub/scripts/canvas_exporter.py > /opt/conda/lib/python3.7/site-packages/nbgrader/plugins/export.py && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER && \
    chown -R jovyan:users /opt/conda/etc/jupyter/nbconfig && \
    chmod -R +r /opt/conda/etc/jupyter/nbconfig

# make compatible with DSMLP version of jupyterhub
RUN pip install jupyterhub==0.9.2

USER jovyan
WORKDIR /home/jovyan

USER root

# tensorflow, pytorch stable versions
# https://pytorch.org/get-started/previous-versions/
# https://www.tensorflow.org/install/source#linux

RUN apt-get update && \
	apt-get install -y \
			libtinfo5 
#			nvidia-cuda-toolkit

#RUN conda install cudatoolkit=10.2 \

# Install CUDA Toolkit 10
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && sudo apt update
RUN sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

RUN sudo apt update
RUN sudo apt install cuda=10.0.130-1 -y

# Install CuDNN 7 and NCCL 2
RUN wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
RUN sudo dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

RUN sudo apt update
RUN sudo apt install -y libcudnn7 libcudnn7-dev libnccl2 libc-ares-dev

RUN sudo apt autoremove -y

# Link libraries to standard locations
RUN echo 'export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}' >> ~/.bashrc


RUN pip install tensorflow-gpu==1.15.0

RUN apt-get install -y apt-utils
RUN apt-get install -y libglew-dev 
RUN apt-get install -y patchelf
RUN apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3

RUN sudo apt install libopenmpi-dev -y
RUN pip install mpi4py

RUN pip install --no-cache-dir 	gym==0.15.3 \
								mujoco-py==1.50.1.56 \
							  	baselines

RUN	chown -R 1000:1000 /home/jovyan

COPY ./tests/ /usr/share/datahub/tests/scipy-ml-notebook
RUN chmod -R +x /usr/share/datahub/tests/scipy-ml-notebook && \
    chown -R 1000:1000 /home/jovyan && \
	chmod +x /run_jupyter.sh

RUN ln -s /usr/local/nvidia/bin/nvidia-smi /opt/conda/bin/nvidia-smi

RUN pip install atari-py
RUN chmod -R 777 /opt/conda/lib/python3.7/site-packages/atari_py/

RUN pip install --upgrade tensorflow-gpu==1.15.0

RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jovyan/.mujoco/mjpro150/bin' >> ~/.bashrc

RUN mkdir ~/.mujoco
RUN cd ~/.mujoco && \
    wget https://www.roboti.us/download/mjpro150_linux.zip && \
    unzip mjpro150_linux.zip
COPY ./.mujoco/mjkey.txt ~/.mujoco/

USER $NB_UID:$NB_GID
ENV PATH=${PATH}:/usr/local/nvidia/bin