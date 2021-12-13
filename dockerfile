FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
MAINTAINER molnarszilard
ENV TZ=Europe/Bucharest
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -yqq update
RUN apt-get -yqq install git cmake vim wget
RUN apt-get -yqq install unzip

#
# Conda
#
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
RUN chmod 777 ./Anaconda3-2020.11-Linux-x86_64.sh
RUN bash ./Anaconda3-2020.11-Linux-x86_64.sh -b -p /home/anaconda
ENV PATH="/home/anaconda/bin:$PATH"
RUN conda create -y -n py27 python=2.7 anaconda
RUN conda init bash


RUN conda install protobuf -yq
RUN ln -s /usr/local/cuda/include/crt/math_functions.hpp /usr/local/cuda/include/math_functions.hpp

SHELL ["conda", "run", "-n", "py27", "/bin/bash", "-c"]

RUN pip install tflearn==0.3.2
RUN pip install tensorflow_gpu==1.13.1

RUN git config --global user.email "molnarszilard10@gmail.com"
RUN git config --global user.name "molnarszilard"

RUN mkdir latent_3d; cd latent_3d; git clone https://github.com/molnarszilard/latent_3d_points.git
RUN latent_3d/latent_3d_points/download_data.sh

EXPOSE 8891
RUN echo "(jupyter notebook --ip=0.0.0.0 --port 8891 --allow-root &)" > start_jupyter.sh
CMD ["/bin/bash"]

# RUN echo "conda activate py2.7" > ~/.bashrc
# # RUN echo $CONDA_DEFAULT_ENV
# # RUN conda activate py27
# RUN export http_proxy=http://cache.utcluj.ro:3128/ https_proxy=http://cache.utcluj.ro:3128/
# RUN pip install open3d
# RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN export PYTHONPATH=/latent_3d/
# RUN cd latent_3d/latent_3d_points/external/structural_losses/ && bash compile.sh
# RUN 'cd /latent_3d/latent_3d_points/ ; bash download_data.sh'
# # RUN cd /
# RUN bash start_jupyter.sh
