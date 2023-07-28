FROM public.ecr.aws/saturncloud/saturn-python-pytorch:2023.05.01
WORKDIR /content
RUN pip install git+https://github.com/huggingface/diffusers transformers accelerate imageio[ffmpeg] -U einops omegaconf decord xformers==0.0.20 safetensors
RUN sudo useradd -ms /bin/bash dockeruser && sudo usermod -aG sudo dockeruser
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN sudo chown -R dockeruser:dockeruser /content && sudo chmod 755 /content
USER dockeruser
RUN git clone -b dev https://github.com/camenduru/Text-To-Video-Finetuning
RUN sudo apt-get update && sudo apt -y install aria2
COPY ./download_list_zeroscope_v2_576w.txt .
RUN aria2c -c -x 16 -s 16 -k 1M -i download_list_zeroscope_v2_576w.txt
RUN pip install boto3
COPY ./bulkvideo.py .
ENTRYPOINT ["python3", "./bulkvideo.py"]
