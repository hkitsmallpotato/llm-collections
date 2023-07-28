FROM public.ecr.aws/saturncloud/saturn-python-pytorch:2023.05.01
WORKDIR /content
RUN pip install git+https://github.com/huggingface/diffusers transformers accelerate imageio[ffmpeg] -U einops omegaconf decord xformers==0.0.20 safetensors
RUN sudo useradd -ms /bin/bash dockeruser && sudo groupadd work && sudo usermod -aG sudo,work dockeruser && sudo usermod -aG work jovyan
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' | sudo tee -a /etc/sudoers > /dev/null
RUN sudo chown -R dockeruser:work /content && sudo chmod 775 /content
USER dockeruser
RUN git clone -b dev https://github.com/camenduru/Text-To-Video-Finetuning
RUN sudo apt-get update && sudo apt -y install aria2
COPY ./download_list_zeroscope_v2_576w.txt .
RUN aria2c -c -x 16 -s 16 -k 1M -i download_list_zeroscope_v2_576w.txt
RUN pip install boto3
COPY ./bulkvideo.py .
USER jovyan
ENTRYPOINT ["python3", "./bulkvideo.py"]
