from modal import Stub, Image, method

IMAGE_MODEL_DIR = "/model"
model_name = "TheBloke/orca_mini_13B-GPTQ"

def download_model():
    from huggingface_hub import snapshot_download
    
    path = snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR)
    print(path)

image = (
    #Image.from_dockerhub("anibali/pytorch:2.0.0-cuda11.8")
    Image.from_dockerhub("cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-devel-ubuntu20.04")
    #Image.from_dockerhub("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04")
    #Image.from_dockerhub("nvcr.io/nvidia/pytorch:23.06-py3")
    .apt_install("git")
    #.pip_install(
    #    "torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118"
    #)
    .pip_install(
        "huggingface-hub"
    )
    .run_commands("git clone https://github.com/turboderp/exllama && cp -r /exllama/* /root && cd /root && pip install -r requirements.txt")
    .run_function(download_model)
    .run_commands("apt-get update -y && apt install build-essential -y")
    #.dockerfile_commands("ENV CPATH=/usr/local/cuda/targets/x86_64-linux/include")
)

stub = Stub(image=image)

@stub.cls(gpu="T4")
class Model:
    def __enter__(self):
        import sys, os
        #sys.path.append("/exllama")
        #os.chdir("/exllama")
        print(sys.path)
        print(os.getcwd())
        from model import ExLlama, ExLlamaCache, ExLlamaConfig
        from tokenizer import ExLlamaTokenizer
        from generator import ExLlamaGenerator
        import os, glob
        
        model_directory = IMAGE_MODEL_DIR
        
        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]
        
        self.config = ExLlamaConfig(model_config_path)
        self.config.model_path = model_path 
        
        self.model = ExLlama(self.config)
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)
        self.cache = ExLlamaCache(self.model)
        self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)

    @method()
    def generate(self, prompt):
        self.generator.settings.token_repetition_penalty_max = 1.2
        self.generator.settings.temperature = 0.95
        self.generator.settings.top_p = 0.65
        self.generator.settings.top_k = 100
        self.generator.settings.typical = 0.5
        
        max_new_tokens = 1024
        
        output = self.generator.generate_simple(prompt, max_new_tokens = max_new_tokens)
        print(output[len(prompt):])

@stub.local_entrypoint()
def main(myrequest: str):
    model = Model()
    new_prompt_template = """### System:
You are an AI assistant that follows instruction extremely well. Help as much as you can.

### User:
{prompt}

### Response:
"""
    #myrequest = "Write a short blog post on how to start a new website."
    model.generate.call(new_prompt_template.format(prompt=myrequest))
