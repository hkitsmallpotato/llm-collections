from modal import Image, Stub, gpu, method, web_endpoint

IMAGE_MODEL_DIR = "/model"


def download_model():
    from huggingface_hub import snapshot_download

    model_name = "TheBloke/WizardCoder-15B-1.0-GPTQ"
    snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR)

#image = (
#    Image.from_dockerhub("cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-devel-ubuntu20.04")
#    #Image.debian_slim(python_version="3.10")
#    .pip_install("huggingface-hub")
#    #.apt_install("curl")
#    #.run_commands("curl -O https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.2.2/auto_gptq-0.2.2+cu118-cp39-cp39-linux_x86_64.whl && pip3 install auto_gptq-0.2.2+cu118-cp39-cp39-linux_x86_64.whl")
#    .apt_install("git")
#    #.pip_install(
#    #    "auto-gptq @ git+https://github.com/PanQiWei/AutoGPTQ.git@b5db750c00e5f3f195382068433a3408ec3e8f3c",
#    #    "transformers @ git+https://github.com/huggingface/transformers.git@f49a3453caa6fe606bb31c571423f72264152fce")
#    .run_commands("git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ && git checkout v0.2.1 && GITHUB_ACTIONS=true PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=\"8.0;8.6+PTX;8.9;9.0\" pip3 install .")
#    .run_function(download_model)
#)

#nvidia/cuda:11.7.0-devel-ubuntu20.04
image = (
    Image.from_dockerhub(
        "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel",
        setup_dockerfile_commands=[
            "RUN apt-get update",
            "RUN apt-get install -y python3 python3-pip", #python-is-python3
        ],
    )
    .apt_install("git", "gcc", "build-essential")
    .run_commands(
        "git clone https://github.com/PanQiWei/AutoGPTQ.git",
        "cd AutoGPTQ && pip3 install -e .",
        gpu="A10G",
    )
    .pip_install(
        "huggingface_hub",
        "transformers",
        "torch",
        "einops",
    )
    .run_function(download_model)
)

stub = Stub("llm_wizard_coder", image=image)

@stub.cls(gpu="T4")
class Model:
    def __enter__(self):
        from transformers import AutoTokenizer, pipeline, logging
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        import argparse
        
        model_name_or_path = "/model"
        
        use_triton = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        
        self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                     use_safetensors=True,
                     #device="cuda:0",
                     device_map="auto",
                     use_triton=use_triton,
                     strict=False,
                     quantize_config=None).to("cuda:0")
        
        # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
        logging.set_verbosity(logging.CRITICAL)
        
        #self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto")
        print("Loaded WizardCoder Model.")
    
    @method()
    def generate(self, prompt, temp, top_p, top_k, max_token):
        from transformers import pipeline
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto")
        outputs = pipe(prompt, max_new_tokens=max_token, do_sample=True, temperature=temp, top_k=top_k, top_p=top_p)
        respond = outputs[0]['generated_text'][len(prompt):]
        print(respond)
        return respond
    
    @method()
    def generate_streaming(self, prompt, temp, top_p, top_k, max_token):
        from threading import Thread
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            inputs=inputs.input_ids.to("cuda:0"),
            attention_mask=inputs.attention_mask.to("cuda:0"),
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_token,
            streamer=streamer,
        )

        # Run generation on separate thread to enable response streaming.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text

        thread.join()

