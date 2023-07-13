from modal import Stub, Image, method, web_endpoint

import time
from fastapi.responses import StreamingResponse

IMAGE_MODEL_DIR = "/model"
HF_MODEL_NAME = "TheBloke/Manticore-13B-Chat-Pyg-Guanaco-SuperHOT-8K-GPTQ"

def download_model():
    from huggingface_hub import snapshot_download
    
    model_name = "TheBloke/Manticore-13B-Chat-Pyg-Guanaco-SuperHOT-8K-GPTQ" # hardcoding necessary due to quirks of modal.com
    path = snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR)
    print(path)


# Key point is that pytorch is really picky on pin the exact version of torch + cuda.
# Once we find a docker image with the exact versions, everything works.
image = (
    Image.from_dockerhub("cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-devel-ubuntu20.04")
    .apt_install("git")
    #.pip_install(
    #    "torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118"
    #)
    .pip_install(
        "huggingface-hub"
    )
    .run_commands("git clone https://github.com/turboderp/exllama && cp -r /exllama/* /root && cd /root && pip install -r requirements.txt")
    .run_function(download_model)
)

stub = Stub("llm_manticore_guanaco", image=image)

@stub.cls(gpu="T4", container_idle_timeout=180)
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
        
        # SuperHOT for long context
        self.config.max_seq_len = 8192
        self.config.compress_pos_emb = 4
        self.config.max_input_len = 8192
        
        self.model = ExLlama(self.config)
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)
        self.cache = ExLlamaCache(self.model)
        self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)

    @method()
    def generate(self, prompt, temp, top_p, top_k, rep_penalty, max_tokens):
        self.generator.settings.token_repetition_penalty_max = rep_penalty
        self.generator.settings.temperature = temp
        self.generator.settings.top_p = top_p
        self.generator.settings.top_k = top_k
        self.generator.settings.typical = 0.5
        
        max_new_tokens = max_tokens

        t0 = time.time()
        
        output = self.generator.generate_simple(prompt, max_new_tokens = max_new_tokens)
        response = output[len(prompt):]
        
        # get num new tokens:
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_tokens[0])
        new_tokens = self.tokenizer.encode(response)
        new_tokens = len(new_tokens[0])

        t1 = time.time()
        _sec = t1-t0
        _tokens_sec = new_tokens/(_sec)

        print(response)
        print(f"Output generated in {_sec} ({_tokens_sec} tokens/s, {new_tokens}, context {prompt_tokens})")

        return response
    
    # copy of generate_simple() so that I could yield each token for streaming without having to change generator.py and make merging updates a nightmare:
    @method()
    async def generate_simple(self, prompt, temp, top_p, top_k, rep_penalty, max_tokens):
        self.generator.settings.token_repetition_penalty_max = rep_penalty
        self.generator.settings.temperature = temp
        self.generator.settings.top_p = top_p
        self.generator.settings.top_k = top_k
        self.generator.settings.typical = 0.5

        max_new_tokens = max_tokens

        t0 = time.time()
        new_text = ""
        last_text = ""
        _full_answer = ""

        self.generator.end_beam_search()

        ids = self.tokenizer.encode(prompt)
        self.generator.gen_begin_reuse(ids)

        for i in range(max_new_tokens):
            token = self.generator.gen_single_token()
            text = self.tokenizer.decode(self.generator.sequence[0])
            new_text = text[len(prompt):]

            # Get new token by taking difference from last response:
            new_token = new_text.replace(last_text, "")
            last_text = new_text

            #print(new_token, end="", flush=True)
            yield new_token

            # [End conditions]:
            #if break_on_newline and # could add `break_on_newline` as a GenerateRequest option?
            #if token.item() == tokenizer.newline_token_id:
            #    print(f"newline_token_id: {tokenizer.newline_token_id}")
            #    break
            if token.item() == self.tokenizer.eos_token_id:
                #print(f"eos_token_id: {tokenizer.eos_token_id}")
                break

        # all done:
        self.generator.end_beam_search() 
        _full_answer = new_text

        # get num new tokens:
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_tokens[0])
        new_tokens = self.tokenizer.encode(_full_answer)
        new_tokens = len(new_tokens[0])

        # calc tokens/sec:
        t1 = time.time()
        _sec = t1-t0
        _tokens_sec = new_tokens/(_sec)

        print(f"full answer: {_full_answer}")

        print(f"Output generated in {_sec} ({_tokens_sec} tokens/s, {new_tokens}, context {prompt_tokens})")


    #@method()
    #@web_endpoint()
    #def generate_stream(self, prompt, temp, top_p, top_k, rep_penalty, max_tokens):
    #    return StreamingResponse(generate_simple(prompt, max_tokens), media_type="text/event-stream")
    


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
