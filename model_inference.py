
import torch

class ModelInference():
    """Base class for LLM model inference
    """

    def __init__(self):
        self._is_loaded = False
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
       
    def load(self, model_path, tokenizer_path, **kwargs):
        self._load(model_path, tokenizer_path, **kwargs)
        self._is_loaded = True

    def unload(self):
        if self._is_loaded:
            del self.model
            del self.tokenizer
            self.model = 'model unloaded'
            self.tokenizer = 'tokenizer unloaded'
        else:
            print('model is not loaded yet')

    def generate(self, question, max_length, temperature, **kwargs):
        response = self._generate( question, max_length, temperature, **kwargs)
        return response
        

    @property
    def is_loaded(self):
        """ Whether model is loaded. """
        return self._is_loaded

    

class VicunaInference(ModelInference):
    """Base class for LLM model inference
    """

    def __init__(self):
        self.model_name = 'vicuna'
        super(VicunaInference, self).__init__()

    def _load(self, model_path='arixon/vicuna-7b', tokenizer_path= 'arixon/vicuna-7b', **kwargs):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        params = {"torch_dtype": torch.float16}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        print('tokenizer loaded')
        self.model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **params).to(self.DEVICE)
        print('model loaded')

    def _generate(self, question, max_length, temperature, **kwargs):
        from chat import chat_one_shot
        response = chat_one_shot(model = self.model, tokenizer = self.tokenizer, 
                            model_name = 'LLM', device =self.DEVICE,
                            message = question, temperature = temperature, 
                            max_new_tokens = max_length)
        return response
       
        

class ChatglmInference(ModelInference):
    """Base class for LLM model inference
    """

    def __init__(self):
        self.model_name = 'chatglm'
        super(ChatglmInference, self).__init__()

    def _load(self, model_path="THUDM/chatglm-6b-int4", tokenizer_path= "THUDM/chatglm-6b-int4", **kwargs):
        from transformers import AutoTokenizer, AutoModel

        # THUDM/chatglm-6b
        # THUDM/chatglm-6b-int4
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(tokenizer_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

    def _generate(self, question, max_length, temperature, **kwargs):
        
        response, _ = self.model.chat(self.tokenizer, question, history=[],
                                    max_length=max_length,
                                    temperature=temperature)
       
        return response
    
modename_dict =  {'vicuna':VicunaInference, 'chatglm':ChatglmInference}
modepath_dict =  {'vicuna':'arixon/vicuna-7b', 'chatglm':'THUDM/chatglm-6b-int4'}