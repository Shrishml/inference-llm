# Import libraries
from fastapi import FastAPI
from pydantic import BaseModel
from model_inference import *
import torch
from fastapi.encoders import jsonable_encoder
app = FastAPI()
import gc


class Promt(BaseModel):
    instruction: str
    input: str
    max_token: int
    temperature: float

class ModelName(BaseModel):
    name: str

inference = modename_dict['vicuna']() 
# default model path
model_path='arixon/vicuna-7b'
tokenizer_path= 'arixon/vicuna-7b'

inference = modename_dict['vicuna']() 
inference.load(model_path, tokenizer_path)
 

# Load the model
@app.post("/load")
def load(input1:ModelName):
    data = jsonable_encoder(input1)
    torch.cuda.empty_cache()
    gc.collect()
    global inference
    
    if 'inference' in globals():
        inference.unload()
        
    model_name =  data['name']
    inference = modename_dict[model_name]()
    inference.load(model_path=modepath_dict[model_name], tokenizer_path=modepath_dict[model_name])
    return model_name

# generate
@app.post("/generate")
def predict(input1:Promt):
    data = jsonable_encoder(input1)
    instruction= data['instruction']
    if inference.model_name=='chatglm':
         instruction = 'Respond in English, '+ instruction
    question = """instruction: {}, input: {}""".format(instruction, data['input'])
    max_length=data['max_token']
    temperature=data['temperature']
    response = inference.generate(question, max_length, temperature)
    
    # Take the first value of prediction
    output =response
    tokens= inference.tokenizer.tokenize(response)
    number_tokens = len(tokens)
    return {'output':output, 'number_of_tokens': number_tokens, 'tokens':tokens}

   