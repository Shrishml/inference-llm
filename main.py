# Import libraries
from fastapi import FastAPI
from pydantic import BaseModel
from model_inference import *
import torch
from fastapi.encoders import jsonable_encoder
app = FastAPI()
import gc
# print('started')
# model_path='arixon/vicuna-7b'
# tokenizer_path= 'arixon/vicuna-7b'
# inference = modename_dict['vicuna']() 
# print('worked')
# print(inference.DEVICE)
# inference.load(model_path='arixon/vicuna-7b', tokenizer_path= 'arixon/vicuna-7b')
# print('finished')

# response = inference.generate('who are you?', 100, 0.4)

# print(response)

print('started')

inference = modename_dict['chatglm']() 
print('worked')
print(inference.DEVICE)
inference.load(model_path='THUDM/chatglm-6b-int4', tokenizer_path= 'THUDM/chatglm-6b-int4')
print('finished')

response = inference.generate('who are you?', 100, 0.4)

print(response)