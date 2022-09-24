from sources.decoder import DecoderFromNamedEntitySequence
import pickle
from flask_restful import Resource
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from utils import Config
from pathlib import Path
from pytorch_transformers import AdamW
import json
from flask import Flask, jsonify, make_response
from flask import request
from sources.model import KoelectraCRF_API, Koelectra_bilstm_CRF_API
import requests
import time
import config
from transformers import ElectraModel, ElectraTokenizer
import os, copy
from transformers import logging
from utils import init_logger, plot_loss_update
init_logger()
logger = logging.getLogger(__name__)

class API(Resource):
  def get(self):
        start = time.time()
        global args
        global model
        global tokenizer
        global device

        with open(config.DICT_DIR + '/ner_to_index_'+ args.select+ '.json', 'rb') as f:
            ner_to_index =  json.load(f)
            index_to_ner = {v: k for k, v in ner_to_index.items()}

        result_dict = {'results':[]}
        inputs = request.args.get('sentence', None)
        decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)
        t = tokenizer.encode_plus(text =inputs,text_pair=None,
                                    add_special_tokens = True,
                                    max_length=64,
                                    padding='max_length',
                                    return_token_type_ids=False,
                                    return_attention_mask=True,
                                    truncation=True,
                                    return_tensors='pt' # PyTorch Tensor format
                                    )
        x_input = t['input_ids'].to(device, dtype=torch.long)
        attention_mask = t['attention_mask'].to(device, dtype=torch.long)
        list_of_input_ids= x_input.tolist()
        list_of_pred_ids = model(x_input,attention_mask)
        for i in range(len(x_input)):
          w, _ = decoder_from_res(list_of_input_ids=[list_of_input_ids[i]], list_of_pred_ids=[list_of_pred_ids[i]], max_len = 64)
        wwww = []
        l = ['word','tag']
        for ii in w:
          ww= {key:value for key, value in ii.items() if key in l}
          wwww.append(ww)
        print("decoder_from_res.sentence_result",decoder_from_res.sentence_result)
        result_dict['sentence'] = [inputs]
        result_dict['results'].append(wwww)
        result_dict['response time'] = str(round(time.time() - start, 2)) + '초'
        rrrr = json.dumps(result_dict, indent=2, ensure_ascii=False)
        return make_response(rrrr)

  def post(self):
        start = time.time()
        global args
        global model
        global tokenizer
        global device
        
        with open(config.DICT_DIR + '/ner_to_index_'+ args.select+ '.json', 'rb') as f:
            ner_to_index =  json.load(f)
            index_to_ner = {v: k for k, v in ner_to_index.items()}

        result_dict = {'results':[]}
        inputs = request.json['sentences']
        sentences = copy.deepcopy(inputs)
        decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)
        for input1 in inputs:
          t = tokenizer.encode_plus(text =input1,text_pair=None,
                                    add_special_tokens = True,
                                    max_length=64,
                                    padding='max_length',
                                    return_token_type_ids=False,
                                    return_attention_mask=True,
                                    truncation=True,
                                    return_tensors='pt' # PyTorch Tensor format
                                    )

          x_input = t['input_ids'].to(device, dtype=torch.long)
          attention_mask = t['attention_mask'].to(device, dtype=torch.long)
          list_of_input_ids= x_input.tolist()
          list_of_pred_ids = model(x_input,attention_mask)
          for i in range(len(x_input)):
            w, d = decoder_from_res(list_of_input_ids=[list_of_input_ids[i]], list_of_pred_ids=[list_of_pred_ids[i]], max_len = 64)
          wwww = []
          l = ['word','tag']
          for ii in w:
            ww= {key:value for key, value in ii.items() if key in l}
            wwww.append(ww)
          result_dict['sentences'] = sentences
          result_dict['results'].append(wwww)
        result_dict['response time'] = str(round(time.time() - start, 2)) + '초'
        rrrr = json.dumps(result_dict, indent=2, ensure_ascii=False)
        return make_response(rrrr)


def load_model():
    global args
    global model
    global tokenizer
    global device

    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.target_gpu))
        logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
        logger.info('We will use the GPU:{}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        logger.info('No GPU available, using the CPU instead.')

    with open(config.DICT_DIR + '/ner_to_index_'+ args.select+ '.json', 'rb') as f:
      ner_to_index =  json.load(f)
    tokenizer = ElectraTokenizer.from_pretrained(config.MODEL_CONFIG['model'])
    model = Koelectra_bilstm_CRF_API(config=config.MODEL_CONFIG, num_classes=len(ner_to_index), tokenizer= tokenizer)
    model.to(device)
    model.load_state_dict(torch.load(args.load_ck),strict=False)
    model.eval()
