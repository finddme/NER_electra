import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig, BertForMaskedLM
from transformers import BertTokenizer
from torchcrf import CRF
import pytorch_lightning as pl
import torch.nn.functional as F
import config
from transformers import ElectraModel, ElectraTokenizer
from transformers import AutoTokenizer, AutoModelForPreTraining
import numpy as np


class KoelectraCRF(nn.Module):
    """ KoBERT with CRF """
    def __init__(self, config, num_classes,tokenizer, vocab=None) -> None:
        super(KoelectraCRF, self).__init__()
        self.electra = model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.tokenizer = tokenizer

        self.dropout = nn.Dropout(config['dropout'])
        self.linear = nn.Linear(config['hidden_size'], num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input_ids, attention_mask, tags=None):
        self.electra.resize_token_embeddings(self.tokenizer.vocab_size) # CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling cublasCreate(handle) 고치려고 넣음
        outputs = self.electra(input_ids=input_ids,
                            attention_mask=attention_mask)               
        loss = outputs[0]
        loss = self.dropout(loss)
        emissions = self.linear(loss)
        if tags is not None:
            log_likelihood, sequence_of_tags = -self.crf(F.log_softmax(emissions, 2), tags, reduction='token_mean'), self.crf.decode(emissions)
            correct =0
            total = 0
            for t,s in zip(tags.tolist(),sequence_of_tags):
                c = 0
                for tt,ss in zip(t,s):
                    if tt !=0 :
                        total +=1
                        if tt==ss:
                            correct += 1
            acc = correct / total

            return log_likelihood, sequence_of_tags, acc

class KoelectraCRF_API(nn.Module):
    def __init__(self, config, num_classes,tokenizer, vocab=None) -> None:
        super(KoelectraCRF_API, self).__init__()
        self.electra = model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.tokenizer = tokenizer

        self.dropout = nn.Dropout(config['dropout'])
        self.linear = nn.Linear(config['hidden_size'], num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input_ids,attention_mask,tags=None):
        self.electra.resize_token_embeddings(self.tokenizer.vocab_size) # CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling cublasCreate(handle) 고치려고 넣음
        outputs = self.electra(input_ids=input_ids,attention_mask=attention_mask)   
               
        loss = outputs[0]
        loss = self.dropout(loss)
        emissions = self.linear(loss)
        if tags is not None:
            log_likelihood, sequence_of_tags = -self.crf(F.log_softmax(emissions, 2), tags, reduction='token_mean'), self.crf.decode(emissions)
            return sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags


class Koelectra_bilstm_CRF(nn.Module):
    """ KoBERT with Bilstm + CRF """
    def __init__(self, config, num_classes,tokenizer, vocab=None) -> None:
        super(Koelectra_bilstm_CRF, self).__init__()
        self.electra = model = ElectraModel.from_pretrained(config['model'])
        self.tokenizer = tokenizer
        self.bilstm = nn.LSTM(config['hidden_size'], (config['hidden_size']) // 2, dropout=config['dropout'], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.linear = nn.Linear(config['hidden_size'], num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input_ids, attention_mask, tags=None):
        self.electra.resize_token_embeddings(self.tokenizer.vocab_size) 
        outputs = self.electra(input_ids=input_ids,
                            attention_mask=attention_mask)             
        loss = outputs[0]
        loss = self.dropout(loss)
        loss, hc = self.bilstm(loss)
        loss = self.dropout(loss)
        loss, hc = self.bilstm(loss)
        loss = self.dropout(loss)
        loss, hc = self.bilstm(loss)
        emissions = self.linear(loss)
        if tags is not None:
            log_likelihood, sequence_of_tags = -self.crf(F.log_softmax(emissions, 2), tags, reduction='token_mean'), self.crf.decode(emissions)
            correct =0
            total = 0
            for t,s in zip(tags.tolist(),sequence_of_tags):
                c = 0
                for tt,ss in zip(t,s):
                    if tt !=0 :
                        total +=1
                        if tt==ss:
                            correct += 1
            acc = correct / total
            
            return log_likelihood, sequence_of_tags, acc

class Koelectra_bilstm_CRF_API(nn.Module):
    def __init__(self, config, num_classes,tokenizer, vocab=None) -> None:
        super(Koelectra_bilstm_CRF_API, self).__init__()
        self.electra = model = ElectraModel.from_pretrained(config['model'])
        self.tokenizer = tokenizer
        self.bilstm = nn.LSTM(config['hidden_size'], (config['hidden_size']) // 2, dropout=config['dropout'], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.linear = nn.Linear(config['hidden_size'], num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input_ids,attention_mask,tags=None):
        self.electra.resize_token_embeddings(self.tokenizer.vocab_size) # CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling cublasCreate(handle) 고치려고 넣음
        outputs = self.electra(input_ids=input_ids,attention_mask=attention_mask)   
               
        loss = outputs[0]
        loss = self.dropout(loss)
        loss, hc = self.bilstm(loss)
        loss = self.dropout(loss) # (1,64,768)
        loss, hc = self.bilstm(loss)
        loss = self.dropout(loss)
        loss, hc = self.bilstm(loss)
        emissions = self.linear(loss)
        if tags is not None:
            log_likelihood, sequence_of_tags = -self.crf(F.log_softmax(emissions, 2), tags, reduction='token_mean'), self.crf.decode(emissions)
            print("log_likelihood",log_likelihood)
            return sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags
