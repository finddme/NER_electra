from sources.mongo_processor import Mongo
import os
import json
import torch
import numpy as np
from pathlib import Path
import config
import argparse
from torch.utils.data import Dataset
from transformers import ElectraModel, ElectraTokenizer

class NERDataset_electra(Dataset):
    def __init__(self, tokenizer, max_len, mongo, select):
        self.ids = mongo.find_item3()
        self.mongo = mongo
        self.max_len = max_len
        self.tokenizer = tokenizer 
        self.select = select
        self.idx = self.load_id()

        if not os.path.exists(config.DICT_DIR + '/ner_to_index_'+ select + '.json'):
            self.ner2index = self.create_ner_dict()
        else: self.ner2index = self.load_dict()
    
    def __getitem__(self, batch):
        sentence, label, real_sentence, attention_mask = self.encode_lines(id=self.idx[batch])
        sentence = self.set_length(sentence.tolist())
        attention_mask = self.set_length(attention_mask.tolist())
        label = self.set_length(label)

        dataset = {'input_ids': torch.tensor(sentence).long(),
                    'attention_mask': torch.tensor(attention_mask).long(),
                    'labels': torch.tensor(label).long()}
        return dataset

    def __len__(self):
        #40분의 1만 가져오기
        return self.ids.count()

    def set_length(self,list_input):
        if len(list_input)>config.MODEL_CONFIG['max_len']:
            list_input = list_input[:config.MODEL_CONFIG['max_len']-1]
            list_input += [3]
        return list_input

    def load_id(self):
        idx = self.mongo.find_id()
        idx = list(idx)
        return idx

    def load_data(self):
        datas = self.mongo.find_item()
        sentences, tags, details, labels, words = [], [], [], [], []

        for i in datas:
            sentences.append(i.get('_source').get('sentence'))
            tags.append(i.get('_source').get('ne_list'))
            labels.append(i.get('_source').get('idxs'))

        for i in tags:
            tmp,tmp2 = [], []
            for j in i:
                tmp.append(j.get("detail"))
                tmp2.append(j.get("word"))
            details.append(tmp)
            words.append(tmp2)

        return sentences, labels, details, words

    def load_dict(self):
        with open(config.DICT_DIR + '/ner_to_index_'+ self.select+ '.json', 'rb') as f:
            return json.load(f)


    def create_ner_dict(self):
        _, _, details, _ = self.load_data()
        details = np.array(details).squeeze(axis=1).tolist()
        details = set([y for x in details for y in x])

        if self.select == 's': #small
            ner_tags = set(self.select_ner_list()[0])
        elif self.select == 'm': #medium
            ner_tags = details - set(self.select_ner_list()[1])
        elif self.select == '42': #medium
            ner_tags = set(self.select_ner_list()[2])
        else: #large
            ner_tags = details
        ner_to_index = {"O": 1, "[PAD]": 0, "[CLS]": 2, "[SEP]": 3}

        for ner_tag in ner_tags:
            ner_to_index["B-" + ner_tag] = len(ner_to_index)
            ner_to_index["I-" + ner_tag] = len(ner_to_index)

        with open(config.DICT_DIR + "/ner_to_index_" + self.select + ".json" , "w", encoding="utf-8") as io:
            json.dump(ner_to_index, io, ensure_ascii=False, indent=4)

        return ner_to_index

    def encode_lines(self, id):
        token2index, categories, index, token_sum_indexs, sentence, attn_mask = self.split_entity(id)

        bio_tags = self.bio_process(categories, index, token_sum_indexs)
        bio_tags2index = [self.ner2index['[CLS]']] + [self.ner2index[x] for x in bio_tags] + [self.ner2index['[SEP]']]
        padding_lenght = self.max_len - len(bio_tags2index)
        bio_tags2index += [0] * padding_lenght 
        return token2index, bio_tags2index, sentence, attn_mask

    def split_entity(self, idx):
        sentence = list(self.mongo.find_item2(data=idx))[0].get("_source").get("sentence")
        category = list(self.mongo.find_item2(data=idx)[0].get("_source").get("ne_list")[0].get('detail'))
        index = list(self.mongo.find_item2(data=idx)[0].get("_source").get("idxs"))
        tokens = self.tokenizer.tokenize(sentence)
        inputs = self.tokenizer.encode_plus(text=sentence,
                                    text_pair=None,
                                    add_special_tokens = True,
                                    max_length=self.max_len,
                                    padding='max_length',
                                    return_token_type_ids=False,
                                    return_attention_mask=True,
                                    truncation=True,
                                    return_tensors='pt' # PyTorch Tensor format
                                    )
        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        token_sum_indexs = self.token_sum_index(tokens)

        return input_ids, category, index, token_sum_indexs, sentence, attn_mask

    def token_sum_index(self, tokens):
        sum_of_token_start_index, sum = [], 0
        for i, token in enumerate(tokens):
            if i == 0:
                sum += len(token) -1
            else:
                if '##' in token:
                    sum += len(token) - 2
                else:
                    sum += len(token) +1
            sum_of_token_start_index.append(sum)
        return sum_of_token_start_index

    def select_ner_list(self):
        list_of_ner_tag = [['직위/직책', '생활용품', '동물 몸의 일부분', '인원수', 'IT 서비스 용어', '증상/증세',
                            '동물 종류', '국가명', '가전디지털', '기후/날씨', '직업', '패션의류/잡화', '식재료',
                            '미디어', '음료', '사기관', '기간', '금액', '개수', '사건', '그룹', '행정구역',
                            '음식', '시설물', '시간', '개인', '약', '의료용어 및 약어', '의료부서', '공기관', '탈 것',
                            '문구/오피스', '온도', '언어', '방향', '나이', '자연', '순위(순서)', '식물 종류', '모양/형태',
                            '서적/문서', '화폐(통화)', '스포츠', '질병', '학문/이론', '홈인테리어'],
                            ['인사말', '승락', '예약', '거절', '식사', '연락', '주문', '부서', '게임행사', '유전자', '금속',
                            '암석', '높이', '원소','IT하드웨어 용어', '천체명칭', '출산/유아동', '민족', '화학', '반려동물 용품',
                            '여행상품'],
                            ['사람','계절','기간','날짜','시간','천체명칭','시설물/건물','행정구역','자연','문화재','도로','국가명','음식',
                            '음료','문화/문명','언어','화폐(통화)','스포츠','게임','직업/직책','패션의류/잡화','수','조직/기관','제작물/작품',
                            '교통수단','제품','사건','동물 종류','동물 신체','모양 및 형태 표현','질병/증상/증세','IT 용어','의료용어 및 약어',
                            '기후/날씨','약','식물','학문/이론','세포/조직/기관','체내 분비물']]

        return list_of_ner_tag

    def bio_process(self, categories, index, token_sum_indexs):
        bio_tag, is_entity_b = [], True
        first_word_index = [x[0] for x in index]

        for i, word_index in enumerate(index):
            for j, tu in enumerate(token_sum_indexs):
                if len(index) - 1 >= i + 1:
                    if first_word_index[i + 1] <= tu: break
                if word_index[0] > tu:
                    if i == 0: bio_tag.append('O')
                    else: continue
                if word_index[0] <= tu and word_index[1] >= tu and is_entity_b:
                    bio_tag.append('B-' + categories[i])
                elif word_index[0] <= tu and word_index[1] >= tu and is_entity_b is False:
                    bio_tag.append('I-' + categories[i])
                elif word_index[1] < tu: bio_tag.append('O')
                else: continue
                is_entity_b = False
            is_entity_b = True

        list_of_ner_ids, list_of_ner_label_tmp = [], []
        for i in bio_tag:
            try:
                list_of_ner_ids.append(self.ner2index[i])
                list_of_ner_label_tmp.append(i)
            except KeyError:
                list_of_ner_ids.append(self.ner2index["O"])
                list_of_ner_label_tmp.append("O")


        return list_of_ner_label_tmp
    
