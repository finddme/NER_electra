class DecoderFromNamedEntitySequence():
  def __init__(self, tokenizer, index_to_ner):
    self.tokenizer = tokenizer
    self.ner_to_index = index_to_ner

  def __call__(self, list_of_input_ids, list_of_pred_ids, max_len):
    for i in list_of_input_ids:
      input_token = self.tokenizer.convert_ids_to_tokens(i)

    
    pred_ner_tag = [self.ner_to_index[pred_id] for pred_id in list_of_pred_ids[0]]
    self.index_to_ner = {v: k for k, v in self.ner_to_index.items()}
    # ----------------------------- parsing list_of_ner_word ----------------------------- #
    list_of_ner_word = []
    list_of_ner = []
    index = []
    s_tok = ['[CLS]','[SEP]','[PAD]','[UNK]']
    ner_tag_cnt = -1
    entity_word, entity_tag = "", ""
    
    for i, pred_ner_tag_str in enumerate(pred_ner_tag[:max_len]):
      if "B-" in pred_ner_tag_str:
        entity_tag = pred_ner_tag_str
        entity_word = input_token[i]
        if entity_word not in s_tok:
          list_of_ner.append({"word": entity_word.replace("▁", " "), "tag": entity_tag[2:], "key":self.index_to_ner[entity_tag], "index":[i-1]})
          ner_tag_cnt += 1
      try:
        if "I-" in pred_ner_tag_str:
          list_of_ner[ner_tag_cnt]['index'] += [i-1]
      except IndexError:
        pass

    # ----------------------------- parsing decoding_ner_sentence ----------------------------- #
    decoding_ner_sentence = ""
    word = ""
    self.sentence_result = []


    for i, (token_str, pred_ner_tag_str) in enumerate(zip(input_token, pred_ner_tag)):
      if i == 0 or i == len(pred_ner_tag):  # remove [CLS], [SEP]
        continue
      self.sentence_result.append(token_str)
      token_str = token_str.replace('##', '')  # '▁' 토큰을 띄어쓰기로 교체

      if 'B-' in pred_ner_tag_str:
        word += '▁'
        word += token_str

      elif 'I-' in pred_ner_tag_str:
        word += token_str
    del input_token[0]
    del input_token[-1]
    del pred_ner_tag[0]
    del pred_ner_tag[-1]

    words = word.split('▁')
    words = words[1:]

    cnt = 0
    tags = 0 #ner tag 개수
    for i in range(len(input_token)):
      try:
        if "I-" in pred_ner_tag[i] and 'O' in pred_ner_tag[i-1]: #O다음 I를 예측한경우
          pass
        
        if len(list_of_ner[tags]['index']) > 1 and "B-" in pred_ner_tag[i]: #b,i 태그 둘다 있을 경우
          str_len = 0

          for num in list_of_ner[tags]['index']:
            str_len += len(input_token[num])

          list_of_ner[tags]['index'] = [cnt, cnt + str_len - 1]
          cnt = cnt + str_len
          tags += 1

        elif len(list_of_ner[tags]['index']) == 1 and "B-" in pred_ner_tag[i]: #b태그만 있을경우
          list_of_ner[tags]['index'] = [cnt, cnt + len(input_token[i]) - 1]
          tags += 1
          cnt += len(input_token[i])

        elif "O" in pred_ner_tag[i] :
          cnt += len(input_token[i])

      except IndexError:
        pass

    for i in range(len(list_of_ner)):
      list_of_ner[i]['word'] = words[i]

    result = list_of_ner
    return result, decoding_ner_sentence

