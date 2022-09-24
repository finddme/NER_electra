
MONGO_HOST = ''
MONGO_PORT = '27017'
MONGO_ID = 'admin'
MONGO_PWD = ''
MONGO_DBNAME = 'NER'


MONGO_COLLECTION = 'a'
MONGO_COLLECTION2 = 'b'

DICT_DIR = 'data_utils'
TEST_RESULT = 'test_result'

MONGO_DB_SAVE = True
TESTSET_SAVE = True
DATA_SPLITER = True

MODEL_CONFIG= {
  "hidden_size": 768,
  "vocab_size" : 8002,
  "max_len" : 128,
  "epochs": 9999,
  "batch_size": 128,
  "dropout": 0.1,
  "learning_rate": 3e-5,
  "warmup_proportion": 0.1,
  "gradient_accumulation_steps": 1,
  "summary_step": 100,
  "adam_epsilon": 1e-8,
  "warmup_steps": 0,
  "max_grad_norm": 1.0,
  "logging_steps": 100,
  "evaluate_during_training": True,
  "save_steps": 1,
  "output_dir": "data_utils/ckpoint",
  "max_steps": 999999999999999999,
  "model": "monologg/koelectra-base-v3-discriminator"
}