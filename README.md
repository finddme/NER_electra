# NER(Named Entity Recognition)
- 개체명으로 인식된 token에 대한 ner class(42종)를 예측하는 KoELECTRA기반 token classification fine-tune task

## Model information
- KoElectra(https://github.com/monologg/KoELECTRA/tree/024fbdd600e653b6e4bdfc64ceec84181b5ce6c4)
- version: KoELECTRA-Base-v3

## Environment
- ubuntu 20.04
- python 3.9.12
- docker image
```
docker pull ayaanayaan/ayaan_ner
```

## Requirements
- pytorch 1.10
- pymongo 4.1.1

## Data
- trainset : 305,560 문장
- validationset: 30,556 문장
- testset : 3,396 문장


## run_classify.py(Koelectra)
```
# Train
python main.py --op train --target_gpu (0/1/2) --ck_path (ck_path)

# Test
python main.py --op test --target_gpu (0/1/2) --load_ck (ck_path)
```

## api.py(Koelectra)
```
python main.py --op api --target_gpu (0/1/2) --load_ck (ck_path) --port (port)
```
