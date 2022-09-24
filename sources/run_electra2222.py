import os, glob, re, logging, json, argparse, torch, wandb
from transformers import logging
from utils import init_logger, plot_loss_update
import numpy as np
from sources.model import KoelectraCRF, Koelectra_bilstm_CRF
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict
import config 
from transformers import ElectraModel, ElectraTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule
from sources.electra_create_dataset import NERDataset_electra
from sources.mongo_processor import Mongo
from sources.early_stopping import Early_Stopping
from sklearn.metrics import classification_report,multilabel_confusion_matrix, confusion_matrix
from sources.processor import ner_load_and_cache_examples as load_and_cache_examples
import operator
import pandas as pd
from sources.decoder import DecoderFromNamedEntitySequence
import datetime as dt

init_logger()
logger = logging.getLogger(__name__)

mongo_train = Mongo(host=config.MONGO_HOST,
                      port=config.MONGO_PORT,
                      id=config.MONGO_ID,
                      pwd=config.MONGO_PWD,
                      db_name=config.MONGO_DBNAME,
                      collection='22_NER_TRAIN',
                      mongo_uri="")

mongo_val = Mongo(host=config.MONGO_HOST,
                  port=config.MONGO_PORT,
                  id=config.MONGO_ID,
                  pwd=config.MONGO_PWD,
                  db_name=config.MONGO_DBNAME,
                  collection='22_NER_VAL',
                  mongo_uri="")

mongo_test = Mongo(host=config.MONGO_HOST,
                   port=config.MONGO_PORT,
                   id=config.MONGO_ID,
                   pwd=config.MONGO_PWD,
                   db_name=config.MONGO_DBNAME,
                   collection='22_NER_TEST',
                   mongo_uri="")

def run_ner(do_train,args):
    logger.info(do_train)
    tokenizer=ElectraTokenizer.from_pretrained(config.MODEL_CONFIG['model'])
    if do_train:
        train_dataset = NERDataset_electra(tokenizer=tokenizer, max_len=config.MODEL_CONFIG['max_len'], mongo=mongo_train, select=args.select)
        val_dataset = NERDataset_electra(tokenizer=tokenizer, max_len=config.MODEL_CONFIG['max_len'], mongo=mongo_val, select=args.select)
        ner2idx = train_dataset.ner2index
        train_dataloader = DataLoader(train_dataset, batch_size = config.MODEL_CONFIG['batch_size'], shuffle=False, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size = config.MODEL_CONFIG['batch_size'], shuffle=False, num_workers=0)
        
        model = Koelectra_bilstm_CRF(config=config.MODEL_CONFIG, num_classes=len(ner2idx), tokenizer= tokenizer)
        if torch.cuda.is_available():
            device = torch.device('cuda:' + str(args.target_gpu))
            logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            logger.info('We will use the GPU:{}'.format(torch.cuda.get_device_name(0)))
        else:
            device = torch.device("cpu")
            logger.info('No GPU available, using the CPU instead.')
        model.to(device)
        train(args, model, args.select, train_dataloader, val_dataloader,tokenizer, device,do_train=True)
    else:
        test_dataset = NERDataset_electra(tokenizer=tokenizer, max_len=config.MODEL_CONFIG['max_len'], mongo=mongo_test, select=args.select)
        ner2idx = test_dataset.ner2index
        test_dataloader = DataLoader(test_dataset, batch_size = config.MODEL_CONFIG['batch_size'], shuffle=False, num_workers=0)
        
        model = Koelectra_bilstm_CRF(config=config.MODEL_CONFIG, num_classes=len(ner2idx), tokenizer= tokenizer)
        if torch.cuda.is_available():
            device = torch.device('cuda:' + str(args.target_gpu))
            logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            logger.info('We will use the GPU:{}'.format(torch.cuda.get_device_name(0)))
        else:
            device = torch.device("cpu")
            logger.info('No GPU available, using the CPU instead.')
        model.to(device)
        evlauate(args, model, args.select, test_dataloader, device,tokenizer, do_train=False)



def train(args,model,select,train_dataloader,val_dataloader, tokenizer,device,do_train=True):
    wandb.init(project="NER", entity="ayaan")
    wandb.watch(model)
    t_total = len(train_dataloader) // config.MODEL_CONFIG['gradient_accumulation_steps'] * config.MODEL_CONFIG['epochs']  # gpu 메모리 효율
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    index_to_ner = {v: k for k, v in train_dataloader.dataset.ner2index.items()}
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                lr=config.MODEL_CONFIG['learning_rate'],
                eps=config.MODEL_CONFIG['adam_epsilon'])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config.MODEL_CONFIG['warmup_steps'], t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", config.MODEL_CONFIG['epochs'])
    logger.info("  Total train batch size = %d", config.MODEL_CONFIG['batch_size'])
    logger.info("  Gradient Accumulation steps = %d", config.MODEL_CONFIG['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", config.MODEL_CONFIG['logging_steps'])
    logger.info("  Save steps = %d", config.MODEL_CONFIG['save_steps'])

    global_step = 0
    early_stopping = Early_Stopping(verbose = True)
    model.zero_grad()
    mb = master_bar(range(int(config.MODEL_CONFIG['epochs'])))
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        count_correct = 0
        total_correct = 0
        logger.info('####################### Epoch {}: Training Start #######################'.format(epoch))
        for step, batch in enumerate(epoch_iterator):
            label_list,pred_list = [],[]
            model.train()
            x_input = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            y_real = batch['labels'].to(device, dtype=torch.long)

            outputs= model(x_input,attention_mask,y_real)

            loss = outputs[0]
            sequence_of_tags = outputs[1]
            train_acc = outputs[2]
            for r,s in zip(y_real.tolist(), sequence_of_tags):
                label_list += r
                pred_list += s

            train_f1=f1_score(label_list, pred_list, average='macro')

            if config.MODEL_CONFIG['gradient_accumulation_steps'] > 1:
                loss = loss / config.MODEL_CONFIG['gradient_accumulation_steps']

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MODEL_CONFIG['max_grad_norm'])
            
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            ckpt ="TRAIN_{}epoch_{}step_loss{}_acc{}.pt".format(epoch,step,loss,train_acc)
            ck_path = os.path.join(args.ck_path, ckpt)

            mb.write('------------------------------------------------[Train] Epoch {}/{}, Steps {}/{}------------------------'.format(epoch, config.MODEL_CONFIG['epochs'], step, len(train_dataloader)))
            mb.write('                                    train loss {}, train acc {}, train f1 {}'.format(loss,train_acc,train_f1))
            wandb.log({"(TRAIN)loss": loss, "epoch":epoch, "custom_step": step, "(TRAIN)accuracy": train_acc})
            if config.MODEL_CONFIG['save_steps'] > 0 and step % config.MODEL_CONFIG['save_steps'] == 0:
                early_stopping(loss,model, ck_path)

            if (step+1) % len(train_dataloader) == 0:
                if do_train:
                    validation(args, model, val_dataloader, epoch)
            if config.MODEL_CONFIG['max_steps'] > 0 and global_step > config.MODEL_CONFIG['max_steps']:
                break
        mb.write("###########Epoch {} done".format(epoch + 1))

        if config.MODEL_CONFIG['max_steps'] > 0 and global_step > config.MODEL_CONFIG['max_steps']:
            break


def validation(args, model, val_dataloader, epoch):
    # Eval!
    logger.info('####################### Epoch {}: Validation Start #######################\n'.format(epoch))
    logger.info("  Num examples = {}".format(len(val_dataloader)))
    device = torch.device('cuda:' + str(args.target_gpu))

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    step = 0

    for batch in progress_bar(val_dataloader):
        step +=1
        model.eval()

        with torch.no_grad():
            label_list,pred_list = [],[]
            x_input = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            y_real = batch['labels'].to(device, dtype=torch.long)

            outputs= model(x_input,attention_mask,y_real)

            loss = outputs[0]
            sequence_of_tags = outputs[1]
            for r,s in zip(y_real.tolist(), sequence_of_tags):
                label_list += r
                pred_list += s
            val_f1=f1_score(label_list, pred_list, average='macro')
            val_acc = outputs[2]
            mb = master_bar(range(step))
            mb.write('------------------------------------------------[Validation] Epoch {}, Steps {}/{}------------------------'.format(epoch, step, len(val_dataloader)))
            mb.write('                                    validataion loss {}, validataion acc {}, validataion f1 {}'.format(loss,val_acc,val_f1))
            wandb.log({"(Validation)loss": loss, "epoch":epoch, "custom_step": step, "(Validation)accuracy": val_acc})
        nb_eval_steps += 1

def evlauate(args, model, select, test_dataloader, device,tokenizer, do_train=False):
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.MODEL_CONFIG['learning_rate'],
                      eps=config.MODEL_CONFIG['adam_epsilon'])
    model.load_state_dict(torch.load(args.load_ck))
    model.eval()
    
    step = 0
    pred_list, labels_list,wwww,count_dict = [], [], [], {}

    with torch.no_grad():
        for batch in progress_bar(test_dataloader):
            step +=1
            x_input = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            y_real = batch['labels'].to(device, dtype=torch.long)

            outputs= model(x_input,attention_mask,y_real)

            loss = outputs[0]
            pred = outputs[1]
            eval_acc = outputs[2]    

            label = y_real.tolist()

            for l,p in zip(label,pred):
                labels_list += l
                pred_list += p
            
            sorted_ner_to_index = sorted(test_dataloader.dataset.ner2index.items(), key=operator.itemgetter(1))
            target_names = [] 
            for ner_tag, index in sorted_ner_to_index:
                if ner_tag in ['[CLS]', '[SEP]', '[PAD]', 'O']:
                    continue
                else:
                    target_names.append(ner_tag)
            label_index_to_print = list(range(4, len(test_dataloader.dataset.ner2index)))

            with open(config.DICT_DIR + '/ner_to_index_'+ args.select+ '.json', 'rb') as f:
                ner_to_index =  json.load(f)
                index_to_ner = {v: k for k, v in ner_to_index.items()}

            decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)

            for lab,pre in zip(label,pred):
                for la,pr in zip(lab,pre):
                    if la not in list(count_dict.keys()):
                        count_dict[la] = 0
                    if la == pr:
                        count_dict[la] += 1

            list_of_input_ids= x_input.tolist()
            for i in range(len(x_input)):
                w, d = decoder_from_res(list_of_input_ids=[list_of_input_ids[i]], list_of_pred_ids=[pred[i]], max_len = 64)

                l = ['word','tag','index']
                for ii in w:
                    ww= {key:value for key, value in ii.items() if key in l}
                    d= tokenizer.decode(list_of_input_ids[i]).replace("[PAD]","")
                    
                    d= d.replace("[SEP]","")
                    d= d.replace("[UNK]","")
                    d= d.replace("[CLS]","")
                    d = d.strip()
                    ww['sentence'] = d
                    wwww.append(ww)

            mb = master_bar(range(step))
            mb.write('------------------------------------------------[Evaluation] Steps {}/{}------------------------'.format(step, len(test_dataloader)))
            mb.write('                                         Evaluation acc {}'.format(eval_acc))
        
        ########################classification_report
        c = classification_report(y_true=labels_list, y_pred=pred_list,output_dict=True)
        for ik,iv in sorted_ner_to_index:
            for ck,cv in c.items():
                if ck == str(iv):
                    c[ik] = c.pop(ck)
        report_list,report_list2, report_dict= [],[],{}
        for k,v in c.items():
            report = {k:v}
            report_list.append(report)

        for i,r in enumerate(report_list):
            current_title = list(report_list[i].keys())[0][2:]
            if i != len(report_list)-1:
                post_title = list(report_list[i+1].keys())[0][2:]
            
            if i != len(report_list)-1 and current_title not in list(report_dict.keys()) and post_title == current_title:
                precision = (r[f"B-{current_title}"]['precision'] + report_list[i+1][f"I-{current_title}"]['precision'])/2
                recall = (r[f"B-{current_title}"]['recall'] + report_list[i+1][f"I-{current_title}"]['recall'])/2
                f1 = (r[f"B-{current_title}"]['f1-score'] + report_list[i+1][f"I-{current_title}"]['f1-score'])/2
                support = r[f"B-{current_title}"]['support']+ report_list[i+1][f"I-{current_title}"]['support']
                report_dict[current_title] = dict(precision=precision)
                report_dict[current_title]['recall'] = recall
                report_dict[current_title]['f1-score'] = f1
                report_dict[current_title]['support'] = support
            else:
                if current_title not in list(report_dict.keys()) and list(report_list[i].keys())[0][:2] == 'B-':
                    report_dict[current_title] = list(report_list[i].values())[0]
                elif current_title not in list(report_dict.keys()):
                    report_dict[list(report_list[i].keys())[0]] = list(report_list[i].values())[0]
                
            
            post_title = current_title
        report_list2.append(report_dict)
        df = pd.DataFrame(report_list2[0]).transpose()
        df = df.astype({'support':'int'})
        print(df)
        x = dt.datetime.now()
        ck_name = str(args.load_ck).replace('./checkpoints/','')
        ck_name = ck_name.replace('/','-')
        df.to_excel(os.path.join(config.TEST_RESULT,'{}_{}_{}.xlsx'.format(x.month, x.day, ck_name)))

    for ik,iv in sorted_ner_to_index:
        for ck,cv in count_dict.items():
            if ck == iv:
                count_dict[ik] = count_dict.pop(ck)
    count_dict2 = {}


    for b,v in count_dict.items():
        if b[2:] not in list(count_dict2.keys()):
            count_dict2[b[2:]] = v
        elif b[2:] in list(count_dict2.keys()):
            count_dict2[b[2:]] +=v

