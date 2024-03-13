import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#GPU 사용 시
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

raw_train = pd.read_csv("dataset/df_final_training2.csv")
raw_test = pd.read_csv("dataset/df_final_validation2.csv")

bertmodel, vocab = get_pytorch_kobert_model()

def transform_rawdataset(raw_df):
    df = raw_df[["type","HS01","HS02","HS03"]]
    df = df.fillna('')
    return df

df_train_raw = transform_rawdataset(raw_train)
df_test_raw = transform_rawdataset(raw_test)

emo = ["분노","슬픔","불안","상처","당황","기쁨"]
emo_label = [0,1,2,3,4,5]
df_train_raw["emotion"] = df_train_raw["type"]
df_test_raw["emotion"] = df_test_raw["type"]

for i in range(1,7):
    df_train_raw.loc[df_train_raw["type"].isin([f"E{j}"for j in range(10*i,20*i)]),"emotion"] = emo_label[i-1]
    df_test_raw.loc[df_test_raw["type"].isin([f"E{j}"for j in range(10*i,20*i)]),"emotion"] = emo_label[i-1]


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         

    def __len__(self):
        return (len(self.labels))
    

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=6,   ##클래스 수 조정##
                 dr_rate=0.2,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
    
def emo_predict(predict_sentence, max_len=64, batch_size=32):

    data = [predict_sentence, '0']
    dataset_another = [data]

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer,vocab,lower=False)
    
    model = BERTClassifier(bertmodel)
    model.load_state_dict(torch.load('./Fine-tuned-BERTClassifier/model.bin', map_location=torch.device('cpu')))

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
        print(pd.DataFrame(out.detach().tolist(),columns=emo))

        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("분노가")
            elif np.argmax(logits) == 1:
                test_eval.append("슬픔이")
            elif np.argmax(logits) == 2:
                test_eval.append("불한가")
            elif np.argmax(logits) == 3:
                test_eval.append("상처이")
            elif np.argmax(logits) == 4:
                test_eval.append("당황이")
            elif np.argmax(logits) == 5:
                test_eval.append("기쁨이")
 

        # return (">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")
        return test_eval[0]
        
def result(sub_obj, emo):
    
    SUBJECT = None
    OBJECT = None
    
    for pos in sub_obj:
        if pos[1] == 'SUBJECT':
            SUBJECT = pos[0]
        elif pos[1] == 'OBJECT':
            OBJECT = pos[0]
            
    return SUBJECT, OBJECT, emo[:-1]
    return f"{SUBJECT}(은)는 {OBJECT}(을)를 향해 {emo[:-1]}(을)를 느낍니다."