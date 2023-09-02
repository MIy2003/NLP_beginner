# %%
import torch
from transformers import BertTokenizer
PRETRAINED_MODEL_NAME = "bert-base-cased"  # 指定英文 BERT-BASE 预训练模型
# 取得此预训练模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
print(tokenizer.tokenize("This is a simple example of a tokenized sentence."))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("This is a simple example of a tokenized sentence.")))
print("PyTorch version: ", torch.__version__)


# %%
#安装BertViz
import sys
#!test -d bertviz_repo || git clone https://github.com/jessevig/bertviz bertviz_repo
if not 'bertviz_repo' in sys.path:
  sys.path += ['bertviz_repo']



# %%
# import packages

from transformers import BertTokenizer, BertModel
from bertviz.bertviz import head_view

# 在 jupyter notebook 中显示 visualzation 的 helper
#@save
def call_html():
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))


# %%
model_version = 'bert-base-cased'
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version)

sentence_a="X asked Y to buy comic books ,"
sentence_b="or he will beat him."
inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
token_type_ids = inputs['token_type_ids']
input_ids = inputs['input_ids']
attention = model(input_ids, token_type_ids=token_type_ids)[-1]
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)
call_html()

# 交给 BertViz 视觉化
head_view(attention, tokens)


# %%
import os
def read_imdb(data_dir, is_train):
    """读取IMDb评论数据集文本序列和标签"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels
data_dir='./aclImdb/'
train_data = read_imdb(data_dir,is_train=True)
test_data = read_imdb(data_dir,is_train=False)
print('训练集数目:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('标签：', y, 'review:', x[0:60])


# %% [markdown]
# def get_corpus(data):
#     words = []
#     for i in data:
#         for j in i.split():
#             words.append(j.strip())
#     return words
# corpus = get_corpus(train_data[0])
# 

# %% [markdown]
# #预处理数据集
# #将每个单词作为一个词元，过滤掉出现次数不到5次的词元，从训练数据集中创建一个词表
# import collections
# import re
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# def tokenize(lines):  
#     """将文本行拆分为单词或字符词元"""
#     return [line.split() for line in lines]
# #train_corpus = [tokenize(line) for line in train_data[0]]
# def count_corpus(corpus):
#     """统计语料库中各词元频率"""
#     return collections.Counter(corpus)  
# def filter_corpus(corpus, min_count):
#     """过滤掉出现次数小于min_count的词元"""
#     return [[token for token in document if collections.Counter(corpus) [token] > min_count] for document in corpus]
#     
# def remove_stopwords(corpus, stopwords):
#     """过滤掉停用词"""
#     return [[token for token in document if token not in stopwords] for document in corpus]
# def remove_punctuation(corpus):
#     """过滤掉标点符号"""
#     return [[re.sub(r'[^\w\s]', '', token) for token in document] for document in corpus]
# def lemmatize_corpus(corpus):
#     """过滤掉不是英文单词的词元"""
#     lemmatizer = WordNetLemmatizer()
#     return [[lemmatizer.lemmatize(token) for token in document] for document in corpus]
# def preprocess_corpus(data, stopwords, min_count):
#     """预处理语料库"""
#     #corpus = remove_punctuation(train_data[0])
#     #corpus = tokenize(string(corpus))
#     corpus = remove_stopwords(data, stopwords)
#     corpus=lemmatize_corpus(corpus)
#     #corpus = filter_corpus(corpus, min_count)
#     return corpus
# train_corpus=preprocess_corpus(train_data[0], stopwords.words('english'), 5) 

# %%
#import nltk
#nltk.download('omw-1.4')

# %%

#截断或填充文本
MAX_LEN=256
train_tokens = [tokenizer.tokenize(line) for line in train_data[0][:10000]]
#train_input=[tokenizer.encode_plus(line) for line in train_data[0][3]]
#print(train_tokens[0])
#print(train_input[0])
#train_input_ids=[tokenizer.encode(sentences, add_special_tokens=True) for sentences in train_tokens]
#print(torch.tensor(train_input_ids).shape)
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps-1] + [102] # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充
num_steps = 256  # 序列长度

train_input_ids=([tokenizer.encode(sentences, add_special_tokens=True) for sentences in train_tokens])
train_input_ids = torch.tensor([truncate_pad(line, num_steps, 0) for line in train_input_ids])
print(train_input_ids.shape)
print(train_input_ids[0])


#train_input_ids=[tokenizer.encode(sentences, add_special_tokens=True, max_length=MAX_LEN) for sentences in train_data[0]]
#from torch.nn.utils.rnn import pad_sequence
#train_input_ids=pad_sequence(train_input_ids, maxlen=MAX_LEN,batch_first=True, padding_value=0)

test_tokens = [tokenizer.tokenize(line) for line in test_data[0][:10000]]
test_input_ids=([tokenizer.encode(sentences, add_special_tokens=True) for sentences in test_tokens])
test_input_ids = torch.tensor([truncate_pad(line, num_steps, 0) for line in test_input_ids])

# %%
#创建attention mask
train_attention_masks=[[int (train_input_id != 0) for train_input_id in lines ] for lines in train_input_ids]
train_attention_masks=torch.tensor(train_attention_masks)
print(train_attention_masks.shape)
print(train_attention_masks[0])

test_attention_masks=torch.tensor([[int (test_input_id != 0) for test_input_id in lines ] for lines in test_input_ids])

# %%
#由于本次任务每个训练集只有一个句子，所以不需要token_type_ids
#创建dataloader
from torch.utils.data import DataLoader, TensorDataset ,RandomSampler
batch_size=16
train_labels=torch.tensor(train_data[1][:10000])
train_data_=TensorDataset(train_input_ids,train_attention_masks,train_labels)
train_sampler=RandomSampler(train_data_)
train_dataloader=DataLoader(train_data_, sampler=train_sampler, batch_size=batch_size,num_workers=10)
print(len(train_dataloader))

test_labels=torch.tensor(test_data[1][:10000])
test_data_=TensorDataset(test_input_ids,test_attention_masks,test_labels)
test_sampler=RandomSampler(test_data_)
test_dataloader=DataLoader(test_data_, sampler=test_sampler, batch_size=batch_size,num_workers=10)

# %%
#创建模型
from transformers import BertForSequenceClassification, BertConfig,AdamW
model=BertForSequenceClassification.from_pretrained("bert-base-cased",num_labels=2,output_attentions=False,output_hidden_states=False)
model.cuda()
optimizer=AdamW(model.parameters(),lr=2e-5,eps=1e-8)
from transformers import get_linear_schedule_with_warmup
epochs=2
total_steps=len(train_dataloader) * epochs
scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)



# %%
#计算正确率
import numpy as np
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# %%
import random
import numpy as np
from torch.autograd import Variable
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
total_loss=0
for epoch in range(epochs):
    model.train()
    for step,(batch_input_ids,batch_attention_masks,batch_labels) in enumerate(train_dataloader):
        batch_input_ids=batch_input_ids.to(device)
        batch_attention_masks=batch_attention_masks.to(device)
        batch_labels=batch_labels.to(device)
        outputs=model(batch_input_ids,token_type_ids=None,attention_mask=batch_attention_masks,labels=batch_labels)
        loss=outputs[0]
        total_loss += loss.item()
        loss.backward()
        #梯度裁剪避免梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        scheduler.step()
    
        if step % 100 == 0 and not step == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch,step,loss.item()))
    print('Total loss {:.4f}'.format(total_loss))

# %%
#在测试集上验证模型

eval_accuracy=0
n_eval_steps=0
model.eval()
for batch in test_dataloader:
    batch_input_ids=batch[0].to(device)
    batch_attention_masks=batch[1].to(device)
    batch_labels=batch[2].to(device)
    with torch.no_grad():
        outputs=model(batch_input_ids,token_type_ids=None,attention_mask=batch_attention_masks)
    logits=outputs[0]
    
    n_eval_steps+=1
    eval_accuracy+=flat_accuracy(logits.detach().cpu().numpy(),batch_labels.cpu().numpy())
print('Accuracy:{}'.format(eval_accuracy/n_eval_steps))  



