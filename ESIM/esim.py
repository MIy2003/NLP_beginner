# %%
# natural language inference
import numpy as np
import re
import os


def read_snli(data_dir, lowercase=False, istrain=True):
    def extract_text(text):
        text = re.sub(r'\(|\)', '', text)
        text = re.sub(r'\s+', ' ', text)
        if lowercase:
            text = text.lower()

        return text.strip()

    label_set = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt' if istrain else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


# %%
train_data = read_snli('./snli_1.0/snli_1.0', istrain=True)
test_data = read_snli('./snli_1.0/snli_1.0', istrain=False)
print('train_data:', len(train_data[0]), len(train_data[1]), len(train_data[2]))
print('test_data:', len(test_data[0]), len(test_data[1]), len(test_data[2]))
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)

# %%
# data preprocessing
from collections import Counter


class Vocab:
    def __init__(self, data, oov_token, max_size=None):
        self.token_to_idex = {}
        self.idx_to_token = {}
        self.max_size = max_size
        self.oov_token = oov_token
        self.oov_idx = 1
        self.token_to_idex[oov_token] = self.oov_idx
        self.idx_to_token[self.oov_idx] = oov_token
        self.idx_to_token[0] = '_PAD_'
        self.token_to_idex['_PAD_'] = 0
        self.idx_to_token[2] = '_BOS_'
        self.token_to_idex['_BOS_'] = 2
        self.idx_to_token[3] = '_EOS_'
        self.token_to_idex['_EOS_'] = 3
        self._build_vocab(data)

    def _build_vocab(self, data):
        tokens = []
        for text in data:
            tokens.extend(text.split())
        token_counter = Counter(tokens)
        if self.max_size is not None:
            token_counter = token_counter.most_common(self.max_size)
        else:
            token_counter = token_counter.most_common()
        token_counter_sorted = sorted(token_counter, key=lambda x: x[1], reverse=True)
        for token, _ in token_counter_sorted:
            self._add_token(token)

    def _add_token(self, token):
        if token not in self.token_to_idex:
            idx = len(self.token_to_idex)
            self.token_to_idex[token] = idx
            self.idx_to_token[idx] = token

    def __len__(self):
        return len(self.token_to_idex)

    def __getitem__(self, token):
        if token in self.token_to_idex:
            return self.token_to_idex[token]
        else:
            return self.oov_idx

    def _to_idx(self, token):
        if token in self.token_to_idex:
            return self.token_to_idex[token]
        else:
            return self.oov_idx

    def to_token(self, idx):
        return self.idx_to_token[idx]

    def encode(self, text):
        return [self._to_idx(token) for token in text.split()]

    def decode(self, idxs):
        return ' '.join([self.idx_to_token[idx] for idx in idxs])


data = train_data[0] + train_data[1]
vocab = Vocab(data, '_UNK_')
print('vocab size:', len(vocab))


def embedding_matrix(vocab, emb_file):
    emb_dim = None
    embedding = {}
    with open(emb_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.split()
            if emb_dim is None:
                emb_dim = len(line) - 1
                emb_matrix = np.zeros((len(vocab), emb_dim))
            try:
                float(line[1])
                if line[0] in vocab.token_to_idex:
                    embedding[line[0]] = np.array(line[1:], dtype=np.float32)
            except ValueError:
                # print('multiple words separated by spaces.')
                continue
    for token, idx in vocab.token_to_idex.items():
        if token in embedding:
            emb_matrix[idx] = embedding[token]
        else:
            if token == '_UNK_':
                # Out of vocabulary words are initialised with random gaussian samples.
                emb_matrix[idx] = np.random.normal(size=(emb_dim,))
    return emb_matrix


emb_matrix = embedding_matrix(vocab, './glove.840B.300d.txt')
print('emb_matrix shape:', emb_matrix.shape)

# %%
# print(emb_matrix[3])

# %%
from torch.utils.data import Dataset, DataLoader
import torch


class SNLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, vocab, padding_idx=0, premise_maxlen=None, hypothesis_maxlen=None):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = vocab.encode
        self.num_sequences = len(self.premises)
        self.premises_lengths = [len(seq.split(' ')) for seq in premises]
        self.premise_maxlen = premise_maxlen
        if self.premise_maxlen is None:
            self.premise_maxlen = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq.split(' ')) for seq in hypotheses]
        self.hypothesis_maxlen = hypothesis_maxlen
        if self.hypothesis_maxlen is None:
            self.hypothesis_maxlen = max(self.hypotheses_lengths)
        self.data = {"ids": list(range(self.num_sequences)),
                     "premises": torch.ones((self.num_sequences,
                                             self.premise_maxlen),
                                            dtype=torch.long) * padding_idx,
                     "hypotheses": torch.ones((self.num_sequences,
                                               self.hypothesis_maxlen),
                                              dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(labels, dtype=torch.long)}

        for i in range(self.num_sequences):
            premise = self.tokenizer(self.premises[i])
            end = min(len(premise), self.premise_maxlen)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])
            hypothesis = self.tokenizer(self.hypotheses[i])
            end = min(len(hypothesis), self.hypothesis_maxlen)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        return {"id": self.data["ids"][idx],
                "premise": self.data["premises"][idx],
                "premise_length": min(self.premises_lengths[idx],
                                      self.premise_maxlen),
                "hypothesis": self.data["hypotheses"][idx],
                "hypothesis_length": min(self.hypotheses_lengths[idx],
                                         self.hypothesis_maxlen),
                "label": self.data["labels"][idx]}


train_dataset = SNLIDataset(train_data[0], train_data[1], train_data[2], vocab, padding_idx=0, premise_maxlen=25,
                            hypothesis_maxlen=25)
test_dataset = SNLIDataset(test_data[0], test_data[1], test_data[2], vocab, padding_idx=0, premise_maxlen=25,
                           hypothesis_maxlen=25)


class SNLIDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield batch


train_dataloader = SNLIDataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = SNLIDataLoader(test_dataset, batch_size=32, shuffle=False)
data = next(iter(train_dataloader))

# %%
print(len(train_dataloader))

# %%
# Enhanced LSTM for Natural Language Inference
#copy from AllenNlP
import torch
from torch import nn
from utils import get_mask, masked_softmax, sort_by_seq_lens, weighted_sum, replace_masked


class VariationalDropout(nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning" (https://arxiv.org/abs/1506.02142) to a
    3D tensor.
    This module accepts a 3D tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
    and samples a single dropout mask of shape ``(batch_size, embedding_dim)`` and applies
    it to every time step.
    """

    def forward(self, input_tensor):
        """
        Apply dropout to input tensor.
        Parameters
        ----------
        input_tensor: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
        Returns
        -------
        output: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)`` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor


class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.
    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase), \
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.
        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        sorted_batch, sorted_lengths, _, restoration_idx = \
            sort_by_seq_lens(sequences_batch, sequences_lengths)
        sorted_lengths = sorted_lengths.to('cpu')
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs


class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.
    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.
        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                              .contiguous())

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2)
                                       .contiguous(),
                                       premise_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch,
                                         prem_hyp_attn,
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           hyp_prem_attn,
                                           hypothesis_mask)

        return attended_premises, attended_hypotheses


# %%
# %%
class Cofig(object):
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = 3
        self.vocab_size = 42394
        self.max_gradient_norm = 10
        self.num_epochs = 5
        self.batch_size = 32
        self.pad_size = 32
        self.learning_rate = 4e-4
        self.patience = 10
        self.embed_dim = 300
        self.embed_matrix = torch.tensor(emb_matrix, dtype=torch.float32)
        self.hidden_size = 300
        self.num_layers = 1  # 50
        self.checkpoint = './data/checkpoints/esim_5.pth.tar'
        self.target_dir = './data/checkpoints'


config = Cofig()


# %%
def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[config.hidden_size:(2 * config.hidden_size)] = 1.0


class ESIM(nn.Module):
    def __init__(self, cofig):
        super(ESIM, self).__init__()
        '''
        Input Encoding Layer:
        The input encoding layer is a biLSTM that encodes the premise and
        hypothesis sequences into a fixed-length vector representation.
        Let x1 and x2 be the premise and hypothesis sequences, respectively.
        The input encoding layer is defined as:
        x1 = BiLSTM(x1)
        x2 = BiLSTM(x2)

        Local Inference Modeling Layer:
        Locality of inference: is modeled by computing the soft attention
        between the premise and hypothesis sequences. The soft attention
        between the premise and hypothesis sequences is computed as:
        atten_weight(x1, x2) = torch.bmm(x1, x2.transpose(1, 2))

        Local inference collected over sequences: is modeled by using the
        softmax to compute the attention weights between the premise and
        hypothesis sequences. The attention weights are then used to compute
        the weighted sum of the premise and hypothesis sequences:
        weight1=softmax(atten_weight(x1, x2),dim=-1)
        x1_align = torch.bmm(weight1, x2.transpose(1, 2)))
        weight2=softmax(atten_weight(x2, x1),dim=-1)
        x2_align = torch.bmm(weight2, x1.transpose(1, 2)))

        Enhancement of local inference information:
        The local inference information is enhanced by concatenating thepremise and hypothesis 
        sequences respectively with their encoding sequence,soft_aligned sequence , and computing
         the element-wise subtraction and multiplication between them:
        x1_enhance = torch.cat([x1, x1_align, x1-x1_align, x1*x1_align], dim=-1)
        x2_enhance = torch.cat([x2, x2_align, x2-x2_align, x2*x2_align], dim=-1)

        Inference Composition Layer:
        The inference composition layer is a biLSTM that encodes the enhanced
        premise and hypothesis sequences which are already projected by MLP into 
        a fixed-length vector representation.
        x1_proj = MLP(x1_enhance)
        x2_proj = MLP(x2_enhance)
        x1_hidden = BiLSTM(x1_proj)
        x2_hidden = BiLSTM(x2_proj)

        Pooling Layer:
        The pooling layer computes the max and average pooling of the premise
        and hypothesis sequences,and concatenates them to form the final representation:
        x1_max = torch.max(x1_hidden, dim=1)
        x1_avg = torch.mean(x1_hidden, dim=1)
        x2_max = torch.max(x2_hidden, dim=1)
        x2_avg = torch.mean(x2_hidden, dim=1)

        Final multilayer perceptron (MLP) classifier Layer:
        has a hidden layer with tanh activation and softmax output layer
        '''

        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0, _weight=config.embed_matrix)
        self.rnn_dropout = VariationalDropout(config.dropout)
        self.encoding = Seq2SeqEncoder(nn.LSTM, config.embed_dim, config.hidden_size, config.num_layers,
                                       bidirectional=True)
        # self.encoding = nn.LSTM(config.embed_dim, config.hidden_size, config.num_layers,
        #                        bias=False, dropout=config.dropout, bidirectional=True)
        self.attention = SoftmaxAttention()
        self.projection = nn.Sequential(nn.Linear(config.hidden_size * 8, config.hidden_size),
                                        nn.ReLU())
        # nn.Dropout(config.dropout))
        self.composition = Seq2SeqEncoder(nn.LSTM, config.hidden_size, config.hidden_size, config.num_layers,
                                          bidirectional=True)
        # self.composition = nn.LSTM(config.hidden_size, config.hidden_size, config.num_layers,
        # bias=False, dropout=config.dropout, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(config.dropout),
                                        nn.Linear(config.hidden_size * 8, config.hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(config.dropout),
                                        nn.Linear(config.hidden_size, config.num_classes),
                                        )
        self.apply(_init_esim_weights)

    def forward(self, premise, hypothesis, premises_lengths, hypotheses_lengths):
        x1 = premise
        x2 = hypothesis
        x1_mask = get_mask(x1, premises_lengths).to(config.device)
        x2_mask = get_mask(x2, hypotheses_lengths).to(config.device)

        x1 = self.embed(x1)
        x2 = self.embed(x2)
        x1 = self.rnn_dropout(x1)
        x2 = self.rnn_dropout(x2)
        x1 = self.encoding(x1, premises_lengths)
        x2 = self.encoding(x2, hypotheses_lengths)
        # atten_weight=torch.bmm(x1,x2.transpose(1,2))
        # weight1=nn.functional.softmax(atten_weight,dim=-1)
        # weight2=nn.functional.softmax(atten_weight.transpose(1,2),dim=-1)

        # weight1,weight2=self.attention(x1,x1_mask,x2,x2_mask)
        # x1_align=torch.bmm(weight1,x2.transpose(1,2))
        # x2_align=torch.bmm(weight2,x1.transpose(1,2))
        x1_align, x2_align = self.attention(x1, x1_mask, x2, x2_mask)
        x1_enhance = torch.cat([x1, x1_align, x1 - x1_align, x1 * x1_align], dim=-1)
        x2_enhance = torch.cat([x2, x2_align, x2 - x2_align, x2 * x2_align], dim=-1)
        x1_proj = self.projection(x1_enhance)
        x2_proj = self.projection(x2_enhance)
        x1_proj = self.rnn_dropout(x1_proj)
        x2_proj = self.rnn_dropout(x2_proj)
        x1_hidden = self.composition(x1_proj, premises_lengths)
        x2_hidden = self.composition(x2_proj, hypotheses_lengths)
        x1_avg = torch.sum(x1_hidden * x1_mask.unsqueeze(1)
                           .transpose(2, 1), dim=1) \
                 / torch.sum(x1_mask, dim=1, keepdim=True)
        x2_avg = torch.sum(x2_hidden * x2_mask.unsqueeze(1)
                           .transpose(2, 1), dim=1) \
                 / torch.sum(x2_mask, dim=1, keepdim=True)

        x1_max, _ = replace_masked(x1_hidden, x1_mask, -1e7).max(dim=1)
        x2_max, _ = replace_masked(x2_hidden, x2_mask, -1e7).max(dim=1)
        x = torch.cat([x1_avg, x1_max, x2_avg, x2_max], dim=1)
        logits = self.classifier(x)
        probs = nn.functional.softmax(logits, dim=-1)

        return logits, probs

    def loss(self, logits, labels):
        # Predicted unnormalized scores(often referred to as logits)
        return nn.functional.cross_entropy(logits, labels)


# %%

import torch.optim as optim
from tqdm import tqdm
from utils import correct_predictions
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def train(model, train_iter, test_iter, config):
    device = config.device
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=0)
    tqdm_batch_iter = tqdm(train_iter)
    train_losses = []
    best_score = 0.0
    patience_counter = 0
    start_epoch = 1
    if not os.path.exists(config.target_dir):
        os.makedirs(config.target_dir)
    if os.path.exists(config.checkpoint):
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_losses = checkpoint["train_losses"]
    for epoch in range(start_epoch, config.num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        for i, batch in enumerate(tqdm_batch_iter):
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits, probalities = model(premises, hypotheses, premises_lengths, hypotheses_lengths)
            loss = model.loss(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
            optimizer.step()

            running_loss += loss.item()
            correct_preds += correct_predictions(probalities, labels)
            description = "Epoch:{},batch:{},loss:{:.4f},acc:{:.4f}".format(epoch, i, running_loss / (i + 1),
                                                                            correct_preds / (
                                                                                    (i + 1) * config.batch_size))
            tqdm_batch_iter.set_description(description)

        epoch_loss = running_loss / len(train_iter)
        epoch_acc = correct_preds / len(train_iter.dataset)

        train_losses.append(epoch_loss)
        scheduler.step(epoch_acc)
        if epoch_acc < best_score:
            patience_counter += 1
        else:
            patience_counter = 0
            best_score = epoch_acc
            # save model
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "train_losses": train_losses},
                       os.path.join(config.target_dir, "best.pth.tar"))
        if patience_counter > config.patience:
            print("early stop")
            break
        # Save the model at each epoch.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "train_losses": train_losses},
                   os.path.join(config.target_dir, "esim_{}.pth.tar".format(epoch)))
        if os.path.exists(os.path.join(config.target_dir, 'esim_5.pth.tar')):
            config.checkpoint = os.path.join(config.target_dir, 'esim_5.pth.tar')
    return train_losses
def eval(test_iter,cofig):
    if os.path.exists(os.path.join(config.target_dir, 'esim_5.pth.tar')):
        config.checkpoint = os.path.join(config.target_dir, 'esim_5.pth.tar')
    if os.path.exists(config.checkpoint):
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint["model"])
    model.eval()
    tqdm_batch_iter = tqdm(test_iter)
    device=cofig.device
    correct_preds = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm_batch_iter):
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)
            logits, probalities = model(premises, hypotheses, premises_lengths, hypotheses_lengths)
            correct_preds += correct_predictions(probalities, labels)
            description = "batch:{},acc:{:.4f}".format(i, correct_preds / ((i + 1) * config.batch_size))
            tqdm_batch_iter.set_description(description)
        test_acc = correct_preds / len(test_iter.dataset)

    print("test acc:{:.4f}".format(test_acc))
    return  test_acc
def visualization(config):
    if os.path.exists(os.path.join(config.target_dir, 'esim_5.pth.tar')):
        config.checkpoint = os.path.join(config.target_dir, 'esim_5.pth.tar')
    if os.path.exists(config.checkpoint):
        checkpoint = torch.load(config.checkpoint)
        train_losses = checkpoint["train_losses"]
        sns.set()
        plt.plot(train_losses, label='train_loss')
        #plt.set_title('TRAINLOSS')
        plt.xlabel('epoch');plt.ylabel('train_loss')
        plt.savefig('./train.png')



# %%
model = ESIM(config).to(config.device)
#train(model, train_dataloader, test_dataloader, config) 
# #to skip trainning and use the pretrained model esim_5.pth.tar(cause it takes too long and needs RTX3090 to train),you can change it if you want to train it from scratch
eval(test_dataloader,config)
visualization(config)