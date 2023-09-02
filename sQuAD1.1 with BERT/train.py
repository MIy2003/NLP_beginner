import torch
from model import BertForQuestionAnswering
from data_process import Preprocessing
from transformers import BertTokenizer, get_scheduler
from tqdm import tqdm
import os
import logging
import collections
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
torch.manual_seed(42)

class Config():
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 12
        self.max_seq_length = 384
        self.max_query_length = 64
        self.doc_stride = 128
        self.learning_rate = 3.5e-5
        self.epochs = 2
        self.num_labels = 2
        self.hidden_size = 768
        self.n_best_size = 10
        self.max_answer_length = 30
        self.bert_pretrained_model_dir = 'bert-base-cased'
        self.model_save_dir = 'checkpoints'
        self.output_prediction_file = 'squad1.1/predictions.json'
        self.output_nbest_file = 'nbest_predictions.json'
        # logger_init(log_file_name='qa', log_level=logging.DEBUG,
        # log_dir = self.logs_save_dir)




config = Config()
tokenizer = BertTokenizer.from_pretrained(config.bert_pretrained_model_dir)
squaddataset = Preprocessing(batch_size=config.batch_size,
                             doc_stride=config.doc_stride,
                             max_query_length=config.max_query_length,
                             max_seq_length=config.max_seq_length,
                             tokenizer=tokenizer,
                             n_best_size=config.n_best_size,
                             max_answer_length=config.max_answer_length)
train_iter, test_iter = squaddataset.load_data('./squad1.1/dev-v1.1.json', './squad1.1/train-v1.1.json')
if not os.path.exists(config.model_save_dir):
    os.makedirs(config.model_save_dir)

logger.info("  Num steps = %d",len(train_iter))
def train(config):
    model = BertForQuestionAnswering(config,config.bert_pretrained_model_dir)
    model.to(config.device)
    #torch.save(model.state_dict(),os.path.join(config.model_save_dir, 'test.pth'))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_scheduler(name='linear',
                                 optimizer=optimizer,
                                 num_warmup_steps=int(len(train_iter) * 0),
                                 num_training_steps=int(config.epochs * len(train_iter)))
    for epoch in range(config.epochs):
        total_loss = 0
        total_acc = 0
        tqdm_train_iter = tqdm(train_iter)
        for index, (batch_input, batch_seg, batch_label, batch_mask, _, _, _) in enumerate(tqdm_train_iter):
            batch_input = batch_input.to(config.device)
            batch_seg = batch_seg.to(config.device)
            batch_label = torch.tensor(batch_label,dtype=torch.long)
            batch_label = batch_label.to(config.device)
            batch_mask = batch_mask.to(config.device)


            loss,start_logits,end_logits= model(input_ids=batch_input,
                         attention_mask=batch_mask,
                         token_type_ids=batch_seg,
                         position_ids=None,
                         start_positions=batch_label[:, 0],
                         end_positions=batch_label[:, 1])

            #loss=outputs.loss
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            acc_start = (start_logits.argmax(1) == batch_label[:, 0]).float().mean()
            acc_end = (end_logits.argmax(1) == batch_label[:, 1]).float().mean()
            acc = (acc_start + acc_end) / 2
            total_acc += acc
            description = "Epoch:{},batch:{},loss:{:.4f},acc:{:.4f}".format(epoch + 1, index + 1,
                                                                            total_loss / (index + 1),
                                                                            total_acc / (index + 1))
            tqdm_train_iter.set_description(description)

    torch.save(model.state_dict(), os.path.join(config.model_save_dir, 'squad.pth'))


def evaluate(test_iter, config, inference):
    model = BertForQuestionAnswering(config, config.bert_pretrained_model_dir)
    model.to(config.device)
    checkpoint_path = os.path.join(config.model_save_dir, 'squad.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    model.eval()


    with torch.no_grad():
        all_results = []
        tqdm_test_iter = tqdm(test_iter)
        for index, (batch_input, batch_seg, _, batch_mask, _, batch_feature_id, _) in enumerate(
                tqdm_test_iter):
            batch_input = batch_input.to(config.device)
            batch_seg = batch_seg.to(config.device)
            batch_mask = batch_mask.to(config.device)
            start_logits, end_logits = model(input_ids=batch_input, attention_mask=batch_mask,
                                             token_type_ids=batch_seg, position_ids=None)


            _Result = collections.namedtuple(  # pylint: disable=invalid-name
                "Result", ["unique_id", "start_logits", "end_logits"])
            for i,unique_id in enumerate(batch_feature_id):
                start_logit = start_logits[i].detach().cpu().tolist()
                end_logit = end_logits[i].detach().cpu().tolist()
                all_results.append(_Result(unique_id=unique_id,
                                    start_logits=start_logit,
                                    end_logits=end_logit))
            #acc_sum_start = (start_logits.argmax(1) == batch_label[:, 0]).float().sum().item()
            #acc_sum_end = (end_logits.argmax(1) == batch_label[:, 1]).float().sum().item()
            #acc_sum += (acc_sum_start + acc_sum_end)
            #n += len(batch_label)
            #description = "Batch:{},acc:{:.4f}".format(index + 1, acc_sum / (2 * n))
            #tqdm_test_iter.set_description(description)
        if inference:
            squaddataset.write_predctions(all_results=all_results,
                                              output_prediction_file=config.output_prediction_file,
                                              output_nbest_file=config.output_nbest_file)


if __name__ =='__main__':
    config = Config()
    if not os.path.exists(os.path.join(config.model_save_dir, 'squad.pth')):
        train(config)
    evaluate(test_iter,config,inference =True)


