from torch.nn import CrossEntropyLoss
from transformers import BertModel
import torch.nn as nn



class BertForQuestionAnswering(nn.Module):
    def __init__(self,config,bert_pretrained_model_dir=None):
        super(BertForQuestionAnswering,self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(bert_pretrained_model_dir,add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing

    def forward(self,input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                start_positions=None,
                end_positions=None):
        outputs= self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        sequence_output = outputs[0]
        # sequence_output: [ batch_size,seq_len,hidden_size]
        logits = self.qa_outputs(sequence_output)  # [batch_size,seq_len,2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp_(0, ignored_index)
            end_positions = end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss,start_logits,end_logits
        else:
            return start_logits,end_logits













