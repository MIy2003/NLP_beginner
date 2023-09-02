import os
import torch
import six
import math
def cache(func):
    def wrapper(*args, **kwargs):
        file_dir = kwargs['file_dir']
        postfix = kwargs['postfix']
        data_path = os.path.join(file_dir, f"{postfix}.pt")
        if not os.path.exists(data_path):
            logger.info(f"* cache file {data_path} not exists,regenerate buffer...")
            data = func(*args, **kwargs)
            torch.save(data, data_path)
        else:
            logger.info(f"* cache file {data_path} exists,load directly...")
            data = torch.load(data_path)
        return data
    return wrapper
from torch.utils.data import DataLoader,Dataset
import json
import collections
import torch
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO )

#format the text and generate tokens and char level offset
def get_offset(context):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    doc_tokens=[]
    char_to_word_offset=[]
    prev_is_whitespace=True
    for c in context:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes
def get_final_text(pred_text, orig_text, tokenizer):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.


    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        logger.warning(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        logger.warning("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        logger.warning("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        logger.warning("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

class Preprocessing (Dataset):
    def __init__(self,batch_size,doc_stride,max_query_length,max_seq_length,tokenizer,n_best_size,max_answer_length):
        super(Preprocessing,self).__init__()
        self.batch_size = batch_size
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    def read_squad_examples(self, filepath, is_training=True):
        with open(filepath,'r') as f:
            data=json.load(f)
            examples=[]
        for topic in data['data']:
            for paragraph in topic['paragraphs']:
                context=paragraph['context']
                context_tokens,word_offset=get_offset(context)
                for qa in paragraph['qas']:
                    question_text=qa['question']
                    qas_id=qa['id']
                    answer=qa['answers']
                    if is_training:
                        assert len(answer)==1
                        answer_offset = answer[0]['answer_start']
                        orig_answer_text = answer[0]['text']
                        answer_length = len(orig_answer_text)
                        start_position = word_offset[answer_offset]
                        end_position = word_offset[answer_offset + answer_length - 1]
                        actual_text = " ".join(
                             context_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(orig_answer_text.strip().split())
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("* Could not find answer: '%s' vs. '%s'",
                                            actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = None
                        end_position = None
                        orig_answer_text = None
                    examples.append({'qas_id':qas_id,'question_text':question_text,'orig_answer_text':orig_answer_text,
                                     'context_tokens':context_tokens,'start_position': start_position,'end_position':end_position})
        logger.info("  Num orig examples = %d", len(examples))
        return examples
    def _improve_answer_span(self,doc_tokens, input_start, input_end,
                             orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(self.tokenizer.tokenize(orig_answer_text))
        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)
    def _check_is_max_context(self,doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index


    @cache
    def convert_examples_to_features(self, file_path, is_training=False, postfix='cache', file_dir='./data'):
        logging.info(f"* using sliding window,doc_stride={self.doc_stride}...")
        examples = self.read_squad_examples(file_path, is_training)
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000
        all_feature = []
        for (example_index, example) in enumerate(examples):
            query_tokens = self.tokenizer.tokenize(example["question_text"])

            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[0:self.max_query_length]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example['context_tokens']):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            #if not is_training :
            #    tok_start_position = -1
            #    tok_end_position = -1
            if is_training :
                tok_start_position = orig_to_tok_index[example["start_position"]]
                if example["end_position"] < len(example['context_tokens']) - 1:
                    tok_end_position = orig_to_tok_index[example["end_position"] + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position,
                    example["orig_answer_text"])

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index,
                                                                split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids =self.tokenizer.convert_tokens_to_ids(tokens)
                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < self.max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length

                start_position = None
                end_position = None
                if is_training:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                if example_index < 20:
                    logger.info("*** Example ***")
                    logger.info("unique_id: %s" % (unique_id))
                    logger.info("example_index: %s" % (example_index))
                    logger.info("doc_span_index: %s" % (doc_span_index))
                    logger.info("tokens: %s" % " ".join(tokens))
                    logger.info("token_to_orig_map: %s" % " ".join([
                        "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                    logger.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                    ]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logger.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    if is_training :
                        answer_text = " ".join(tokens[start_position:(end_position + 1)])
                        logger.info("start_position: %d" % (start_position))
                        logger.info("end_position: %d" % (end_position))
                        logger.info(
                            "answer: %s" % (answer_text))
                feature=InputFeatures(unique_id=unique_id,
                                      example_index=example_index,
                                      doc_span_index=doc_span_index,
                                      tokens=tokens,
                                      token_to_orig_map=token_to_orig_map,
                                      token_is_max_context=token_is_max_context,
                                      input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      start_position=start_position,
                                      end_position=end_position)
                all_feature.append(feature)
                unique_id += 1
        logger.info("  Num split examples = %d", len(all_feature))
        return all_feature
    def write_predctions(self,all_results,output_prediction_file,output_nbest_file):
        """Write final predictions to the json file and log-odds of null if needed."""
        logging.info("Writing predictions to: %s" % (output_prediction_file))
        logging.info("Writing nbest to: %s" % (output_nbest_file))
        example_index_to_features = collections.defaultdict(list)

        all_examples = self.read_squad_examples('./squad1.1/dev-v1.1.json', is_training=False)
        all_features = self.convert_examples_to_features('./squad1.1/dev-v1.1.json', is_training=False,
                                                         postfix='test_128_384_64', file_dir='./data')
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        #scores_diff_json = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                start_indexes = _get_best_indexes(result.start_logits, self.n_best_size)
                end_indexes = _get_best_indexes(result.end_logits, self.n_best_size)
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self.max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index]))
                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)

                _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "NbestPrediction", ["text", "start_logit", "end_logit"])

                seen_predictions = {}
                nbest = []
                for pred in prelim_predictions:
                    if len(nbest) >= self.n_best_size:
                        break
                    feature = features[pred.feature_index]
                    if pred.start_index > 0:  # this is a non-null prediction
                        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                        orig_doc_start = feature.token_to_orig_map[pred.start_index]
                        orig_doc_end = feature.token_to_orig_map[pred.end_index]
                        orig_tokens = example['context_tokens'][orig_doc_start:(orig_doc_end + 1)]
                        tok_text = " ".join(tok_tokens)

                        # De-tokenize WordPieces that have been split off.
                        tok_text = tok_text.replace(" ##", "")
                        tok_text = tok_text.replace("##", "")

                        # Clean whitespace
                        tok_text = tok_text.strip()
                        tok_text = " ".join(tok_text.split())
                        orig_text = " ".join(orig_tokens)

                        final_text = get_final_text(tok_text, orig_text, self.tokenizer)
                        if final_text in seen_predictions:
                            continue

                        seen_predictions[final_text] = True
                    else:
                        final_text = ""
                        seen_predictions[final_text] = True

                    nbest.append(
                        _NbestPrediction(
                            text=final_text,
                            start_logit=pred.start_logit,
                            end_logit=pred.end_logit))
                    # In very rare edge cases we could have no valid predictions. So we
                    # just create a nonce prediction in this case to avoid failure.
                    if not nbest:
                        nbest.append(
                            _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

                    assert len(nbest) >= 1

                    total_scores = []
                    best_non_null_entry = None
                    for entry in nbest:
                        total_scores.append(entry.start_logit + entry.end_logit)
                        if not best_non_null_entry:
                            if entry.text:
                                best_non_null_entry = entry

                    probs = _compute_softmax(total_scores)

                    nbest_json = []
                    for (i, entry) in enumerate(nbest):
                        output = collections.OrderedDict()
                        output["text"] = entry.text
                        output["probability"] = probs[i]
                        output["start_logit"] = entry.start_logit
                        output["end_logit"] = entry.end_logit
                        nbest_json.append(output)

                    assert len(nbest_json) >= 1
                    all_predictions[example['qas_id']] = nbest_json[0]["text"]
                    all_nbest_json[example['qas_id']] = nbest_json
        if output_prediction_file:
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")

        if output_nbest_file:
            with open(output_nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    def generate_batch(self, data_batch):

        batch_input, batch_seg, batch_label, batch_qid = [], [], [], []
        batch_mask,batch_example_id, batch_feature_id, batch_map =[], [], [], []
        for item in data_batch:
            # item: unique_id,example_index,doc_span_index,tokens,token_to_orig_map,
            # token_is_max_context,input_ids,input_mask,segment_ids,start_position,end_position
            batch_example_id.append(item.example_index)
            batch_feature_id.append(item.unique_id)
            batch_input.append(item.input_ids)  # input_ids ,[max_len, batch_size]
            batch_seg.append(item.segment_ids)  # seg
            batch_label.append([item.start_position, item.end_position])  # ed
            #batch_qid.append(item.)  # qid
            batch_map.append(item.token_to_orig_map)  # ori_map
            batch_mask.append(item.input_mask)#input_mask

        batch_input = torch.tensor(batch_input)  # [max_len,batch_size]
        batch_seg = torch.tensor(batch_seg)  # [max_len, batch_size]
        #batch_label = torch.tensor(batch_label, dtype=torch.long)
        batch_mask = torch.tensor(batch_mask,dtype=torch.long)
        # [max_len,batch_size] , [max_len, batch_size] , [batch_size,2], [batch_size,], [batch_size,]
        return batch_input, batch_seg, batch_label,batch_mask, \
               batch_example_id, batch_feature_id, batch_map
    def load_data(self, test_file_path, train_file_path):
        doc_stride = str(self.doc_stride)
        max_seq_length = str(self.max_seq_length)
        max_query_length = str(self.max_query_length)
        postfix = doc_stride + '_' + max_seq_length + '_' + max_query_length
        test_data = self.convert_examples_to_features(file_path=test_file_path,
                                                      is_training=False,
                                                      postfix='test_'+postfix,
                                                      file_dir='./data')
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False,
                               collate_fn=self.generate_batch)
        train_data = self.convert_examples_to_features(file_path=train_file_path,
                                                       is_training=True,
                                                       postfix='train_'+postfix,
                                                       file_dir='./data')
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True,
                                collate_fn=self.generate_batch)
        return train_iter, test_iter


