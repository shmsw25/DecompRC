# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import collections
import logging
import json
import math
import os
import random
import six
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import qa.tokenization as tokenization
from qa.modeling import BertConfig, BertClassifier, BertForQuestionAnswering, \
    BertForQuestionAnsweringWithKeyword
from qa.optimization import BERTAdam
from qa.multipara_prepro import get_dataloader
from hotpot_evaluate_v1 import normalize_answer

MODEL_DIR = "../DecompRC/model/"

class DecompRC(object):

    def __init__(self, batch_size=4):
        parser = argparse.ArgumentParser()
        parser.add_argument("--bert_config_file", default=MODEL_DIR+"uncased_L-12_H-768_A-12/bert_config.json")
        parser.add_argument("--vocab_file", default=MODEL_DIR+"uncased_L-12_H-768_A-12/vocab.txt")

        parser.add_argument("--init_checkpoint", type=str,
                            help="Initial checkpoint (usually from a pre-trained BERT model).", \
                            default=MODEL_DIR+"pytorch_model.bin")
        parser.add_argument("--do_lower_case", default=True, action='store_true',
                            help="Whether to lower case the input text. Should be True for uncased "
                                 "models and False for cased models.")
        parser.add_argument("--max_seq_length", default=300, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                                 "longer than this will be truncated, and sequences shorter than this will be padded.")
        parser.add_argument("--doc_stride", default=128, type=int,
                            help="When splitting up a long document into chunks, how much stride to take between chunks.")
        parser.add_argument("--max_query_length", default=64, type=int,
                            help="The maximum number of tokens for the question. Questions longer than this will "
                                 "be truncated to this length.")

        self.args = parser.parse_args()
        #self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bert_config = BertConfig.from_json_file(self.args.bert_config_file)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.args.vocab_file, do_lower_case=self.args.do_lower_case)

        #### Loading models....
        self.qa_model = []
        for checkpoint in ["model1.pt", "model2.pt", "model3.pt"]:
            self.qa_model.append(self.load(BertForQuestionAnswering(bert_config, 4),
                                           MODEL_DIR + "onehop/" + checkpoint))

        self.verifier_model = self.load(BertClassifier(bert_config, 2, "max"),
                                        MODEL_DIR + "scorer/best-model.pt")

        self.bridge_decomposer = self.load(BertForQuestionAnsweringWithKeyword(bert_config, 2),
                                           MODEL_DIR + "decom-bridge/model.pt")
        self.intersec_decomposer = self.load(BertForQuestionAnswering(bert_config, 2),
                                             MODEL_DIR + "decom-intersec/model.pt")
        self.batch_size = batch_size

    def load(self, model, checkpoint):
        state_dict = torch.load(checkpoint, map_location='cpu')
        filter = lambda x: x[7:] if x.startswith('module.') else x
        state_dict = {filter(k):v for (k,v) in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def get_output(self, model, question, paragraphs):
        dataloader, examples, eval_features = get_dataloader(model, question, paragraphs,
                                                            self.tokenizer, self.batch_size)
        if model=="qa":
            predictions = self.get_qa_prediction(dataloader, examples, eval_features)
            assert len(examples)==len(predictions)
            return predictions

        if model=="classifier":
            predictions = self.get_classifier_prediction(dataloader, examples, eval_features)
            return predictions

        if model=="span-predictor":
            bridge_subq = self.get_span_prediction(self.bridge_decomposer, dataloader, examples, eval_features, True)
            intersec_subq = self.get_span_prediction(self.intersec_decomposer, dataloader, examples, eval_features, False)
            return bridge_subq, intersec_subq

    def get_qa_prediction(self, dataloader, examples, eval_features):
        RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits", "switch"])
        all_results = []
        def _get_raw_results(model1):
            raw_results = []
            for batch in tqdm(dataloader, desc="Evaluating"):
                example_indices = batch[-1]
                batch_to_feed = [t.to(self.device) for t in batch[:-1]]
                with torch.no_grad():
                    batch_start_logits, batch_end_logits, batch_switch = model1(batch_to_feed)

                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].detach().cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                    switch = batch_switch[i].detach().cpu().tolist()
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    raw_results.append(RawResult(unique_id=unique_id,
                                                start_logits=start_logits,
                                                end_logits=end_logits,
                                                switch=switch))
            return raw_results

        all_raw_results = [_get_raw_results(m) for m in self.qa_model]
        for i in range(len(all_raw_results[0])):
            result = [all_raw_result[i] for all_raw_result in all_raw_results]
            assert all([r.unique_id == result[0].unique_id for r in result])
            start_logits = sum([np.array(r.start_logits) for r in result]).tolist()
            end_logits = sum([np.array(r.end_logits) for r in result]).tolist()
            switch = sum([np.array(r.switch) for r in result]).tolist()
            all_results.append(RawResult(unique_id=result[0].unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits,
                                         switch=switch))

        return get_qa_prediction(examples, eval_features, all_results)

    def get_classifier_prediction(self, dataloader, examples, eval_features):

        all_results = collections.defaultdict(list)
        all_results_per_key = collections.defaultdict(list)
        for batch in tqdm(dataloader, desc="Evaluating"):
            example_indices = batch[-1]
            batch_to_feed = tuple(t.to(self.device) for t in batch[:-1])
            with torch.no_grad():
                batch_predicted_label = self.verifier_model(batch_to_feed)
            for i, example_index in enumerate(example_indices):
                logit = batch_predicted_label[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                example = examples[eval_feature.example_index]
                all_results[example.qas_id].append( \
                                (logit, eval_feature.switch, example.all_answers,
                                 example.question_text))
        for example_index, results in all_results.items():
            example_index_, sent_index = example_index[:-2], example_index[-1]
            logit = [0, 0]
            switch = results[0][1]
            f1 = results[0][2]
            for (logit_, switch_, f1_, _) in results:
                logit[0] += logit_[0]
                logit[1] += logit_[1]
                assert switch == switch_ and f1 == f1_
            logit_indicator = (np.exp(logit)/sum(np.exp(logit))).tolist()[1]
            #logit_indicator = logit[1]/np.linalg.norm(logit) #/len(results)
            #assert len(switch)==1 #and switch[0] == int(f1>0.6)
            all_results_per_key[example_index_].append(( \
                            logit_indicator, f1, int(sent_index), results[0][3]))

        assert len(all_results_per_key)==1
        return sorted(list(all_results_per_key.values())[0], key=lambda x: x[0], reverse=True)

    def get_span_prediction(self, model, dataloader, examples, eval_features, has_keyword):

        RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "keyword_logits", "switch"])
        all_results = []
        em_all_results = collections.defaultdict(list)
        accs = []
        for batch in dataloader:
            example_indices = batch[-1]
            batch_to_feed = [t.to(self.device) for t in batch[:-1]]
            with torch.no_grad():
                if has_keyword:
                    batch_start_logits, batch_end_logits, batch_keyword_logits, batch_switch = model(batch_to_feed)
                else:
                    batch_start_logits, batch_end_logits, batch_switch = model(batch_to_feed)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                switch = batch_switch[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                gold_start_positions = eval_feature.start_position
                gold_end_positions = eval_feature.end_position
                gold_switch = eval_feature.switch
                if has_keyword:
                    keyword_logits = batch_keyword_logits[i].detach().cpu().tolist()
                    gold_keyword_positions = eval_feature.keyword_position
                else:
                    keyword_logits = None
                if gold_switch == [1]:
                    acc = np.argmax(switch) == 1
                elif has_keyword:
                    start_logits = start_logits[:len(eval_feature.tokens)]
                    end_logits = end_logits[:len(eval_feature.tokens)]
                    scores = []
                    for (i, s) in enumerate(start_logits):
                        for (j, e) in enumerate(end_logits[i:]):
                            for (k, key) in enumerate(keyword_logits[i:i+j+1]):
                                scores.append(((i, i+j, i+k), s+e+key))
                    scores = sorted(scores, key=lambda x: x[1], reverse=True)
                    acc = scores[0][0] in [(s, e, key) for (s, e, key) in \
                            zip(gold_start_positions, gold_end_positions, gold_keyword_positions)]
                else:
                    start_logits = start_logits[:len(eval_feature.tokens)]
                    end_logits = end_logits[:len(eval_feature.tokens)]
                    scores = []
                    for (i, s) in enumerate(start_logits):
                        for (j, e) in enumerate(end_logits[i:]):
                            scores.append(((i, i+j), s+e))
                    scores = sorted(scores, key=lambda x: x[1], reverse=True)
                    acc = scores[0][0] in  zip(gold_start_positions, gold_end_positions)

                em_all_results[eval_feature.example_index].append((unique_id, acc))
                all_results.append(RawResult(unique_id=unique_id,
                                            start_logits=start_logits,
                                            end_logits=end_logits,
                                            keyword_logits=keyword_logits,
                                            switch=switch))

        return get_span_prediction(examples, eval_features, all_results, has_keyword)


def get_qa_prediction(examples, features, results, n_best_size=10):

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
       "PrelimPrediction",
       ["feature_index", "start_index", "end_index", "logit", "no_answer_logit"])

    example_index_to_features = collections.defaultdict(list)
    for feature in features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in results:
        unique_id_to_result[result.unique_id] = result

    predictions_list = []
    for example_index, example in enumerate(examples):
        prelim_predictions = []
        for (feature_index, feature) in enumerate(example_index_to_features[example_index]):
            result = unique_id_to_result[feature.unique_id]
            switch = np.argmax(result.switch[:3])
            if switch > 0:
                prelim_predictions.append(_PrelimPrediction(
                            feature_index=feature_index, start_index=-switch, end_index=-switch,
                            logit=result.switch[switch]-result.switch[3], no_answer_logit=result.switch[3]))
                continue

            scores = []
            start_logits = result.start_logits[:len(feature.tokens)]
            end_logits = result.end_logits[:len(feature.tokens)]
            for (i, s) in enumerate(start_logits):
                for (j, e) in enumerate(end_logits[i:i+10]):
                    scores.append(((i, i+j), s+e-result.switch[3]))

            scores = sorted(scores, key=lambda x: x[1], reverse=True)

            for (start_index, end_index), score in scores:
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
                if length > 10:
                    continue
                prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    start_index=start_index,
                    end_index=end_index,
                    logit=score, no_answer_logit=result.switch[3]))

        predictions = []
        for pred in sorted(prelim_predictions, key=lambda x: x.logit, reverse=True):
            feature = example_index_to_features[example_index][pred.feature_index]

            if pred.start_index == pred.end_index == -1:
                final_text = "yes"
                sp_fact = " ".join(feature.doc_tokens)
            elif pred.start_index == pred.end_index == -2:
                final_text = "no"
                sp_fact = " ".join(feature.doc_tokens)
            else:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                try:
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                except Exception:
                    print ("Error during postprocessing")
                    from IPython import embed; embed()
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = feature.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text)
                sp_fact = " ".join(feature.doc_tokens[:orig_doc_start] + ["@@"] + \
                                orig_tokens + ["@@"] + feature.doc_tokens[orig_doc_end+1:])

            predictions.append({'text': final_text, 'logit': pred.logit})
            if len(predictions)==n_best_size:
                break
        predictions_list.append(predictions)

    return predictions_list

def get_span_prediction(examples, features, result, with_keyword):

    prelim_predictions = []
    yn_predictions = []

    assert len(examples)==1
    example = examples[0]

    feature = sorted(features, key=lambda f: f.unique_id)[0]
    gold_start_positions = feature.start_position
    gold_end_positions = feature.end_position
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
       "PrelimPrediction",
       ["start_index", "end_index", "keyword_index", "logit"])

    if len(result)!=1:
        from IPython import embed; embed()
    result = result[0]

    switch = np.argmax(result.switch)
    if switch == 1:
        prelim_predictions.append(
            _PrelimPrediction(
                start_index=-1,
                end_index=-1,
                keyword_index=-1,
                logit=result.switch[1]))
    elif switch == 0:
        scores = []
        start_logits = result.start_logits[:len(feature.tokens)]
        end_logits = result.end_logits[:len(feature.tokens)]
        if with_keyword:
            keyword_logits = result.keyword_logits[:len(feature.tokens)]
            for (i, s) in enumerate(start_logits):
                for (j, e) in enumerate(end_logits[i:]):
                    for (k, key) in enumerate(keyword_logits[i:i+j+1]):
                        scores.append(((i, i+j, i+k), s+e+key))
        else:
            for (i, s) in enumerate(start_logits):
                for (j, e) in enumerate(end_logits[i:]):
                    scores.append(((i, i+j, i), s+e))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        for (start_index, end_index, keyword_index), score in scores:
            if start_index >= len(feature.tokens):
                continue
            if end_index >= len(feature.tokens):
                continue
            if not (start_index <= keyword_index <= end_index):
                continue
            if start_index not in feature.token_to_orig_map or end_index not in feature.token_to_orig_map:
                continue
            if start_index-1 in feature.token_to_orig_map and feature.token_to_orig_map[start_index-1]==feature.token_to_orig_map[start_index]:
                continue
            if end_index+1 in feature.token_to_orig_map and feature.token_to_orig_map[end_index+1]==feature.token_to_orig_map[end_index]:
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index
            if length <= 2:
                continue
            prelim_predictions.append(
            _PrelimPrediction(
                start_index=start_index,
                end_index=end_index,
                keyword_index=keyword_index,
                logit=score))
    else:
        raise NotImplementedError()

    prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: x.logit,
            reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
       "NbestPrediction", ["text", "text2", "logit"])

    seen_predictions = {}
    nbest = []

    def get_text(start_index, end_index, keyword_index):
        if start_index == end_index == -1:
            final_text = example.all_answers[-1]
        else:
            feature = features[0]

            tok_tokens = feature.tokens[start_index:(end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[start_index]
            orig_doc_end = feature.token_to_orig_map[end_index]
            orig_doc_keyword = feature.token_to_orig_map[keyword_index]

            orig_tokens = feature.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            orig_tokens2 = orig_tokens.copy()
            for i in range(orig_doc_keyword, orig_doc_keyword-5, -1):
                if i-orig_doc_start<0: break
                if orig_tokens[i-orig_doc_start] in ['the', 'a', 'an']:
                    orig_tokens2[i-orig_doc_start] = 'which'
                    assert orig_tokens[i-orig_doc_start] != 'which'
                    break

            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())

            final_text = get_final_text(tok_text, " ".join(orig_tokens))
            final_text2 = get_final_text(tok_text, " ".join(orig_tokens2))

        return final_text, final_text2


    for pred in prelim_predictions:
        prediction, prediction2 = get_text(pred.start_index, pred.end_index, pred.keyword_index)
        orig_question = ' '.join(example.doc_tokens)

        if with_keyword:
            question1 = prediction2 if with_keyword else prediction
            question2 = orig_question.replace(prediction, '[ANSWER]')
            assert '[ANSWER]' in question2
            for token in [', [ANSWER]', '[ANSWER] ,', '[ANSWER] who', \
                          '[ANSWER] when', '[ANSWER] where', '[ANSWER] which', \
                          '[ANSWER] that', '[ANSWER] whose']:
                if token in question2:
                    if token=='[ANSWER] whose':
                        question = question2.replace(token, " [ANSWER] 's ")
                    else:
                        question2 = question2.replace(token, ' [ANSWER] ')
        else:
            orig_question_tokens = orig_question.split(' ')
            prediction_tokens = prediction.split(' ')
            start, end = None, None
            for i in range(len(orig_question_tokens)-len(prediction_tokens)+1):
                if orig_question_tokens[i:i+len(prediction_tokens)]==prediction_tokens:
                    start, end = i, i+len(prediction_tokens)
                    break
            if start is None and end is None:
                for i in range(len(orig_question_tokens)-len(prediction_tokens)+1):
                    text = ' '.join(orig_question_tokens[i:i+len(prediction_tokens)])
                    if normalize_answer(text)==normalize_answer(prediction):
                        start, end = i, i+len(prediction_tokens)
                        break
            if start is None and end is None:
                for i in range(len(orig_question_tokens)-len(prediction_tokens)+1):
                    text = ' '.join(orig_question_tokens[i:i+len(prediction_tokens)])
                    if normalize_answer(text).startswith(normalize_answer(prediction)):
                        start, end = i, len(orig_question_tokens)
                        print ("==== to long question ====")
                        print (' '.join(orig_question_tokens))
                        print (' '.join(orig_question_tokens[start:end]))
                        break

            try:
                assert start is not None and end is not None
            except Exception:
                print (orig_question)
                print (prediction)
            try:
                question1, question2 = intersection_convert_to_queries(
                    orig_question_tokens, start, end-1)
            except Exception:
                embed()
                assert False
            question1, question2 = ' '.join(question1), ' '.join(question2)

        def postprocess(question):
            question = question.strip()
            while '  ' in question:
                question = question.replace('  ', ' ')
            if not question.endswith('?'):
                question += '?'
            while question.replace(' ', '').endswith('??'):
                question = question[:-1]
            return question
        return postprocess(question1), postprocess(question2)

def get_final_text(pred_text, orig_text, do_lower_case=True):
    """Project the tokenized prediction back to the original text."""
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
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
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
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

def intersection_convert_to_queries(questions, start, end):
    q1, q2 = [], []
    for i, q in enumerate(questions):
        if q==',' and i in [start-1, start, end, end+1]:
            continue
        if i==0:
            if start==0 and q.startswith('wh'):
                status1, status2 = -1, 1
            elif (not q.startswith('wh')) and questions[start].startswith('wh'):
                status1, status2 = 1, 0
            else:
                status1, status2 = 0, 1
        if i<start:
            q1.append(q)
            if status1==0:
                q2.append(q)
        elif i>=start and i<=end:
            if status2==1 and i==start:
                if q=='whose':
                    q1.append('has')
                    continue
                if i>0 and (q in ['and', 'that'] or q.startswith('wh')):
                    continue
            q1.append(q)
            if status2==0:
                q2.append(q)
        elif i>end:
            if i==end+1 and len(q1)>0 and q=='whose':
                q2.append('has')
            elif i!=end+1 or len(q1)==0 or status1==-1  or not (q in ['and', 'that'] or q.startswith('wh')):
                q2.append(q)
    if len(q1)>0 and q1[-1] != '?':
        q1.append('?')
    if len(q2)>0 and q2[-1] != '?':
        q2.append('?')

    return q1, q2


