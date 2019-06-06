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

import tokenization
from modeling import BertConfig, BertClassifier, BertForQuestionAnswering, \
    BertForQuestionAnsweringWithKeyword
from optimization import BERTAdam

from prepro import get_dataloader, get_dataloader_given_examples
from evaluate_qa import write_predictions
from evaluate_span import write_predictions as span_write_predictions

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "switch"])


def main():
    parser = argparse.ArgumentParser()
    BERT_DIR = "/home/sewon/for-inference/model/uncased_L-12_H-768_A-12/"
    ## Required parameters
    parser.add_argument("--bert_config_file", default=BERT_DIR+"bert_config.json", \
                        type=str, help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default=BERT_DIR+"vocab.txt", type=str, \
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default="out", type=str, \
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_file", type=str, \
                        help="SQuAD json for training. E.g., train-v1.1.json", \
                        default="")
    parser.add_argument("--predict_file", type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json", \
                        default="")
    parser.add_argument("--init_checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).", \
                        default=BERT_DIR+"pytorch_model.bin")
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
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=128, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--save_checkpoints_steps", default=1000, type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--iterations_per_loop", default=1000, type=int,
                        help="How many steps to make in each estimator call.")
    parser.add_argument("--n_best_size", default=3, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--accumulate_gradients", type=int, default=1, help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--eval_period', type=int, default=2000)

    parser.add_argument('--max_n_answers', type=int, default=5)
    parser.add_argument('--merge_query', type=int, default=-1)
    parser.add_argument('--reduce_layers', type=int, default=-1)
    parser.add_argument('--reduce_layers_to_tune', type=int, default=-1)

    parser.add_argument('--only_comp', action="store_true", default=False)

    parser.add_argument('--train_subqueries_file', type=str, default="") #500
    parser.add_argument('--predict_subqueries_file', type=str, default="") #500
    parser.add_argument('--prefix', type=str, default="") #500

    parser.add_argument('--model', type=str, default="qa") #500
    parser.add_argument('--pooling', type=str, default="max")
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--output_dropout_prob', type=float, default=0)
    parser.add_argument('--wait_step', type=int, default=30)
    parser.add_argument('--with_key', action="store_true", default=False)
    parser.add_argument('--add_noise', action="store_true", default=False)


    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
        if not args.predict_file:
            raise ValueError(
                "If `do_train` is True, then `predict_file` must be specified.")

    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.do_train and args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.info("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None

    eval_dataloader, eval_examples, eval_features, _ = get_dataloader(
                logger=logger, args=args,
                input_file=args.predict_file,
                subqueries_file=args.predict_subqueries_file,
                is_training=False,
                batch_size=args.predict_batch_size,
                num_epochs=1,
                tokenizer=tokenizer)
    if args.do_train:
        train_dataloader, train_examples, _, num_train_steps = get_dataloader(
                logger=logger, args=args, \
                input_file=args.train_file, \
                subqueries_file=args.train_subqueries_file, \
                is_training=True,
                batch_size=args.train_batch_size,
                num_epochs=args.num_train_epochs,
                tokenizer=tokenizer)

    #a = input()
    if args.model == 'qa':
        model = BertForQuestionAnswering(bert_config, 4)
        metric_name = "F1"
    elif args.model == 'classifier':
        if args.reduce_layers != -1:
            bert_config.num_hidden_layers = args.reduce_layers
        model = BertClassifier(bert_config, 2, args.pooling)
        metric_name = "F1"
    elif args.model == "span-predictor":
        if args.reduce_layers != -1:
            bert_config.num_hidden_layers = args.reduce_layers
        if args.with_key:
             Model= BertForQuestionAnsweringWithKeyword
        else:
            Model = BertForQuestionAnswering
        model = Model(bert_config, 2)
        metric_name = "Accuracy"
    else:
        raise NotImplementedError()

    if args.init_checkpoint is not None and args.do_predict and \
                len(args.init_checkpoint.split(','))>1:
        assert args.model == "qa"
        model = [model]
        for i, checkpoint in enumerate(args.init_checkpoint.split(',')):
            if i>0:
                model.append(BertForQuestionAnswering(bert_config, 4))
            print ("Loading from", checkpoint)
            state_dict = torch.load(checkpoint, map_location='cpu')
            filter = lambda x: x[7:] if x.startswith('module.') else x
            state_dict = {filter(k):v for (k,v) in state_dict.items()}
            model[-1].load_state_dict(state_dict)
            model[-1].to(device)

    else:
        if args.init_checkpoint is not None:
            print ("Loading from", args.init_checkpoint)
            state_dict = torch.load(args.init_checkpoint, map_location='cpu')
            if args.reduce_layers != -1:
                state_dict = {k:v for k, v in state_dict.items() \
                    if not '.'.join(k.split('.')[:3]) in \
                    ['encoder.layer.{}'.format(i) for i in range(args.reduce_layers, 12)]}
            if args.do_predict:
                filter = lambda x: x[7:] if x.startswith('module.') else x
                state_dict = {filter(k):v for (k,v) in state_dict.items()}
                model.load_state_dict(state_dict)
            else:
                model.bert.load_state_dict(state_dict)
                if args.reduce_layers_to_tune != -1:
                    model.bert.embeddings.required_grad = False
                    n_layers = 12 if args.reduce_layers==-1 else args.reduce_layers
                    for i in range(n_layers - args.reduce_layers_to_tune):
                        model.bert.encoder.layer[i].require_grad=False

        model.to(device)

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

    if args.do_train:
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
            ]

        optimizer = BERTAdam(optimizer_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_steps)

        global_step = 0

        best_f1 = 0
        wait_step = 0
        model.train()
        global_step = 0
        stop_training = False

        for epoch in range(int(args.num_train_epochs)):
            for step, batch in tqdm(enumerate(train_dataloader)):
                global_step += 1
                batch = [t.to(device) for t in batch]
                loss = model(batch, global_step)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if global_step % args.gradient_accumulation_steps == 0:
                    optimizer.step()    # We have accumulated enought gradients
                    model.zero_grad()
                if global_step % args.eval_period == 0:
                    model.eval()
                    f1 =  predict(args, model, eval_dataloader, eval_examples, eval_features, \
                                  device, write_prediction=False)
                    logger.info("%s: %.3f on epoch=%d" % (metric_name, f1*100.0, epoch))
                    if best_f1 < f1:
                        logger.info("Saving model with best %s: %.3f -> %.3f on epoch=%d" % \
                                (metric_name, best_f1*100.0, f1*100.0, epoch))
                        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                        torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                        model = model.cuda()
                        best_f1 = f1
                        wait_step = 0
                        stop_training = False
                    else:
                        wait_step += 1
                        if best_f1 > 0.1 and wait_step == args.wait_step:
                            stop_training = True
                    model.train()
            if stop_training:
                break

    elif args.do_predict:
        if type(model)==list:
            model = [m.eval() for m in model]
        else:
            model.eval()
        f1 = predict(args, model, eval_dataloader, eval_examples, eval_features, device)
        logger.info("Final %s score: %.3f%%" % (metric_name, f1*100.0))


def predict(args, model, eval_dataloader, eval_examples, eval_features, device, \
            write_prediction=True):
    all_results = []

    assert args.model == "qa" or type(model) != list

    if args.model == 'qa':

        RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "switch"])

        def _get_raw_results(model1):
            raw_results = []
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                example_indices = batch[-1]
                batch_to_feed = [t.to(device) for t in batch[:-1]]
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
        if type(model)==list:
            all_raw_results = [_get_raw_results(m) for m in model]
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
        else:
            all_results = _get_raw_results(model)


        output_prediction_file = os.path.join(args.output_dir, args.prefix+"predictions.json")
        output_nbest_file = os.path.join(args.output_dir, args.prefix+"nbest_predictions.json")

        f1 = write_predictions(logger, eval_examples, eval_features, all_results,
                        args.n_best_size if write_prediction else 1,
                        args.max_answer_length,
                        args.do_lower_case,
                        output_prediction_file if write_prediction else None,
                        output_nbest_file if write_prediction else None,
                        args.verbose_logging,
                        write_prediction=write_prediction)
        return f1

    elif args.model=='classifier':

        all_results = collections.defaultdict(list)
        all_results_per_key = collections.defaultdict(list)
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            example_indices = batch[-1]
            batch_to_feed = tuple(t.to(device) for t in batch[:-1])
            with torch.no_grad():
                batch_predicted_label = model(batch_to_feed)
            for i, example_index in enumerate(example_indices):
                logit = batch_predicted_label[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                eval_example = eval_examples[eval_feature.example_index]
                all_results[eval_example.qas_id].append( \
                                (logit, eval_feature.switch, eval_example.all_answers,
                                 eval_example.question_text))
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
            assert len(switch)==1 #and switch[0] == int(f1>0.6)
            all_results_per_key[example_index_].append(( \
                            logit_indicator, f1, int(sent_index), results[0][3]))
        accs = {}
        sents_scores = {}
        for key, results in all_results_per_key.items():
            ranked_labels = sorted(results, key=lambda x: (-x[0], x[2]))
            acc = ranked_labels[0][1]
            accs[key] = acc
            sents_scores[key] = [r for r in sorted(results, key=lambda x: x[2])]
        if write_prediction:
            output_prediction_file = os.path.join(args.output_dir, args.prefix+"class_scores.json")
            logger.info("Save score file into: "+output_prediction_file)
            with open(output_prediction_file, "w") as f:
                json.dump(sents_scores, f)

        return np.mean(list(accs.values()))

    elif args.model == "span-predictor":

        RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "keyword_logits", "switch"])

        has_keyword = args.with_key
        em_all_results = collections.defaultdict(list)
        accs = []
        for batch in eval_dataloader:
            example_indices = batch[-1]
            batch_to_feed = [t.to(device) for t in batch[:-1]]
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

        output_prediction_file = os.path.join(args.output_dir, args.prefix+"predictions.json")
        output_nbest_file = os.path.join(args.output_dir, args.prefix+"nbest_predictions.json")

        for example_index, results in em_all_results.items():
            acc = sorted(results, key=lambda x: x[0])[0][1]
            accs.append(acc)


        if write_prediction:
            is_bridge = 'bridge' in args.predict_file
            is_intersec = 'intersec' in args.predict_file
            assert (is_bridge and not is_intersec) or (is_intersec and not is_bridge)

            print ("Accuracy", np.mean(accs))
            f1 = span_write_predictions(logger, eval_examples, eval_features, all_results,
                        args.n_best_size if write_prediction else 1,
                        args.max_answer_length,
                        args.do_lower_case,
                        output_prediction_file if write_prediction else None,
                        output_nbest_file if write_prediction else None,
                        args.verbose_logging,
                        write_prediction=write_prediction,
                        with_key=args.with_key,
                        is_bridge=is_bridge)

        return np.mean(accs)

    raise NotImplementedError()


if __name__ == "__main__":
    main()
