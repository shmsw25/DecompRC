import json
import tokenization
import collections
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from prepro_util import *
from hotpot_evaluate_v1 import f1_score as hotpot_f1_score


title_s = "<title>"
title_e = "</title>"

def get_dataloader(logger, args, input_file, subqueries_file, is_training, \
                   batch_size, num_epochs, tokenizer):

    if args.model == 'qa':
        train_examples = read_squad_examples(
            logger=logger, input_file=input_file, subqueries_file=subqueries_file, \
            is_training=is_training, debug=args.debug, \
            merge_query=args.merge_query, only_comp=args.only_comp)
    elif args.model == 'classifier':
        train_examples = read_classification_examples(
            logger=logger, input_file=input_file, is_training=is_training, debug=args.debug)
    elif args.model == 'span-predictor':
        train_examples = read_span_predictor_examples(
            logger=logger, input_file=input_file, is_training=is_training, debug=args.debug)

    else:
        raise NotImplementedError()

    return get_dataloader_given_examples(logger, args, train_examples, is_training, batch_size,
                                         num_epochs, tokenizer)

def get_dataloader_given_examples(logger, args, examples, is_training, batch_size, \
                                  num_epochs, tokenizer):
    num_train_steps = int(len(examples) / batch_size * num_epochs)
    if args.model == 'span-predictor':
        curr_convert_examples_to_features = span_convert_examples_to_features
    else:
        curr_convert_examples_to_features = convert_examples_to_features

    train_features, n_answers_with_truncated_answers = curr_convert_examples_to_features(
        logger=logger,
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        max_n_answers=args.max_n_answers if is_training or args.model=="span-predictor" else 1,
        is_training=is_training and args.model!='classifier',
        is_classifier=args.model.endswith("classifier"),
        force_span=args.model == "span-predictor",
        add_noise=args.add_noise)

    if is_training:
        switch_dict = collections.Counter()
        for f in train_features:
            for (s, m) in zip(f.switch, f.answer_mask):
                if m==0: break
                switch_dict[s] += 1
        print (switch_dict)

    logger.info("***** Running {} *****".format('training' if is_training else 'evaluation'))
    logger.info("  Num orig examples = %d", len(examples))
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Batch size = %d", batch_size)
    if is_training:
        logger.info("  Num steps = %d", num_train_steps)
        logger.info("  %% of tuncated_answers = %.2f%%" % \
                    (100.0*n_answers_with_truncated_answers/len(train_features)))

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    if is_training:
        if args.model in ["qa", "span-predictor"]:
            all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
            all_switches = torch.tensor([f.switch for f in train_features], dtype=torch.long)
            all_answer_mask = torch.tensor([f.answer_mask for f in train_features], dtype=torch.long)
            assert all_start_positions.size() == all_end_positions.size() == \
                all_switches.size() == all_answer_mask.size()

            if args.model == "qa" or 'intersec' in  args.predict_file:
                dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                        all_start_positions, all_end_positions, all_switches, all_answer_mask)
            elif args.model == "span-predictor":
                all_keyword_positions = torch.tensor([f.keyword_position for f in train_features], dtype=torch.long)
                assert all_start_positions.size() == all_keyword_positions.size()
                dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                        all_start_positions, all_end_positions, all_keyword_positions, all_switches, all_answer_mask)
            else:
                raise NotImplementedError()
        elif args.model.endswith("classifier"):
            all_labels = torch.tensor([f.switch for f in train_features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_labels)
        sampler=RandomSampler(dataset)
    else:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                        all_example_index)
        sampler=SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader, examples, train_features, num_train_steps

def read_squad_examples(logger, input_file, subqueries_file, is_training, debug,
                        merge_query, only_comp):

    def _process_sent(sent):
        if type(sent) != str:
            return [_process_sent(s) for s in sent]
        return sent.replace('–', '-').replace('&', 'and').replace('&amp;', 'and')

    input_data = []
    for _input_file in input_file.split(','):
        with open(_input_file, "r") as reader:
            this_data = json.load(reader)["data"]
            if debug:
                this_data = this_data[:50]
            input_data += this_data
        print ("Load {}, have {} data".format(_input_file, len(input_data)))

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    if len(subqueries_file) > 0:
        with open(subqueries_file, "r") as reader:
            subqueries_data = json.load(reader)

        use_distant=False
        if use_distant:
            name = 'out/decomposed/comparison_my_{}_distant_supervision.json'.format(
                        'train' if is_training else 'dev')
            print ("Loading distant supervision from {}".format(name))
            with open(name, 'r') as reader:
                distant_supervision = json.load(reader)

        _input_data = []
        for entry in input_data:
            orig_context = entry['paragraphs'][0]['context']
            qa = entry['paragraphs'][0]['qas'][0]
            if qa['id'] not in subqueries_data:
                continue
            if not use_distant:
                answers = [(qa['final_answers'][0], qa['final_answers'][0])]
            elif qa['id'] in distant_supervision and len(distant_supervision[qa['id']])>0:
                answers =  distant_supervision[qa['id']]
            else:
                continue
            assert all([len(a)==2 for a in answers])
            for i in range(2):
                qa1 = qa.copy()
                if True: #subqueries_data[qa['id']]['context'] is None:
                    context = orig_context
                else:
                    context = subqueries_data[qa['id']]['context'][i]
                if type(context)==str:
                    context = [context]
                context = [c.replace('  ', ' ') for c in context]

                qa1['answers'] = [[{'text': a[i]} for a in answers] for _ in range(len(context))]
                qa1['question'] = subqueries_data[qa['id']]['query'][i]
                qa1['id']  = '{}-{}'.format(qa['id'], i)
                qa1['final_answers'] = [a[i] for a in answers]
                if use_distant:
                    for a in answers:
                        if a[i] not in ['yes', 'no'] and a[i] not in context:
                            print (qa1)
                            assert False
                _input_data.append({'paragraphs':[{'context': context, 'qas': [qa1]}]})

        input_data = _input_data

    if only_comp:
        with open('/home/sewon/data/hotpotqa/hotpot_{}_v1.json'.format( \
                                    'train' if is_training else 'dev_distractor'), 'r') as f:
            orig_data = json.load(f)
            id2type = {entry['_id'].split('-')[0]:entry['type'] for entry in orig_data}
            id2type.update({k+"-inv":v for k, v in id2type.items()})


    examples = []
    for entry in tqdm(input_data):

        if 'paragraphs' not in entry:
            qa = {'question': entry['question'],
                  'final_answers': entry['final_answers'],
                  'answers': [[] for _ in range(len(entry['context']))],
                  'id': entry['id']}
            entry['paragraphs'] = [{'context': entry['context'], 'qas': [qa]}]

        for paragraph in entry["paragraphs"]:

            if only_comp:
                assert len(entry['paragraphs'])==1 and len(entry['paragraphs'][0]['qas'])==1
                q_type = id2type[entry['paragraphs'][0]['qas'][0]['id']]
                assert q_type in ['comparison', 'bridge']
                if subqueries_data is not None:
                    assert q_type == "comparison"
                elif q_type != "comparison":
                    continue

            context = paragraph['context']
            qas = paragraph['qas']

            if type(context)==str:
                context = [context]
                for i, qa in enumerate(qas):
                    if 'is_impossible' in qa:
                        assert (len(qa['answers'])==0 and qa['is_impossible']) or \
                                (len(qa['answers'])>0 and not qa['is_impossible'])
                    qas[i]["answers"] = [qa["answers"]]
            try:
                assert np.all([len(qa['answers'])==len(context) for qa in qas])
            except Exception:
                from IPython import embed; embed()
                assert False

            context = [c.lower() for c  in context]

            doc_tokens_list, char_to_word_offset_list = [], []
            for paragraph_text in context:
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)
                doc_tokens_list.append(doc_tokens)
                char_to_word_offset_list.append(char_to_word_offset)

            for qa in qas:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                switch = 0

                assert len(qa['answers']) == len(context)
                if 'final_answers' in qa:
                    all_answers = qa['final_answers']
                else:
                    all_answers = []
                    for answers in qa['answers']:
                        all_answers += [a['text'] for a in answers]

                if (not is_training) and len(all_answers)==0:
                    all_answers = ["None"]

                assert len(all_answers)>0

                original_answers_list, start_positions_list, end_positions_list, switches_list = [], [], [], []
                for (paragraph_text, doc_tokens, char_to_word_offset, answers) in zip( \
                        context, doc_tokens_list, char_to_word_offset_list, qa['answers']):

                    if len(answers)==0:
                        original_answers = [""]
                        start_positions, end_positions = [0], [0]
                        switches = [3]
                    else:
                        original_answers, switches, start_positions, end_positions = detect_span(\
                                answers, paragraph_text, doc_tokens, char_to_word_offset)
                    original_answers_list.append(original_answers)
                    start_positions_list.append(start_positions)
                    end_positions_list.append(end_positions)
                    switches_list.append(switches)

                examples.append(SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens_list,
                        orig_answer_text=original_answers_list,
                        all_answers=all_answers,
                        start_position=start_positions_list,
                        end_position=end_positions_list,
                        switch=switches_list))

    return examples


def read_span_predictor_examples(logger, input_file, is_training, debug):
    def _process_sent(sent):
        if type(sent) != str:
            return [_process_sent(s) for s in sent]
        return sent.replace('–', '-').replace('&', 'and').replace('&amp;', 'and')

    input_data = []
    for _input_file in input_file.split(','):
        if _input_file in ['bridge', 'intersec'] and not is_training: continue
        with open(_input_file, "r") as reader:
            data = json.load(reader)["data"]
            if debug:
                data = data[:200]
            if 'paragraphs' in data[0]:
                data = [d['paragraphs'][0]['qas'][0] for d in data]
                for d in data:
                    d['question'] = d['question'].lower()
            input_data += data

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in tqdm(input_data):
        question_text = entry['question']
        id_ = entry['id']
        #indices = entry['indices']

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        start_position = None
        end_position = None
        keyword_position = None

        for c in question_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            elif c == 'S':
                if start_position is None:
                    start_position = len(doc_tokens)
                elif end_position is None:
                    end_position = len(doc_tokens)-1
                else:
                    raise NotImplemented
            elif c == 'K':
                if keyword_position is None:
                    keyword_position = len(doc_tokens)
                else:
                    assert  keyword_position+1 == len(doc_tokens)
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        if start_position is None and end_position is None and keyword_position is None:
            all_answers = [question_text]
            original_answers = [question_text]
            switch = 1
            start_position, end_position, keyword_position = 0, 0, 0
        elif start_position is not None and end_position is not None:
            all_answers = [' '.join(doc_tokens[start_position:end_position+1])]
            original_answers = all_answers
            switch = 0
            if keyword_position is None:
                keyword_position = start_position
        else:
            print (question_text)
            raise NotImplementedError()

        examples.append(SquadExample(
                qas_id=id_,
                question_text="",
                doc_tokens=doc_tokens,
                orig_answer_text=original_answers,
                all_answers=all_answers + [" ".join(doc_tokens)],
                start_position=start_position,
                end_position=end_position,
                keyword_position=keyword_position,
                switch=switch))

    return examples


def read_classification_examples(logger, input_file, is_training, debug):
    def _process_sent(sent):
        if type(sent) != str:
            return [_process_sent(s) for s in sent]
        return sent.replace('–', '-').replace('&', 'and').replace('&amp;', 'and')

    def _to_content(article):
        content = " ".join([sent.strip() for sent in article[1]])
        return "<title> {} </title> {}".format(_process_sent(article[0]), content).lower()


    input_file, decomposed_files = input_file.split(',', 1)
    with open(input_file, 'r') as f:
        orig_data = json.load(f)['data']

    id2question = {}
    for d in orig_data:
        assert len(d['paragraphs'])==len(d['paragraphs'][0]['qas'])==1
        qa  = d['paragraphs'][0]['qas'][0]
        id2question[qa['id']] = (qa['question'], qa['final_answers'])

    nbest_output = collections.defaultdict(dict)

    if is_training:
        comp_keys = set()
        for d in orig_data:
            assert len(d['paragraphs'])==len(d['paragraphs'][0]['qas'])==1
            qa  = d['paragraphs'][0]['qas'][0]
            if qa['type']=='comparison':
                comp_keys.add(qa['id'])
        print ("Train set has {} comparison questions".format(len(comp_keys)))
    else:
        with open('data/decomposed-predictions/type_dev_predictions.json', 'r') as f:
            type_predictions = json.load(f)
            comp_keys = set([k for k, v in type_predictions.items() if v[0][0]>=0.5])
            print ("{} out of {} are comparison".format(len(comp_keys), len(type_predictions)))

    assert 'comparison' not in input_file or input_file.startswith('comparison')
    for name in decomposed_files.split(','):
        print ("Loading {} {}".format('train' if is_training else 'dev', name))
        with open('data/decomposed-predictions/{}_decomposed_{}_nbest_predictions.json'.format(
                name, 'train' if is_training else 'dev'), 'r') as f:
            curr_output = json.load(f)
        if name=='comparison':
            comp_keys = comp_keys & set(curr_output.keys())
        for (k, v) in curr_output.items():
            if k in comp_keys:
                if is_training or name!='comparison':
                    continue
                else:
                    v = [v[0].copy()]
            if name=='comparison' and k not in comp_keys:
                continue
            nbest_output[k][name] = v

    print ("{} orig data, {} predictions".format(len(orig_data),len(nbest_output)))

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    input_data = []
    n_each = 8 if is_training else 1

    ids = list(set(nbest_output.keys())&set(id2question.keys()))
    ids = np.random.permutation(list(ids))
    for id_ in ids:
        if debug and len(input_data)>5000:
            break

        nbest_out = nbest_output[id_]
        question, groundtruth = id2question[id_]

        answer_set = []
        evidence_set = []
        label_set = []
        f1_set = []

        if 'comparison' in nbest_out:
            assert len(nbest_out)==1
            names = ['comparison']
        else:
            names = ['bridge', 'intersec', 'onehop']
            assert set(nbest_out.keys())==set(names)

        for name in names:
            for pred in nbest_out[name][:n_each]:
                max_f1 = max(hotpot_f1_score(pred['text'], gt)[0] for gt in groundtruth)
                if is_training and 0.4<=max_f1<=0.6:
                    continue
                answer_set.append(("({}) {}".format(name, pred['text'])))
                evidence = pred['evidence'].lower()
                paragraphs = [line.strip() for line in evidence.split('<title>') if len(line.strip())>0]
                paragraphs = list(set(paragraphs))
                evidence = " ".join(['<title> {}'.format(para) for para in paragraphs])
                evidence_set.append(evidence)
                f1_set.append(max_f1)
                label_set.append(int(max_f1 > 0.6))

        if is_training and (len(f1_set)==0 or max(f1_set)<0.4):
            continue

        assert len(f1_set)>0

        j = 0
        ratio = sum(label_set)*1.0/len(label_set)
        for i, (answer, evidence, f1, label) in enumerate(zip( \
                            answer_set, evidence_set, f1_set, label_set)):
            def put(j):
                input_data.append({
                        'id': '{}-{}'.format(id_, j),
                        'question': question,
                        'context': evidence,
                        'answer': answer,
                        'all_answers': f1,
                        'label': label})
            put(j)
            j+=1

    examples = []
    for entry in input_data:
        qas_id = entry['id']
        question = entry['question']
        context = entry['context']
        answer = entry['answer']
        label = entry['label']
        doc_tokens = []
        prev_is_whitespace = True
        for c in context:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False

        examples.append(SquadExample(
                    qas_id=qas_id,
                    question_text=question + " " + answer,
                    doc_tokens=[doc_tokens],
                    switch=[label],
                    all_answers=entry['all_answers'],
                    orig_answer_text=[""], start_position=[0], end_position=[0]))
    return examples

def convert_examples_to_features(logger, examples, tokenizer, max_seq_length,
                    doc_stride, max_query_length, max_n_answers,
                    is_training, is_classifier=False, force_span=False, add_noise=0):
    """Loads a data file into a list of `InputBatch`s."""


    def _convert_examples_to_features(example_index, example):
        unique_id = 1000*example_index

        truncated = []
        features = []
        features_with_truncated_answers = []
        counter_n_answers = collections.Counter()

        query_tokens = tokenizer.tokenize(example.question_text)
        if is_training and add_noise>0 and np.random.random()<add_noise*0.5:
            length = len(query_tokens)
            keywords = []
            for i, token in enumerate(query_tokens):
                if token in ['what']:
                    keywords.append((0, i))
                elif token in ['which']:
                    keywords.append((1, i))
                elif token in ['who', 'when', 'where', 'whom', 'why']:
                    keywords.append((2, i))
            drop_random = False
            if len(keywords)>0:
                key, i = sorted(keywords)[0]
                if key==0:
                    assert query_tokens[i] == 'what'
                    if len(query_tokens)>i+1 and query_tokens[i+1] in ['is', 'was']:
                        query_tokens = query_tokens[:i] + query_tokens[i+2:]
                    else:
                        query_tokens = query_tokens[:i] + query_tokens[i+1:]
                elif key==1:
                    assert query_tokens[i] == 'which'
                    query_tokens = query_tokens[:i] + ["the"] + query_tokens[i+1:]
                elif key==2 and query_tokens[i] == "where" and len(query_tokens)>i+1:
                    if query_tokens[i+1] in ['did', 'do', 'does']:
                        query_tokens = query_tokens[:i] + ["the", "place"] + query_tokens[i+2:]
                    elif query_tokens[i+1] in ['was', 'is']:
                        query_tokens = query_tokens[:i] + query_tokens[i+2:]
                    else:
                        drop_random=True
                else:
                    drop_random=True
            else:
                drop_random=True
            if drop_random:
                i = np.random.choice(range(len(query_tokens)))
                query_tokens = query_tokens[:i] + query_tokens[i+1:]
        elif is_training and add_noise>0 and np.random.random()<add_noise*0.5:
            i = np.random.choice(range(len(query_tokens)))
            query_tokens = query_tokens[:i] + query_tokens[i+1:]

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]


        assert len(example.doc_tokens) == len(example.orig_answer_text) == \
            len(example.start_position) == len(example.end_position) == len(example.switch)

        for (doc_tokens, original_answer_text_list, start_position_list, end_position_list, switch_list) in \
                zip(example.doc_tokens, example.orig_answer_text, example.start_position, \
                    example.end_position, example.switch):

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_positions = []
            tok_end_positions = []
            if is_training or force_span:
                for (orig_answer_text, start_position, end_position) in zip( \
                            original_answer_text_list, start_position_list, end_position_list):
                    if orig_answer_text in ['yes', 'no']:
                        tok_start_positions.append(-1)
                        tok_end_positions.append(-1)
                        continue
                    tok_start_position = orig_to_tok_index[start_position]
                    if end_position < len(doc_tokens) - 1:
                        tok_end_position = orig_to_tok_index[end_position + 1] - 1
                    else:
                        tok_end_position = len(all_doc_tokens) - 1
                    (tok_start_position, tok_end_position) = _improve_answer_span(
                        all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                        orig_answer_text)
                    tok_start_positions.append(tok_start_position)
                    tok_end_positions.append(tok_end_position)
                to_be_same = [len(original_answer_text_list), \
                                    len(start_position_list), len(end_position_list),
                                    len(switch_list), \
                                    len(tok_start_positions), len(tok_end_positions)]
                assert all([x==to_be_same[0] for x in to_be_same])


            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
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
                start_offset += min(length, doc_stride)

            truncated.append(len(doc_spans))


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

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                        split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                start_positions = []
                end_positions = []
                switches = []
                answer_mask = []
                if is_training or force_span:
                    for (orig_answer_text, start_position, end_position, switch, \
                                tok_start_position, tok_end_position) in zip(\
                                original_answer_text_list, start_position_list, end_position_list, \
                                switch_list, tok_start_positions, tok_end_positions):
                        if orig_answer_text not in ['yes', 'no'] or switch == 3:
                            # For training, if our document chunk does not contain an annotation
                            # we throw it out, since there is nothing to predict.
                            doc_start = doc_span.start
                            doc_end = doc_span.start + doc_span.length - 1
                            if (tok_start_position < doc_start or
                                    tok_end_position < doc_start or
                                    tok_start_position > doc_end or tok_end_position > doc_end):
                                continue
                            doc_offset = len(query_tokens) + 2
                            start_position = tok_start_position - doc_start + doc_offset
                            end_position = tok_end_position - doc_start + doc_offset
                        else:
                            start_position, end_position = 0, 0
                        start_positions.append(start_position)
                        end_positions.append(end_position)
                        switches.append(switch)
                    to_be_same = [len(start_positions), len(end_positions), len(switches)]
                    assert all([x==to_be_same[0] for x in to_be_same])
                    if sum(to_be_same) == 0:
                        #if is_training and np.random.random()<0.5:
                        #    continue
                        start_positions = [0]
                        end_positions = [0]
                        switches = [3]

                    counter_n_answers[len(start_positions)] += 1

                    if len(start_positions) > max_n_answers:
                        features_with_truncated_answers.append(len(features))
                        start_positions = start_positions[:max_n_answers]
                        end_positions = end_positions[:max_n_answers]
                        switches = switches[:max_n_answers]
                    answer_mask = [1 for _ in range(len(start_positions))]
                    for _ in range(max_n_answers-len(start_positions)):
                        start_positions.append(0)
                        end_positions.append(0)
                        switches.append(0)
                        answer_mask.append(0)

                elif is_classifier:
                    switches = example.switch

                features.append(
                    InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        doc_tokens=doc_tokens,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_position=start_positions,
                        end_position=end_positions,
                        switch=switches,
                        answer_mask=answer_mask))


                unique_id += 1

        return features, counter_n_answers, truncated
    outputs = Parallel(n_jobs=10, verbose=2)(delayed(_convert_examples_to_features)(example_index, example) \
                                              for example_index, example in enumerate(examples))

    features, counter_n_answers, truncated = [], collections.Counter(), []
    for (f, c, t) in outputs:
        features += f
        counter_n_answers.update(c)
        truncated += t


    if force_span:
        assert len([f for f in features if 3 in f.switch])==0

    return features, 0

def span_convert_examples_to_features(logger, examples, tokenizer, max_seq_length,
                    doc_stride, max_query_length, max_n_answers,
                    is_training, is_classifier=False, force_span=False, add_noise=0):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    truncated = []
    features = []
    features_with_truncated_answers = []


    for (example_index, example) in tqdm(enumerate(examples)):

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        if len(all_doc_tokens) > max_seq_length-1:
            all_doc_tokens = all_doc_tokens[:max_seq_length-1]

        start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            end_position = len(all_doc_tokens) - 1
        if example.keyword_position < len(example.doc_tokens) - 1:
            keyword_position = orig_to_tok_index[example.keyword_position + 1] - 1
        else:
            keyword_position = len(all_doc_tokens) - 1
        if is_training and end_position >= len(all_doc_tokens):
            continue

        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []

        for i in range(len(all_doc_tokens)):
            #if i+1==len(all_doc_tokens) or  tok_to_orig_index[i]<tok_to_orig_index[i+1]:
            token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
            #token_is_max_context[len(tokens)] = False
            tokens.append(all_doc_tokens[i])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=0,
                        doc_tokens=example.doc_tokens,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_position=[start_position],
                        end_position=[end_position],
                        keyword_position=[keyword_position],
                        switch=[example.switch],
                        answer_mask=[1]))
        unique_id+=1

    return features, 0




def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
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













