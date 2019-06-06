import json
import collections
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from qa.Example import *

def get_dataloader(model, question, paragraphs, tokenizer, batch_size):

    if model == 'qa':
        examples = read_qa_example(question, paragraphs)
        max_seq_length = 300
    elif model == 'classifier':
        examples = read_classification_examples(question, paragraphs)
        max_seq_length = 400
    elif model == 'span-predictor':
        examples = read_span_predictor_examples(question)
        max_seq_length = 100
    else:
        raise NotImplementedError()

    if model == 'span-predictor':
        curr_convert_examples_to_features = convert_examples_to_span_features
    else:
        curr_convert_examples_to_features = convert_examples_to_features

    features = curr_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=128,
        max_query_length=64)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    sampler=SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader, examples, features

def read_qa_example(questions, context):

    def _process_sent(sent):
        if type(sent) != str:
            return [_process_sent(s) for s in sent]
        return sent.replace('–', '-').replace('&', 'and').replace('&amp;', 'and')

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples =[]
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
    for question in questions:
        examples.append(SquadExample(qas_id="custom",
                            question_text=question,
                            doc_tokens=doc_tokens_list,
                            orig_answer_text=[],
                            all_answers=[],
                            start_position=[],
                            end_position=[],
                            switch=[]))
    return examples

def read_span_predictor_examples(question):
    def _process_sent(sent):
        if type(sent) != str:
            return [_process_sent(s) for s in sent]
        return sent.replace('–', '-').replace('&', 'and').replace('&amp;', 'and')

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True

    start_position = None
    end_position = None
    keyword_position = None

    for c in question:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    return [SquadExample(
                qas_id="custom",
                question_text="",
                doc_tokens=doc_tokens,
                orig_answer_text=[],
                all_answers=[],
                start_position=0,
                end_position=0,
                keyword_position=0,
                switch=0)]

def read_classification_examples(questions, paragraphs):
    def _process_sent(sent):
        if type(sent) != str:
            return [_process_sent(s) for s in sent]
        return sent.replace('–', '-').replace('&', 'and').replace('&amp;', 'and')

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for question_id, question in enumerate(questions):
        doc_tokens = []
        prev_is_whitespace = True
        for c in " ".join(paragraphs):
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False

        examples.append(SquadExample(
                    qas_id="custom-{}".format(question_id),
                    question_text=question + " ",
                    doc_tokens=[doc_tokens],
                    switch=[0],
                    all_answers=[""],
                    orig_answer_text=[""], start_position=[0], end_position=[0]))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                    doc_stride, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    truncated = []
    features = []
    features_with_truncated_answers = []

    for example_index, example in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        for doc_tokens in example.doc_tokens:
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

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

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
                start_offset += min(length, doc_stride)


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
                input_mask = [1] * len(input_ids)

                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

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
                        start_position=[],
                        end_position=[],
                        switch=[],
                        answer_mask=[]))


                unique_id += 1
    return features

def convert_examples_to_span_features(examples, tokenizer, max_seq_length,
                    doc_stride, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    features = []

    assert len(examples)==1
    example = examples[0]

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

    return [InputFeatures(
                    unique_id=unique_id,
                    example_index=0,
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
                    answer_mask=[1])]




def _check_is_max_context(doc_spans, cur_span_index, position):
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










