import os
import json
import collections
import math
import six
import numpy as np
import tokenization
from collections import defaultdict

from hotpot_evaluate_v1 import normalize_answer, f1_score
from run_decomposition import intersection_convert_to_queries


rawResult = collections.namedtuple("RawResult",
                                  ["unique_id", "start_logits", "end_logits", "keyword_logits"])


def write_predictions(logger, all_examples, all_features, all_results, n_best_size,
                     max_answer_length, do_lower_case, output_prediction_file,
                     output_nbest_file, verbose_logging, write_prediction=True,
                     with_key=False, is_bridge=True):

    """Write final predictions to the json file."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
       "PrelimPrediction",
       ["start_index", "end_index", "keyword_index", "logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        yn_predictions = []

        feature = sorted(features, key=lambda f: f.unique_id)[0]
        gold_start_positions = feature.start_position
        gold_end_positions = feature.end_position

        result = unique_id_to_result[feature.unique_id]
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
            if with_key:
                keyword_logits = result.keyword_logits[:len(feature.tokens)]
                for (i, s) in enumerate(start_logits):
                    for (j, e) in enumerate(end_logits[i:]):
                        for (k, key) in enumerate(keyword_logits[i:i+j+1]):
                            if not (i==0 and j==len(feature.tokens)-1):
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
                length = end_index - start_index + 1
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

        if len(prelim_predictions)==0:
            embed()
            assert False

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

                final_text = get_final_text(tok_text, " ".join(orig_tokens), do_lower_case, \
                                            logger, verbose_logging)
                final_text2 = get_final_text(tok_text, " ".join(orig_tokens2), do_lower_case, \
                                            logger, verbose_logging)
                if '##' in final_text:
                    print (tok_text)
                    print (' '.join(orig_tokens))
                    print (final_text)
                    embed()
                    assert False


            return final_text, final_text2


        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            final_text, final_text2 = get_text(pred.start_index, pred.end_index, pred.keyword_index)
            if final_text in seen_predictions:
                continue

            nbest.append(
               _NbestPrediction(
                   text=final_text,
                   text2=final_text2,
                   logit=pred.logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
               _NbestPrediction(text="empty", text2="empty", logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.logit)

        probs = _compute_softmax(total_scores)
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output['text'] = entry.text
            output['text2'] = entry.text2
            output['probability'] = probs[i]
            output['logit'] = entry.logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = (nbest_json[0]["text"],
                                           nbest_json[0]["text2"],
                                           example.all_answers[:-1], # groundtruth
                                           example.all_answers[-1]) # orig_question
        all_nbest_json[example.qas_id] = nbest_json


    f1_scores = []

    for (prediction, _, groundtruth, orig_question) in all_predictions.values():
        f1_scores.append(max([f1_score(prediction, gt)[0] for gt in groundtruth]))

    if write_prediction:
        logger.info("Writing predictions to: %s" % (output_prediction_file))
        logger.info("Writing nbest to: %s" % (output_nbest_file))

        final_predictions = {}
        final_nbest_predictions = defaultdict(list)
        for key in  all_predictions:
            orig_question = all_predictions[key][-1]
            for d in all_nbest_json[key]:
                orig_question, question1, question2 = \
                    get_decomposed(orig_question, d['text'], d['text2'], is_bridge, with_key)
                final_nbest_predictions[key].append((question1, question2, orig_question, orig_question))
            final_predictions[key] = final_nbest_predictions[key][0]

        l = [v for k, v in sorted(final_predictions.items(), key=lambda x: x[0])]
        print (l[0])
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(final_predictions, indent=4) + "\n")

        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(final_nbest_predictions, indent=4) + "\n")

    return np.mean(f1_scores)

def get_decomposed(orig_question, prediction, prediction2, is_bridge, with_key):
    while '  ' in orig_question:
        orig_question = orig_question.replace('  ', ' ')
    if is_bridge:
        question1 = prediction2 if with_key else prediction
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

        assert start is not None and end is not None
        question1, question2 = intersection_convert_to_queries(
                orig_question_tokens, start, end-1)
        question1, question2 = ' '.join(question1), ' '.join(question2)

    return orig_question, question1, question2

def get_final_text(pred_text, orig_text, do_lower_case, logger, verbose_logging):
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
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
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
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
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
