import os
import sys
import json
import argparse
import numpy as np
from collections import Counter, defaultdict

from hotpot_evaluate_v1 import normalize_answer, f1_score as hotpot_f1_score

def main():
    parser = argparse.ArgumentParser("Preprocess HOTPOT data")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--task", type=str, default="decompose")
    parser.add_argument("--out_name", default="out/onehop")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    out_name = args.out_name
    data_type = args.data_type
    new_data_path = os.path.join("data", "decomposed", "{}.json")
    new_data1_path = os.path.join("data", "decomposed", "{}.1.json")
    new_data2_path = os.path.join("data", "decomposed", "{}.2.json")

    if not os.path.isdir(os.path.join('data', 'decomposed-predictions')):
        os.makedirs(os.path.join('data', 'decomposed-predictions'))


    if args.task == "decompose":
        data_type, reasoning_type = data_type.split('_')
        assert data_type in ['dev', 'train'] and reasoning_type in ['b', 'i']

        with open(os.path.join('data', 'hotpot-all', '{}.json'.format(data_type)), 'r') as f:
            orig_data = json.load(f)['data']

        with open(os.path.join(out_name, '{}_predictions.json'.format(data_type)), 'r') as f:
            result = json.load(f)


        if not os.path.isdir(os.path.join('data', 'decomposed')):
            os.makedirs(os.path.join('data', 'decomposed'))

        prepro(orig_data, result,
                        new_data_path.format(args.data_type),
                        new_data1_path.format(args.data_type),
                        new_data2_path.format(args.data_type))

    elif args.task == 'plug':
        with open('{}/{}_1_nbest_predictions.json'.format(out_name, args.data_type), 'r') as f:
            out1 = json.load(f)
        with open(new_data1_path.format(data_type), 'r') as f:
            data1 = json.load(f)['data']
        with open(new_data2_path.format(data_type), 'r') as f:
            data2 = json.load(f)['data']

        print (new_data1_path.format(data_type), new_data2_path.format(data_type))

        new_data2 = []
        for i, (d1, d) in enumerate(zip(data1, data2)):
            assert len(d['paragraphs'])==1 and len(d['paragraphs'][0]['qas'])==1
            q = d['paragraphs'][0]['qas'][0]
            assert d1['paragraphs'][0]['qas'][0]['id'] == q['id'] and q['id'] in out1
            assert '[answer]' in q['question'], q['question']
            qas = []
            for j, prediction in enumerate(out1[q['id']][:args.topk]):
                qas.append({'question': q['question'].replace('[answer]', prediction['text']),
                      'id': "{}-{}".format(q['id'], j),
                      'answers': q['answers']})
                if 'index' in q:
                    qas[-1]['index'] = q['index']
                if 'final_answers' in q:
                    qas[-1]['final_answers'] = q['final_answers']
            new_data2.append({
                'paragraphs': [{'context': d['paragraphs'][0]['context'], 'qas': qas}]
            })

        with open(new_data2_path.format(data_type), 'w') as f:
            json.dump({'data': new_data2}, f)

    elif args.task.startswith('aggregate'):
        with open(new_data_path.format(data_type), 'r') as f:
            data = json.load(f)['data']
        data0 = [d['paragraphs'][0]['qas'][0] for d in data]
        contexts = [d['paragraphs'][0]['context'] for d in data]
        with open(new_data1_path.format(data_type), 'r') as f:
            data1 = [d['paragraphs'][0]['qas'][0] for d in json.load(f)['data']]
        with open(new_data2_path.format(data_type), 'r') as f:
            data2 = [d['paragraphs'][0]['qas'][0] for d in json.load(f)['data']]

        with open(os.path.join(out_name, '{}_1_nbest_predictions.json'.format(data_type)), 'r') as f:
            out1 = json.load(f)
        with open(os.path.join(out_name, '{}_2_nbest_predictions.json'.format(data_type)), 'r') as f:
            out2 = json.load(f)

        new_nbest_predictions = {}

        if args.task == 'aggregate-bridge':
            for (d0, d1, d2, context) in zip(data0, data1, data2, contexts):
                assert d0['id'] == d1['id'] == d2['id'].split('-')[0]
                answers = []
                o1 = out1[d1['id']]
                answer1_set, answer2_set = [], []
                for i in range(30): #args.topk):
                    if len(o1)==i:
                        break
                    answer1 = o1[i]['text']
                    if is_filtered(answer1_set, answer1):
                        continue
                    answer1_set.append(answer1)
                    try:
                        o2 = out2['{}-{}'.format(d1['id'], i)]
                    except Exception:
                        continue
                    for j in range(30): #args.topk):
                        if len(o2)==j:
                            break
                        answer2 = o2[j]['text']
                        if is_filtered(answer2_set, answer2):
                            continue
                        answer2_set.append(answer2)
                        answers.append((answer1, answer2, (i, j), (o1[i]['logit'], o2[j]['logit'])))

                answers = sorted(answers, key=lambda x: -sum(x[3]))[:args.topk]
                new_nbest_predictions[d0['id']] = [{
                        'text': answer2,
                        'evidence': o1[i]['evidence'] + " " + out2['{}-{}'.format(d1['id'], i)][j]['evidence'],
                        'queries': d1['question'] + " " + d2['question'].replace('[answer]', answer1),
                        'logit': logit
                    } for (answer1, answer2, (i, j), logit) in answers]

            with open('data/decomposed-predictions/bridge_decomposed_{}_nbest_predictions.json'.format(data_type.split('_')[0]), 'w') as f:
                json.dump(new_nbest_predictions, f)

        elif args.task == 'aggregate-intersec':
            new_nbest_predictions = {}
            for (d0, d1, d2, context) in zip(data0, data1, data2, contexts):
                assert d0['id'] == d1['id'] == d2['id']
                o1 = {o['text']:o['logit'] for o in filter_duplicate(out1[d1['id']])}
                o2 = {o['text']:o['logit'] for o in filter_duplicate(out2[d2['id']])}
                o1_items, o2_items = list(o1.items()).copy(), list(o2.items()).copy()
                for (text, logit) in o1_items:
                    tokens = [t for token in text.split('and') for t in token.split(',')]
                    tokens = [t.strip() for t in tokens if len(t.strip())>0]
                    if len(tokens)>0:
                        for token in tokens:
                            o1[token]=logit
                for (text, logit) in o2_items:
                    tokens = [t for token in text.split('and') for t in token.split(',')]
                    tokens = [t.strip() for t in tokens if len(t.strip())>0]
                    if len(tokens)>0:
                        for token in tokens:
                            o2[token]=logit

                combined_answers = [(t, (o1.get(t, 0), o2.get(t, 0))) for t in list(set(o1.keys())|set(o2.keys()))]
                answers_logits = sorted(combined_answers, key=lambda x: -sum(x[1]))
                answers = [a[0] for a in answers_logits]
                new_pred = []
                for answer, logit in answers_logits:
                    if answer in o1:
                        evidence1 = [o['evidence'] for o in out1[d1['id']] if o['text']==answer]
                        if len(evidence1)==0:
                            evidence1 = [o['evidence'] for o in out1[d1['id']] if answer in o['text']]
                        assert len(evidence1)>0, (answer, [o['text'] for o in out1[d1['id']]])
                    else:
                        evidence1 = ['']
                    if answer in o2:
                        evidence2 = [o['evidence'] for o in out2[d2['id']] if o['text']==answer]
                        if len(evidence2)==0:
                            evidence2 = [o['evidence'] for o in out2[d2['id']] if answer in o['text']]
                        assert len(evidence2)>0, (answer, [o['text'] for o in out2[d2['id']]])
                    else:
                        evidence2 = ['']
                    evidence = '{} {}'.format(evidence1[0], evidence2[0])
                    assert len(evidence.strip())>0
                    new_pred.append({'text': answer, 'evidence': evidence,
                                    'logit': logit, 'queries': d1['question']+" "+d2['question']})
                new_nbest_predictions[d0['id']] = new_pred


            with open('data/decomposed-predictions/intersec_decomposed_{}_nbest_predictions.json'.format(data_type.split('_')[0]), 'w') as f:
                json.dump(new_nbest_predictions, f)
        else:
            raise NotImplementedError()

    elif args.task == 'onehop':
        with open("data/hotpot-all/{}.json".format(data_type), 'r') as f:
            data = json.load(f)['data']
        data0 = [d['paragraphs'][0]['qas'][0] for d in data]
        contexts = [d['paragraphs'][0]['context'] for d in data]
        with open(os.path.join(out_name, '{}_nbest_predictions.json'.format(data_type)), 'r') as f:
            out0 = json.load(f)
        orig_nbest_predictions = {}
        for (d0, context) in zip(data0, contexts):
            orig_pred = [o for o in filter_duplicate(out0[d0['id']])]
            orig_nbest_predictions[d0['id']] = orig_pred
        with open('data/decomposed-predictions/onehop_decomposed_{}_nbest_predictions.json'.format(data_type), 'w') as f:
            json.dump(orig_nbest_predictions, f)

    else:
        raise  NotImplementedError("{} Not Supported".format(args.task))


def prepro(orig_data, result, new_data_path, new_data1_path, new_data2_path):
    new_data0 = []
    new_data1 = []
    new_data2 = []
    for datapoint in orig_data:
        paragraph = datapoint['paragraphs'][0]['context']
        qa = datapoint['paragraphs'][0]['qas'][0]
        if qa['id'] in result:
            (question1, question2, _, question) = result[qa['id']]
            assert len(qa['final_answers'])>0
            if len(new_data1)==0:
                print (question1)
                print (question2)
            d0 = {'context': paragraph, 'qas': [{
                'id': qa['id'], 'question': question.lower(),
                'final_answers': qa['final_answers'], 'answers': qa['answers']
            }]}
            d1 = {'context': paragraph, 'qas': [{
                'id': qa['id'], 'question': question1.lower(),
                'final_answers': qa['final_answers'], 'answers': qa['answers']
            }]}
            d2 = {'context': paragraph, 'qas': [{
                'id': qa['id'], 'question': question2.lower(),
                'final_answers': qa['final_answers'], 'answers': qa['answers']
            }]}
            new_data0.append({'paragraphs': [d0]})
            new_data1.append({'paragraphs': [d1]})
            new_data2.append({'paragraphs': [d2]})

    print (len(new_data0), len(new_data1), len(new_data2))

    with open(new_data_path, 'w') as f:
        json.dump({'data': new_data0}, f)
    with open(new_data1_path, 'w') as f:
        json.dump({'data': new_data1}, f)
    with open(new_data2_path, 'w') as f:
        json.dump({'data': new_data2}, f)

def _normalize_answer(text):
    if '<title>' in text:
        text = text.replace('<title>', '')
    if '</title>' in text:
        text = text.replace('</title>', '')

    list1 = ['/title>'[i:] for i in range(len('/title>'))]
    list2 = ['</title>'[:-i] for i in range(1, len('</title>'))] + \
                    ['<title>'[:-i] for i in range(1, len('<title>'))]

    for prefix in list1:
        if text.startswith(prefix):
            text = text[len(prefix):]

    for prefix in list2:
        if text.endswith(prefix):
            text = text[:-len(prefix)]

    if '(' in text and ')' not in text:
        texts = [t.strip() for t in text.split('(')]
        text = texts[np.argmax([len(t) for t in texts])]
    if ')' in text and '(' not in text:
        texts = [t.strip() for t in text.split(')')]
        text = texts[np.argmax([len(t) for t in texts])]

    text = normalize_answer(text)
    return text

def is_filtered(answer_set, new_answer):
    new_answer = _normalize_answer(new_answer)
    if len(new_answer)==0:
        return True
    for answer in answer_set:
        if _normalize_answer(answer) == new_answer:
            return True
    return False

def filter_duplicate(orig_answers):
    answers = []
    for answer in orig_answers:
        if is_filtered([a['text'] for a in answers], answer['text']): continue
        answers.append(answer)
    return answers

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

if __name__ == '__main__':
    main()






