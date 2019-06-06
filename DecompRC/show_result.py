import os
import json
import argparse
import numpy as np

from tqdm import tqdm
from collections import defaultdict, Counter
from prettytable import PrettyTable

from hotpot_evaluate_v1 import f1_score

def f1(pred, a):
    return f1_score(pred, a[1])[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="/home/sewon/data/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--prediction_file", default="final_predictions.json")
    args = parser.parse_args()
    np.random.seed(1995)

    nbest_output = defaultdict(dict)

    with open(args.data_file, 'r') as f:
        orig_data = json.load(f)

    id2question = {}
    for d in orig_data:
        id2question[d['_id']] = (d['question'],
                                 d.get('answer', 'unknown'), d.get('type', 'bridge'), d['context'])

    with open('data/decomposed-predictions/type_dev_predictions.json', 'r') as f:
        type_predictions = json.load(f)
        comp_keys = set([k for k, v in type_predictions.items() if v[0][0]>=0.5])
    with open('data/decomposed-predictions/comparison_decomposed_dev_nbest_predictions.json', 'r') as f:
        curr_output = json.load(f)

    comp_keys = comp_keys & set(curr_output.keys())

    for (k, v) in curr_output.items():
        if k in comp_keys:
            nbest_output[k]['comparison'] = v

    def print_result(all_f1s):
        pt = PrettyTable()
        pt.field_names =  ["name", "F1"]
        pt.add_row(["Ovearll", "%.3f" % (100.0*np.mean([v for k, v in all_f1s.items()]))])
        for key in  ['bridge','comparison']:
            pt.add_row([key, "%.3f" % (100.0*np.mean([v for k, v in all_f1s.items() if id2question[k][2]==key]))])
        print (pt)

    for name in ['bridge','intersec','onehop']:
        fn = '{}_decomposed_dev_nbest_predictions.json'.format(name)
        with open("data/decomposed-predictions/"+fn, 'r') as f:
            curr_output = json.load(f)
        if list(curr_output.keys())[0].startswith('long-'):
            curr_output = {k[5:]:v for k, v in curr_output.items()}
        for (k, v) in curr_output.items():
            if k not  in comp_keys:
                if len(v)==0:
                    v = [{'text': 'UNKNOWN', 'evidence': 'UNKNOWN', 'logit':-9999}]
                nbest_output[k][name] = v

    ### Upperbound
    print ("=== upperbound ===")
    f1s_upperbound = {}
    for k, dic in nbest_output.items():
        candidates = [ps[0]['text'] for ps in dic.values()]
        max_f1 = max([f1(candi, id2question[k]) for candi in candidates])
        f1s_upperbound[k] = max_f1
    print_result(f1s_upperbound)

    all_f1s = {}
    with open('out/scorer/dev_class_scores.json', 'r') as f:
        verifier_output = json.load(f)

    verifier_f1s = {}
    reasoning_counter = Counter()
    reasoning_list = ['(bridge)', '(intersec)', '(onehop)', '(comparison)']
    final_prediction = defaultdict(dict)
    for k, v in verifier_output.items():
        prediction = sorted(v, key=lambda x: -x[0])[0][-1]
        which_reasoning = [name in prediction for name in reasoning_list]
        reasoning = reasoning_list[which_reasoning.index(True)]
        answer = prediction[prediction.index(reasoning)+len(reasoning):].strip()
        verifier_f1s[k] = f1(answer, id2question[k])
        reasoning_counter[reasoning[1:-1]]+=1
        final_prediction['answer'][k] = (answer, reasoning[1:-1])

    ### Final results from Decomposition Scorer
    print ("=== final ===")
    print_result(verifier_f1s)

    with open(args.prediction_file, 'w') as f:
        json.dump(final_prediction, f)

if __name__ == '__main__':
    main()


