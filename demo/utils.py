import numpy as np
from collections import Counter, defaultdict
from hotpot_evaluate_v1 import normalize_answer

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

def get_answer_i(answer1_i, answer2_i):
    o1 = {o['text']:o['logit'] for o in filter_duplicate(answer1_i)}
    o2 = {o['text']:o['logit'] for o in filter_duplicate(answer2_i)}
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
    combined_answers = [{'text': t, 'logit': o1.get(t, -10000)+o2.get(t, -10000)} for t in list(set(o1.keys())|set(o2.keys()))]
    answer_i = sorted(combined_answers, key=lambda x: x['logit'], reverse=True)[0]
    return answer_i

