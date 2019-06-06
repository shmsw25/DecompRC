import json
import numpy as np
from collections import Counter, defaultdict
from flask import Flask, render_template, redirect, request, jsonify, make_response

from qa.my_main import DecompRC
from utils import get_answer_i

app = Flask(__name__)
model = DecompRC(batch_size=50)
cached_output = {}

with open('hotpot_examples.json', 'r') as f:
    data = json.load(f)

contextss = data['contextss']
context_questions = data['context_questions']
titles = data['titles']
all_contextss = data['all_contextss']
all_questions = data['all_questions']

def get_answer(paragraphs, question, reasoningType):
    (q1_b, q2_b), (q1_i, q2_i) = model.get_output("span-predictor", question, paragraphs)

    if reasoningType==0:
        print ("Only run bridging")
        answer1_b = model.get_output("qa", [q1_b], paragraphs)[0][0]
        q2_b = q2_b.replace('[ANSWER]', answer1_b['text'])
        answer2_b = model.get_output("qa", [q2_b], paragraphs)[0][0]
        return {'q_type': 'Bridging',
                  'subq1': q1_b, 'answer1': answer1_b['text'],
                            'subq2': q2_b, 'answer2': answer2_b['text']}

    if reasoningType==1:
        print ("Only run intersection")
        answer1_i, answer2_i = model.get_output("qa", [q1_i, q2_i], paragraphs)
        answer_i = get_answer_i(answer1_i, answer2_i)
        return {'q_type': 'Intersection', 'subq1': q1_i, 'subq2': q2_i,
                           'answer2': answer_i['text']}


    answer1_b, answer1_i = model.get_output("qa", [q1_b, q1_i], paragraphs)
    answer1_b = answer1_b[0]

    q2_b = q2_b.replace('[ANSWER]', answer1_b['text'])
    answer2_b, answer2_i = model.get_output("qa", [q2_b, q2_i], paragraphs)
    answer2_b = answer2_b[0]
    answer_i = get_answer_i(answer1_i, answer2_i)
    final_pred = model.get_output("classifier", ["{} {} ({}) {}".format(q1_b, q2_b, 'bridge', answer2_b['text']),
                                    "{} {} ({}) {}".format(q1_i, q2_i, 'intersec', answer_i['text'])], paragraphs)
    #print (answer1_b['logit']+answer2_b['logit'], answer_i['logit'])
    #if answer1_b['logit']+answer2_b['logit']>answer_i['logit']:
    if final_pred[0][2] == 0:
        return {'q_type': 'Bridging',
                  'subq1': q1_b, 'answer1': answer1_b['text'],
                            'subq2': q2_b, 'answer2': answer2_b['text']}
    return {'q_type': 'Intersection', 'subq1': q1_i, 'subq2': q2_i,
                           'answer2': answer_i['text']}

def get_key(paragraphs, question, reasoningType):
    return (paragraphs.replace('\n', '').replace(' ', '').lower(),  question.lower(), reasoningType)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/select', methods=['GET', 'POST'])
def select():
    return jsonify(result={"contextss" : contextss,
        "titles": titles,
        "context_questions": context_questions,
        "all_contextss": all_contextss,
        'all_questions': all_questions})

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    paragraphs = request.args.get('paragraphs')
    question = request.args.get('question')
    reasoningType = int(request.args.get('reasoningType'))

    if get_key(paragraphs, question, reasoningType) in cached_output:
        answer = cached_output[get_key(paragraphs, question, reasoningType)]
    else:
        answer = get_answer([p for p in paragraphs.split('\n') if len(p.strip())>0],
                        question,
                        reasoningType)
    return jsonify(result=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2019, threaded=True)







