class SquadExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 all_answers=None,
                 start_position=None,
                 end_position=None,
                 keyword_position=None,
                 switch=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.all_answers=all_answers
        self.start_position = start_position
        self.end_position = end_position
        self.keyword_position = keyword_position
        self.switch = switch

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "question: "+self.question_text
        return s

class InputFeatures(object):

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 doc_tokens,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 keyword_position=None,
                 switch=None,
                 answer_mask=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.doc_tokens = doc_tokens
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.keyword_position = keyword_position
        self.switch = switch
        self.answer_mask = answer_mask




def find_span_from_text(context, tokens, answer):
    if answer.strip() in ['yes', 'no']:
        return [{'text': answer, 'answer_start': 0}]

    if answer not in context:
        return []

    offset = 0
    spans = []
    scanning = None
    process = []

    for i, token in enumerate(tokens):
        while context[offset:offset+len(token)] != token:
            offset += 1
            if offset >= len(context):
                break
        if scanning is not None:
            end = offset + len(token)
            if answer.startswith(context[scanning[-1]:end]):
                if context[scanning[-1]:end] == answer:
                    spans.append(scanning[0])
                elif len(context[scanning[-1]:end]) >= len(answer):
                    scanning = None
            else:
                scanning = None
        if scanning is None and answer.startswith(token):
            if token == answer:
                spans.append(offset)
            if token != answer:
                scanning = [offset]
        offset += len(token)
        if offset >= len(context):
            break
        process.append((token, offset, scanning, spans))

    answers = []

    for span in spans:
        if context[span:span+len(answer)] != answer:
            print (context[span:span+len(answer)], answer)
            print (context)
            assert False
        answers.append({'text': answer, 'answer_start': span})
    #if len(answers)==0:
    #    print ("*"*30)
    #    print (context, answer)
    return answers

def detect_span(_answers, context, doc_tokens, char_to_word_offset):
    orig_answer_texts = []
    start_positions = []
    end_positions = []
    switches = []

    if 'answer_start' not in _answers[0]:
        answers = []
        for answer in _answers:
            answers += find_span_from_text(context, doc_tokens, answer['text'])
    else:
        answers = _answers

    for answer in answers:
        orig_answer_text = answer["text"]
        answer_offset = answer["answer_start"]
        answer_length = len(orig_answer_text)

        if orig_answer_text in ["yes", "no"]:
            start_position, end_position = 0, 0
            switch = 1 if orig_answer_text == "yes" else 2
        else:
            switch = 0
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length - 1]

        orig_answer_texts.append(orig_answer_text)
        start_positions.append(start_position)
        end_positions.append(end_position)
        switches.append(switch)

    return orig_answer_texts, switches, start_positions, end_positions


