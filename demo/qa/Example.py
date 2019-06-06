
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








