from typing import List, NamedTuple
from collections import defaultdict
from pyctcdecode import build_ctcdecoder

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float
        

class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

            
        alph = [""] + [c.upper() for c in self.alphabet]
        self.decoder = build_ctcdecoder(
            alph,
            kenlm_model_path="./3-gram.arpa",  # either .arpa or .bin file
            alpha=0.5,  # tuned on a val set
            beta=1.0,  # tuned on a val set
        )

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        s = ''
        EMPTY_IND = 0
        last_char = self.EMPTY_TOK
        for ind in inds:
            if ind == EMPTY_IND:
                last_char = self.ind2char[ind]
                continue
            if last_char != self.ind2char[ind]:
                s += self.ind2char[ind]
            last_char = self.ind2char[ind]
        return s
    
    def extend_and_merge(frame, state, ind2char):
        new_state = defaultdict(float)
        for next_char_index, next_char_proba in enumerate(frame):
            for (pref, last_char), pref_proba in state.items():
                next_char = ind2char[next_char_index]
                if next_char == last_char:
                    new_pref = pref
                else:
                    if next_char != EMPTY_TOK:
                        new_pref = pref + next_char
                    else:
                        new_pref = pref
                    last_char = next_char
                new_state[(new_pref, last_char)] += pref_proba * next_char_proba
                
    def truncate(state, beam_size):
        state_list = list(state.items())
        state_list.sort(key = lambda x: -x[1])
        return dict(state_list[:beam_size])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
            
        state = {('', EMPTY_TOK) : 1.0}
        for frame in probs:
            state = extend_and_merge(frame,state, self.ind2char)
            state = truncate(state, beam_size)
        state_list = list(state.items())
        state_list.sort(key = lambda x: -x[1])
        
        
        return [Hypothesis(text = v[0][0], prob = v[-1]) for v in state_list]

    def ctc_beam_search_lm(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        logits = [log_probs[i][:probs_length[i]].numpy() for i in range(len(probs_length))]
        with multiprocessing.get_context("fork").Pool() as pool:
            pred_list = self.lm_decoder.decode_batch(pool, logits, beam_width=beam_size)
        res = []
        for text in pred_list:
            res.append(text.lower())
        return res


# class CTCCharTextEncoder(CharTextEncoder):
#     EMPTY_TOK = "^"

#     def __init__(self, alphabet: List[str] = None):
#         super().__init__(alphabet)
#         vocab = [self.EMPTY_TOK] + list(self.alphabet)
#         self.ind2char = dict(enumerate(vocab))
#         self.char2ind = {v: k for k, v in self.ind2char.items()}

#     def ctc_decode(self, inds: List[int]) -> str:
#         # TODO: your code here
#         s = ''
#         EMPTY_IND = 0
#         last_char = self.EMPTY_TOK
#         for ind in inds:
#             if ind == EMPTY_IND:
#                 last_char = self.ind2char[ind]
#                 continue
#             if last_char != self.ind2char[ind]:
#                 s += self.ind2char[ind]
#             last_char = self.ind2char[ind]
#         return s
    
#     def extend_and_merge(frame, state, ind2char):
#         new_state = defaultdict(float)
#         for next_char_index, next_char_proba in enumerate(frame):
#             for (pref, last_char), pref_proba in state.items():
#                 next_char = ind2char[next_char_index]
#                 if next_char == last_char:
#                     new_pref = pref
#                 else:
#                     if next_char != EMPTY_TOK:
#                         new_pref = pref + next_char
#                     else:
#                         new_pref = pref
#                     last_char = next_char
#                 new_state[(new_pref, last_char)] += pref_proba * next_char_proba
                
#     def truncate(state, beam_size):
#         state_list = list(state.items())
#         state_list.sort(key = lambda x: -x[1])
#         return dict(state_list[:beam_size])

#     def ctc_beam_search(self, probs: torch.tensor, probs_length,
#                         beam_size: int = 100) -> List[Hypothesis]:
#         """
#         Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
#         """
#         assert len(probs.shape) == 2
#         char_length, voc_size = probs.shape
#         assert voc_size == len(self.ind2char)
#         hypos: List[Hypothesis] = []
            
#         state = {('', EMPTY_TOK) : 1.0}
#         for frame in probs:
#             state = extend_and_merge(frame,state, self.ind2char)
#             state = truncate(state, beam_size)
#         state_list = list(state.items())
#         state_list.sort(key = lambda x: -x[1])
        
        
#         return [Hypothesis(text = v[0][0], prob = v[-1]) for v in state_list]

    
    