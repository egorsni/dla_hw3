# Don't forget to support cases when target_text == ''
import editdistance

def calc_wer(target_text, pred_text) -> float:
    if not target_text:
        if pred_text:
            return 1
        return 0
    target_text_splitted = target_text.split(' ')
    pred_text_splitted = pred_text.split(' ')
    
    return editdistance.eval(target_text_splitted, pred_text_splitted) / len(target_text_splitted)


def calc_cer(target_text, pred_text) -> float:
    # TODO: your code here
    return editdistance.eval(target_text, pred_text) / len(target_text)