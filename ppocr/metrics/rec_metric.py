import os, sys
grand_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(grand_dir)

import torch
import string
from rapidfuzz.distance import Levenshtein
from ppocr.postprocess.rec_postprocess import CTCDecoder
from ppocr.datasets.label_converter import decode


class RecMetric(object):
    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 **kwargs
                 ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.pred_decoder = CTCDecoder().decode
        self.label_decoder = decode
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()
    
    def __call__(self, preds, labels):
        probs, preds = torch.max(preds, dim=2)

        with torch.no_grad():
            correct_num = 0
            all_num = 0
            norm_edit_dis = 0.0

            for (pred, prob, target) in zip(preds, probs, labels):
                # decode
                pred = self.pred_decoder(pred.detach().cpu().numpy(), prob.detach().cpu().numpy())
                target = self.label_decoder(target.detach().cpu().numpy())

                if self.ignore_space:
                    pred = pred.replace(' ', '')
                    target = target.replace(' ', '')

                if self.is_filter:
                    pred = self._normalize_text(pred)
                    target = self._normalize_text(target)

                # compute edit distance
                norm_edit_dis += Levenshtein.normalized_distance(pred, target)
                if pred == target:
                    correct_num += 1

                all_num += 1

            self.correct_num += correct_num
            self.all_num += all_num
            self.norm_edit_dis += norm_edit_dis
            return {
                'acc': correct_num / (all_num + self.eps),
                'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps)
            }
    
    def get_metric(self):
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        self.reset()
        return {'acc': acc, 'norm_edit_dis': norm_edit_dis}
    
    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
