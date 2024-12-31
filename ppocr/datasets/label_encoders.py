import numpy as np
from ppocr.datasets.charset import alphabet


class BaseRecLabelEncode(object):
    """Convert between text-label and text-index"""

    def __init__(self, max_text_length):
        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"

        dict_character = list(alphabet)
        dict_character = self.add_special_char(dict_character)
        self.character = dict_character

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None

        text_list = []
        for char in text:
            if char not in self.dict:
                print('{} is not in dict'.format(char))
                return None            
            text_list.append(self.dict[char])

        if len(text_list) == 0:
            return None
        
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(
        self, max_text_length, **kwargs
    ):
        super(CTCLabelEncode, self).__init__(max_text_length)

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        
        # real length
        data["length"] = np.array(len(text))

        # pad to max length
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)

        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class NRTRLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(
        self, max_text_length, **kwargs
    ):
        super(NRTRLabelEncode, self).__init__(max_text_length)

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        
        if len(text) >= self.max_text_len - 1:
            return None
        
        # real length
        data["length"] = np.array(len(text))

        # insert start(<s>) and end(</s>) token's idx
        text.insert(0, 2)
        text.append(3)

        # pad to max length
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)

        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank", "<unk>", "<s>", "</s>"] + dict_character
        return dict_character