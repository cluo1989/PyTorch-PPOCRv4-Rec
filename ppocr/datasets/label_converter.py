from .charset import alphabet as charset
from .error_label_dict import errlabel

word2indexs = {}
indexs2word = {}

for idx, char in enumerate(charset):
    word2indexs[char] = idx
    indexs2word[idx] = char

def encode(word):
    indexs = []
    for c in word:
        try:
            if c in errlabel:
                c = errlabel[c]

            if len(c) == 1:
                idx = word2indexs[c]
                indexs.append(idx)
            else:
                for cc in c:
                    idx = word2indexs[cc]
                    indexs.append(idx)
        except:
            return None
        
    return indexs

def decode(indexs):
    chars = []
    for idx in indexs:
        chars.append(charset[idx])

    return ''.join(chars)


if __name__ == '__main__':
    word = 'Ⅱ Pytorch 多机多卡训练方法。'
    print(f'input:{word}')

    indexs = encode(word)
    print(f'codes:{indexs}')

    words = decode(indexs)
    print(f'words:{words}')
