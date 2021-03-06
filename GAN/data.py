import nltk
from nltk.corpus import brown
import numpy as np
import torch

charset = {" ": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9,
           "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19,
           "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26}


def convert_data(minimum_len=10, maximum_len=100, fixed_len=216, save=False, return_tensor=False):
    sentences = brown.sents()
    instances = []
    max_len = 0
    for sentence in sentences:
        vectorized_sentence = np.zeros((1, len(charset.keys())))
        if minimum_len < len(sentence) <= maximum_len:
            for word in sentence:
                vector = convert_word(word)
                if vector is not None:
                    space_one_hot = np.zeros((1, len(charset.keys())))
                    space_one_hot[0, 0] = 1
                    vectorized_sentence = np.vstack((vectorized_sentence, vector, space_one_hot))
            vectorized_sentence = np.delete(vectorized_sentence, (0), axis=0)
            vectorized_sentence = vectorized_sentence[:-1, :]
            cur_len = len(vectorized_sentence)
            if len(vectorized_sentence) >= fixed_len:
                # padding = np.zeros((fixed_len - cur_len, len(charset.keys())))
                # vectorized_instance = np.vstack((vectorized_sentence, padding))
                instances.append(vectorized_sentence[:fixed_len])
    data = np.array(instances, dtype=int)
    print(data.shape, data.dtype)
    if save:
        np.save('brown_corpus.npy', data)
    if return_tensor:
        return torch.from_numpy(data)
    return data


def convert_word(word: str):
    word = word.lower()
    has_chars = set(word).intersection(set(charset.keys()))
    if has_chars:
        vectorized_word = []
        for character in word:
            try:
                value = charset[character]
                one_hot = np.zeros(len(charset.keys()))
                one_hot[value] = 1
                vectorized_word.append(one_hot)
            except KeyError:
                pass
        return np.array(vectorized_word)
    return None


def ceasar_shift(clear_text, key:int, save=False, return_tensor=False):
    limit = len(charset.keys())
    crypted_data = np.zeros(clear_text.shape, dtype=int)
    for i in range(len(clear_text)):
        for j in range(len(clear_text[i])):
            cur_val = np.argmax(clear_text[i][j])
            if cur_val > 0 or np.any(clear_text[i][j]):
                new_val = (cur_val + key) % limit
                crypted_data[i][j][new_val] = 1
    print(crypted_data.shape, crypted_data.dtype)
    if save:
        np.save('brown_corpus_ceasar_shift.npy', crypted_data)
    if return_tensor:
        return torch.from_numpy(data)
    return crypted_data




if __name__ == "__main__":
    data = convert_data(fixed_len=100, save=True)
    crypted_data = ceasar_shift(data,10, True)
    decrypted_data =  ceasar_shift(crypted_data,-10)
    data = data.reshape(-1)
    decrypted_data = decrypted_data.reshape(-1)
    sum_right = (decrypted_data == data).sum()
    print(sum_right, len(data))
    # print((result==result2).mean())



