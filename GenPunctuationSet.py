import os
import re
import ftfy
import pickle as pkl
from scipy.stats import halflogistic
from nltk import word_tokenize, sent_tokenize
from collections import deque
from math import floor

books_root_dir = ''
punctuation = {
    '?': 1,
    '.': 2,
    '...': 2,
    '..': 2,
    '!': 2,
    ',': 3,
    ';': 3,
    ':': 3
}
forbidden_characters = set(['``', '-', '--'])
contraction_creators = set(['\'re', '\'ll', '\'d', '\'n\'t', 'n\'t', '\'s', '\'t', '\'ve'])

def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()

def book_to_sentence_dset(filepath):
    with open(filepath, 'r') as f:
        my_string = f.read()
    cur_book_sentences = sent_tokenize(my_string)
    n_sentences = len(cur_book_sentences)
    dataset = []
    overlapping_sentences = []
    i = 0
    while i < n_sentences:
        additional_words_left = floor(halflogistic.rvs(size=1, scale=.8)[0] * 10)
        base_sentence = word_tokenize(text_standardize(ftfy.fix_text(cur_book_sentences[i])))
        j = i + 1
        while additional_words_left > 0 and j < n_sentences:
            cur_sentence = word_tokenize(text_standardize(ftfy.fix_text(cur_book_sentences[j])))
            additional_words_left -= len(cur_sentence)
            base_sentence += cur_sentence[:additional_words_left]
            j += 1
        overlapping_sentences.append((i, j - 1 if additional_words_left < 0 else j))
        i = overlapping_sentences[-1][1]
        training_phrase = deque()
        for t in range(len(base_sentence)):
            if base_sentence[t] in punctuation:
                if training_phrase:
                    training_phrase.pop()
                    training_phrase.append((base_sentence[t - 1], punctuation[base_sentence[t]]))
            else:
                if base_sentence[t] not in forbidden_characters:
                    if base_sentence[t] not in contraction_creators:
                        training_phrase.append((base_sentence[t], 0))
                    else:
                        if training_phrase:
                            half_word, _ = training_phrase.pop()
                            training_phrase.append((half_word + base_sentence[t], 0))
        dataset.append(training_phrase)
    return [dataset, overlapping_sentences]

if __name__ == '__main__':
    book_to_dset = {}
    for book_name in os.listdir(books_root_dir):
        book_to_dset = {book_name: book_to_sentence_dset(books_root_dir + book_name)}
        print('Processed ', book_name)
        with open('pickled_data/' + book_name.split('.txt')[0] + '.pkl', 'wb') as f:
            pkl.dump(book_to_dset, f)
