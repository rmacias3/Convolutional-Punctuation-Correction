Punctuation Correction via Temporal Convolutional Networks + BERT Embeddings

1) Download the dataset from https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html

2) Make "books_root_dir" in GenPunctuationSet.py point to the directory where you downloaded 1)

3) Run GenPunctuationSet.py generate the sentence/punctuation pair dataset

4) Split the dataset generated in 3) into training/validation/test sets as you please.

5) Run train_segmenter.py to train the sentence punctuator.

Notes:
I received around 93% accuracy on the validation set. Keep in mind that the gutenberg corpus is not perfectly clean.
A half-logistic distribution is used to generate unpunctuated sentences of varying lengths.
