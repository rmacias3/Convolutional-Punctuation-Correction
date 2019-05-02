import os
import torch
from random import shuffle

import numpy as np
import pickle as pkl
import torch.optim as optim
import torch.nn.functional as F

from tcn import TemporalConvNet
from overrides import overrides
from allennlp.models import Model
from allennlp.data import Instance
from typing import Iterator, List, Dict
from allennlp.data.tokenizers import Token
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from pytorch_pretrained_bert import BertTokenizer
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.learning_rate_schedulers import NoamLR
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.iterators import BucketIterator, MultiprocessIterator
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder

torch.manual_seed(1)
MAX_WORD_LIST_LENGTH = 64
#Not placing token_type_ids here implies that we have a zeros tensor the same shape as the input passed in as described in the BERT paper.
bert_indexer = PretrainedBertIndexer('bert-base-uncased') 
token_embedding = PretrainedBertEmbedder('bert-base-uncased', top_layer_only=True)


class GutenbergDatasetReader(DatasetReader):
    """
    DatasetReader for Sentence Segmentation data, which is formatted as follows:

    'Zane Grey___The Young Forester.txt': [
        [
Ex #1       deque([('THE', 0), ('YOUNG', 0), ('FORESTER', 0), ('By', 0), ('Zane', 0),('Grey', 0), 
                ('I', 2), ('CHOOSING', 0), ('A', 0), ('Some', 0), ('way', 0)]),

Ex #2       deque([('Some', 0), ('way', 0), ('a', 0), ('grizzly', 0), ('bear', 0), ('would', 0), 
                ('come', 0), ('in', 0), ('when', 0), ('I', 0), ('tried', 0), ('to', 0), ('explain', 0), 
                    ('forestry', 0), ('to', 0), ('my', 0), ('brother', 2), ('Hunting', 0), ('grizzlies', 2), 
                        ('he', 0), ('cried', 0), ('Why', 0)]),

            ...
    """
    def __init__(self, 
                token_indexers: Dict[str, TokenIndexer] = None, 
                tokenizer: BertTokenizer = None, 
                lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        #Remember to change the maximum number of words here for the sentence segmentation task
        self.token_indexers = token_indexers or {"tokens": PretrainedBertIndexer('bert-base-uncased')}

    def text_to_instance(self, tokens: List[Token], tags: List[int] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}
        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field
        return Instance(fields)

    def _read(self, folder_path: str) -> Iterator[Instance]:
        books = [book for book in os.listdir(folder_path)]
        shuffle(books)
        not_gibberish = lambda x: all(_char not in x for _char in [')', '_', '(', '[', ']', '=e', '=a', '+', '|', '||', '*****', '--'])
        for idx, book in enumerate(books):
            with open(os.path.join(folder_path, book), 'rb') as f:
                book_data = pkl.load(f)
                book_name = next(iter(book_data.keys()))
                book_sentences = book_data[book_name][0]
                n = len(book_sentences)
                if n > 150:
                    for idx in range(50, max(0, n - 50)): #We usually get garbage for the first and last 50 or so sentences
                        sentence = book_sentences[idx]
                        if 0 < len(sentence) <= 64:
                            words, punctuation = zip(*sentence)
                            if not_gibberish(words):
                                yield self.text_to_instance([Token(word) for word in words], list(punctuation))

class TCNSegmenter(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 tcn: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 num_classes: int = None,
                 encoder_out_shape: int = None) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.word_embeddings.cuda(device=1)
        self.encoder = tcn
        # We don't need to apply log softmax here because it is applied inside of 
        # the given sequence_cross_entropy_with_logits function.
        self.hidden2tag = torch.nn.Linear(in_features=self.encoder.get_output_dim(), 
                                          out_features=num_classes if num_classes is not None else vocab.get_vocab_size("labels"))
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence).detach()
        encoder_out = self.encoder(embeddings.transpose(1, 2), sentence['tokens-offsets'])
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

class TCN(Seq2SeqEncoder):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.input_size = input_size
        self.num_channels = num_channels

    def forward(self, inputs, offsets):
        """Inputs have to have dimension (N, C_in, L_in)"""
        untrimmed_output = self.tcn(inputs).transpose(1, 2)
        gather_idx = offsets.unsqueeze(2).repeat(1, 1, self.input_size)
        return untrimmed_output.gather(1, gather_idx)
    
    @overrides
    def get_output_dim(self):
        return self.num_channels[-1]

    @overrides
    def get_input_dim(self):
        return self.input_size

    @overrides
    def is_bidirectional(self) -> bool:
        return False

def trainSentenceSegmenter():
    vocab_reader = GutenbergDatasetReader({'tokens': bert_indexer})
    vocab_train_set = vocab_reader.read('../Dataset/Training')
    vocab_validation_set = vocab_reader.read('../Dataset/Validation')
    bert_vocab = Vocabulary.from_instances(vocab_train_set + vocab_validation_set)
    # reader = GutenbergDatasetReader({'tokens': bert_indexer}, lazy=True)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding}, allow_unmatched_keys=True)
    n_hid, levels, emb_size = 1000, 5, 768
    num_channels = [n_hid] * (levels - 1) + [emb_size]
    kernel_size, dropout = 3, 0.1
    tcn = TCN(emb_size, num_channels, kernel_size, dropout)
    model = TCNSegmenter(word_embeddings, tcn, bert_vocab, num_classes=4)
    optimizer = optim.Adam(model.parameters())
    iterator = MultiprocessIterator(BucketIterator(batch_size=48, sorting_keys=[("sentence", "num_tokens")]), num_workers=5, output_queue_size=1500)
    iterator.index_with(bert_vocab)
    trainer = Trainer(model=model,
                    optimizer=optimizer,
                    iterator=iterator,
                    train_dataset=vocab_train_set,
                    validation_dataset=vocab_validation_set,
                    serialization_dir='serialized_models',
                    model_save_interval=60*45,
                    patience=2,
                    num_epochs=10,
                    cuda_device=0)
    trainer.train()


if __name__ == "__main__":
    trainSentenceSegmenter()