import sentencepiece as spm
from .Tokenizer import Tokenizer

class SPTokenizer(Tokenizer):
    
    # sentencepiece tokenizer

    def __init__(self, spm_path):

        self.model = spm.SentencePieceProcessor(spm_path)

    def encode(self, s):
        return self.model.encode(s)

    def decode(self, l):
        return self.model.decode(l)

    def bos_id(self):
        return self.model.bos_id()
    
    def eos_id(self):
        return self.model.eos_id()

    def unk_id(self):
        return self.model.unk_id()

    def pad_id(self):
        return self.model.pad_id()

    def vocab_size(self):
        return self.model.vocab_size()
    
    def id2token(self, l):
        return self.model.id_to_piece(l)
    