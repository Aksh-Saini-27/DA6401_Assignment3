import torch
from datasets import load_dataset
import spacy

class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.dataset = load_dataset("bentrevett/multi30k", split=split)
        
        # FAST TOKENIZATION: Using spacy.blank eliminates the need to download 
        # statistical models, preventing OSErrors and network timeouts!
        self.spacy_de = spacy.blank("de")
        self.spacy_en = spacy.blank("en")
        
        # Special tokens configuration
        self.special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
        self.unk_idx, self.pad_idx, self.sos_idx, self.eos_idx = 0, 1, 2, 3
        
        self.src_vocab = {}
        self.tgt_vocab = {}
        self.src_itos = {}
        self.tgt_itos = {}

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        train_data = load_dataset("bentrevett/multi30k", split='train')
        
        def create_vocab(data, spacy_nlp, lang_key):
            vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
            for example in data:
                # .tokenizer is called implicitly here
                for token in spacy_nlp(example[lang_key]):
                    word = token.text.lower()
                    if word not in vocab:
                        vocab[word] = len(vocab)
            return vocab
        
        self.src_vocab = create_vocab(train_data, self.spacy_de, 'de')
        self.tgt_vocab = create_vocab(train_data, self.spacy_en, 'en')
        
        self.src_itos = {idx: word for word, idx in self.src_vocab.items()}
        self.tgt_itos = {idx: word for word, idx in self.tgt_vocab.items()}

    def process_data(self):
        processed_data = []
        for example in self.dataset:
            src_tokens = [tok.text.lower() for tok in self.spacy_de(example['de'])]
            tgt_tokens = [tok.text.lower() for tok in self.spacy_en(example['en'])]
            
            src_indices = [self.sos_idx] + [self.src_vocab.get(tok, self.unk_idx) for tok in src_tokens] + [self.eos_idx]
            tgt_indices = [self.sos_idx] + [self.tgt_vocab.get(tok, self.unk_idx) for tok in tgt_tokens] + [self.eos_idx]
            
            processed_data.append({
                'src': torch.tensor(src_indices, dtype=torch.long),
                'tgt': torch.tensor(tgt_indices, dtype=torch.long)
            })
            
        return processed_data
