from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

class WordPieceTokenizer:
    def __init__(self, vocab_size=3000, min_frequency=2, pad=True):
        max_seq_len = 256
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.decoder = decoders.WordPiece()
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )
        if pad:
            self.tokenizer.enable_padding(length=max_seq_len)
            self.tokenizer.enable_truncation(max_length=max_seq_len)
        self.trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["[CLS]", "[SEP]", "[UNK]"],
        )

    def train(self, texts):
        self.tokenizer.train_from_iterator(texts, self.trainer)

    def tokenize(self, text):
        encoding = self.tokenizer.encode(text)
        return encoding.tokens
    
    def tokenize_ids(self, text):
        encoding = self.tokenizer.encode(text)
        return encoding.ids

if __name__ == '__main__':
    # Example usage:
    word_piece_tokenizer = WordPieceTokenizer()

    with open("./t8.shakespeare.txt", "r") as shakespeare:
        word_piece_tokenizer.train(shakespeare)
    # Train the tokenizer on your dataset

    # Tokenize a sentence
    # this one is from hamilton
    sentence = """LAURENS
    The ten-dollar founding father without a father
    got a lot farther by working a lot harder,
    by being a lot smarter,
    by being a self-starter,
    by fourteen, they placed him in charge of a
    trading charter.
    JEFFERSON
    And every day while slaves were being slaughtered and carted
    away across the waves, he struggled and kept his guard up.
    Inside, he was longing for something to be a part of,
    the brother was ready to beg, steal, borrow or barter.
    MADISON
    Then a hurricane came, and devastation reigned,
    our man saw his future drip, dripping down the drain,
    put a pencil to his temple, connected it to his brain,
    and he wrote his first refrain, a testament to his pain."""
    tokens = word_piece_tokenizer.tokenize(sentence)
    print(tokens)
    ints = word_piece_tokenizer.tokenize_ids(sentence)
    print(ints)