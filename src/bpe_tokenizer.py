from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

with open("/srv/data/lt2326-h26/a1/tokenizer/balanced.txt", "r") as text:
    sentences = text.read().splitlines()

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False) # remove space at the beginning of the sentence
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(vocab_size=10000)
tokenizer.train_from_iterator(sentences, trainer)

# save the tokenizer
tokenizer.save("/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_10k.json")