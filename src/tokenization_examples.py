from tokenizers import Tokenizer
from char_tokenizer import char_vocab, encode, decode, id_to_character

# load and read the first 3 sentences from each language
with open("/srv/data/lt2326-h26/a1/valid/en.txt", "r") as text:
    english_sentences = [text.readline().strip() for _ in range(3)]

with open("/srv/data/lt2326-h26/a1/valid/tr.txt", "r") as text:
    turkish_sentences = [text.readline().strip() for _ in range(3)]

with open("/srv/data/lt2326-h26/a1/valid/zh.txt", "r") as text:
    chinese_sentences = [text.readline().strip() for _ in range(3)]

sentences = {
    "English": english_sentences,
    "Turkish": turkish_sentences,
    "Chinese": chinese_sentences
}

# load both bpe tokenizers
bpe_2k_tokenizer = Tokenizer.from_file(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_2k.json"
)

bpe_10k_tokenizer = Tokenizer.from_file(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_10k.json"
)

for language, language_sentences in sentences.items():
    for sentence in language_sentences:
        print("Language:", language)
        print("Sentence:", sentence)

        # character tokenization
        encoded = encode(sentence, char_vocab)
        decoded = decode(encoded, id_to_character)
        print("Character tokens:")
        print(list(decoded))
        print("Number of character tokens:", len(encoded))

        # bpe-2K tokenization
        encoding = bpe_2k_tokenizer.encode(sentence)
        print("BPE-2K tokens:")
        print(encoding.tokens)
        print("Number of BPE-2K tokens:", len(encoding.ids))

        # bpe-10K tokenization
        encoding = bpe_10k_tokenizer.encode(sentence)
        print("BPE-10K tokens:")
        print(encoding.tokens)
        print("Number of BPE-10K tokens:", len(encoding.ids))

        print()