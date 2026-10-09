from tokenizers import Tokenizer
import pandas as pd
from char_tokenizer import char_vocab, encode

# English valid txt
with open("/srv/data/lt2326-h26/a1/valid/en.txt", "r") as text:
    english_val_sentences = text.read().splitlines()

# Turkish valid txt
with open("/srv/data/lt2326-h26/a1/valid/tr.txt", "r") as text:
    turkish_val_sentences = text.read().splitlines()

# Chinese valid txt
with open("/srv/data/lt2326-h26/a1/valid/zh.txt", "r") as text:
    chinese_val_sentences = text.read().splitlines()

# store all sentences from every language into a dictionary 
languages = {
    "en": english_val_sentences,
    "tr": turkish_val_sentences,
    "zh": chinese_val_sentences
}

bpe_2k_tokenizer = Tokenizer.from_file("/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_2k.json")
bpe_10k_tokenizer = Tokenizer.from_file("/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_10k.json")

tokenizers = {
    "char": char_vocab,
    "bpe_2k": bpe_2k_tokenizer,
    "bpe_10k": bpe_10k_tokenizer
}

def calculate_stats(sentences, tokenizer, tokenizer_name):
    total_tokens = 0 
    total_char = 0
    for sentence in sentences:
        # calculate the total number of tokens
        # for char tokenizer
        if tokenizer_name == "char":
            token_ids = encode(sentence, char_vocab)
            total_tokens += len(token_ids)
        # for bpe tokenizer
        else:
            encoding = tokenizer.encode(sentence)
            total_tokens += len(encoding.ids)
        # calculate the total number of characters
        char_per_sentence = len(sentence)
        total_char += char_per_sentence

    # calculate the average number of tokens per sentence
    avg_tokens = total_tokens / len(sentences)
    # calculate the average number of characters per token
    avg_char_per_token = total_char / total_tokens

    if tokenizer_name == "char":
        vocab_size = len(char_vocab)
    else:
        vocab_size = len(tokenizer.get_vocab())

    return {
    "vocab_size": vocab_size,
    "total_tokens": total_tokens,
    "avg_tokens": avg_tokens,
    "avg_char_per_token": avg_char_per_token
}

results = {}
for language, sentences in languages.items():
    #print(language, len(sentences))
    for tokenizer_name, tokenizer in tokenizers.items():
        #print(tokenizer_name)
        stats = calculate_stats(sentences, tokenizer, tokenizer_name)
        #print(stats)
        if language not in results:
            results[language] = {}
        results[language][tokenizer_name] = stats

rows = []
for language in results:
    for tokenizer_name in results[language]:
        stats = results[language][tokenizer_name]
        rows.append({
            "language": language,
            "tokenizer": tokenizer_name,
            "vocab_size": stats["vocab_size"],
            "total_tokens": stats["total_tokens"],
            "avg_tokens": stats["avg_tokens"],
            "avg_char_per_token": stats["avg_char_per_token"]
        })

df = pd.DataFrame(rows)
df.to_csv("/home/gusgoutdi@GU.GU.SE/a1_tokenization/results/tokenizer_stats.csv", index=False)
print(df)
