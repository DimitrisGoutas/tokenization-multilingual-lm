from tokenizers import Tokenizer
import pandas as pd

# load the bpe tokenizers
bpe_2k_tokenizer = Tokenizer.from_file(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_2k.json"
)

bpe_10k_tokenizer = Tokenizer.from_file(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_10k.json"
)

# load the Chinese validation data
with open("/srv/data/lt2326-h26/a1/valid/zh.txt", "r", encoding="utf-8") as text:
    chinese_sentences = text.read().splitlines()

# check if a character is Chinese
def is_chinese_character(char):
    return "\u4e00" <= char <= "\u9fff"

# count Chinese characters
total_characters = 0

bpe_2k_one_token = 0
bpe_2k_multiple_tokens = 0

bpe_10k_one_token = 0
bpe_10k_multiple_tokens = 0

for sentence in chinese_sentences:
    for char in sentence:
        if not is_chinese_character(char):
            continue

        total_characters += 1
        # tokenize the character with bpe-2k
        bpe_2k_tokens = bpe_2k_tokenizer.encode(char).ids
        if len(bpe_2k_tokens) == 1:
            bpe_2k_one_token += 1
        else:
            bpe_2k_multiple_tokens += 1
        # tokenize the character with bpe-10k
        bpe_10k_tokens = bpe_10k_tokenizer.encode(char).ids
        if len(bpe_10k_tokens) == 1:
            bpe_10k_one_token += 1
        else:
            bpe_10k_multiple_tokens += 1

# calculate percentages
bpe_2k_one_percentage = bpe_2k_one_token / total_characters * 100
bpe_2k_multiple_percentage = bpe_2k_multiple_tokens / total_characters * 100

bpe_10k_one_percentage = bpe_10k_one_token / total_characters * 100
bpe_10k_multiple_percentage = bpe_10k_multiple_tokens / total_characters * 100

print("Total Chinese characters:", total_characters)

print()
print("BPE-2k:")
print("One token:", bpe_2k_one_token)
print("One token percentage:", bpe_2k_one_percentage)
print("Multiple tokens:", bpe_2k_multiple_tokens)
print("Multiple tokens percentage:", bpe_2k_multiple_percentage)

print()
print("BPE-10k:")
print("One token:", bpe_10k_one_token)
print("One token percentage:", bpe_10k_one_percentage)
print("Multiple tokens:", bpe_10k_multiple_tokens)
print("Multiple tokens percentage:", bpe_10k_multiple_percentage)


results = [
    {
        "tokenizer": "bpe_2k",
        "one_token": bpe_2k_one_token,
        "one_token_percentage": bpe_2k_one_percentage,
        "multiple_tokens": bpe_2k_multiple_tokens,
        "multiple_tokens_percentage": bpe_2k_multiple_percentage
    },
    {
        "tokenizer": "bpe_10k",
        "one_token": bpe_10k_one_token,
        "one_token_percentage": bpe_10k_one_percentage,
        "multiple_tokens": bpe_10k_multiple_tokens,
        "multiple_tokens_percentage": bpe_10k_multiple_percentage
    }
]

df = pd.DataFrame(results)
df.to_csv(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/results/chinese_fragmentation.csv",
    index=False
)

print()
print(df)