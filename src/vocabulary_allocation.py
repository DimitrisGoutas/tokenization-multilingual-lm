from tokenizers import Tokenizer
from collections import Counter
import pandas as pd

tokenizer_names = ["bpe_2k", "bpe_10k"]

threshold = 0.8

data_paths = {
    "en": "/srv/data/lt2326-h26/a1/train/en.txt",
    "tr": "/srv/data/lt2326-h26/a1/train/tr.txt",
    "zh": "/srv/data/lt2326-h26/a1/train/zh.txt"
}

results = []
for tokenizer_name in tokenizer_names:
    tokenizer = Tokenizer.from_file(
        "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/"
        + tokenizer_name + ".json"
    )

    # count how often each token appears in each language
    token_counts = {
        "en": Counter(),
        "tr": Counter(),
        "zh": Counter()
    }

    for language, path in data_paths.items():
        with open(path, "r", encoding="utf-8") as text:
            sentences = text.read().splitlines()
        for sentence in sentences:
            token_ids = tokenizer.encode(sentence).ids
            token_counts[language].update(token_ids)

    # total number of tokens in each language
    total_language_tokens = {
        "en": sum(token_counts["en"].values()),
        "tr": sum(token_counts["tr"].values()),
        "zh": sum(token_counts["zh"].values())
    }

    # count the number of tokens used by each language 
    allocation = {
        "en": 0,
        "tr": 0,
        "zh": 0,
        "shared": 0,
        "unused": 0
    }

    # store the tokens and their frequencies
    token_details = {
        "en": [],
        "tr": [],
        "zh": []
    }

    vocab = tokenizer.get_vocab()
    for token, token_id in vocab.items():
        # get how often a token appears in each language
        en_count = token_counts["en"][token_id]
        tr_count = token_counts["tr"][token_id]
        zh_count = token_counts["zh"][token_id]
        # calculate the proportion of this token used by each language
        total_count = en_count + tr_count + zh_count
        if total_count > 0:
            # normalize the token frequency by the total tokens in each language
            en_rate = en_count / total_language_tokens["en"]
            tr_rate = tr_count / total_language_tokens["tr"]
            zh_rate = zh_count / total_language_tokens["zh"]
            total_rate = en_rate + tr_rate + zh_rate
            probabilities = {
                "en": en_rate / total_rate,
                "tr": tr_rate / total_rate,
                "zh": zh_rate / total_rate
            }

            # find the language with the highest proportion
            language = max(probabilities, key=probabilities.get)
            max_probability = probabilities[language]

            # assign the token to a language if at least 80% of its
            # occurrences come from that language
            if max_probability >= threshold:
                allocation[language] += 1
                decoded_token = tokenizer.decode([token_id])
                token_details[language].append((decoded_token, en_count, tr_count, zh_count))
            else:
                allocation["shared"] += 1
        else:
            allocation["unused"] += 1

    print("Tokenizer:", tokenizer_name)

    print("Vocabulary allocation:")
    print("English:", allocation["en"])
    print("Turkish:", allocation["tr"])
    print("Chinese:", allocation["zh"])
    print("Shared:", allocation["shared"])
    print("Unused:", allocation["unused"])

    # 10 frequent examples for each language
    for language in ["en", "tr", "zh"]:
        if language == "en":
            position = 1
        elif language == "tr":
            position = 2
        else:
            position = 3
        token_details[language].sort(
            key=lambda x: x[position],
            reverse=True
        )

        print()
        print(language, "associated tokens:")

        for token in token_details[language][:10]:
            print(token)

    results.append({
        "tokenizer": tokenizer_name,
        "english": allocation["en"],
        "turkish": allocation["tr"],
        "chinese": allocation["zh"],
        "shared": allocation["shared"],
        "unused": allocation["unused"]
    })

    print()

df = pd.DataFrame(results)
df.to_csv("/home/gusgoutdi@GU.GU.SE/a1_tokenization/results/vocabulary_allocation.csv", index=False)
print(df)