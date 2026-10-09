import pandas as pd
import matplotlib.pyplot as plt

# plot vocabulary allocation for the bpe tokenizers
vocab_data = pd.read_csv(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/results/vocabulary_allocation.csv"
)

vocab_data.plot(
    x="tokenizer",
    y=["english", "turkish", "chinese", "shared", "unused"],
    kind="bar"
)

plt.ylabel("Number of vocabulary tokens")
plt.xlabel("Tokenizer")
plt.title("Vocabulary allocation by language")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/plots/vocabulary_allocation.png"
)
plt.close()


# plot average characters per token
stats = pd.read_csv(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/results/tokenizer_stats.csv"
)

char_per_token = stats.pivot(
    index="tokenizer",
    columns="language",
    values="avg_char_per_token"
)

char_per_token.plot(kind="bar")

plt.ylabel("Average characters per token")
plt.xlabel("Tokenizer")
plt.title("Characters per token by language")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/plots/characters_per_token.png"
)
plt.close()


# plot test bpc for each model and language
bpc_data = pd.read_csv(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/results/bpc_results.csv"
)

bpc_data.plot(
    x="model",
    y=["english", "turkish", "chinese"],
    kind="bar"
)

plt.ylabel("Bits per character (BPC)")
plt.xlabel("Model")
plt.title("Language model performance by language")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/plots/bpc_results.png"
)
plt.close()