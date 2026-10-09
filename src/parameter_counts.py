from model import TransformerLanguageModel
from tokenizers import Tokenizer
from char_tokenizer import char_vocab

bpe_2k_tokenizer = Tokenizer.from_file(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_2k.json"
)

bpe_10k_tokenizer = Tokenizer.from_file(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_10k.json"
)

vocab_sizes = {
    "char": len(char_vocab),
    "bpe_2k": len(bpe_2k_tokenizer.get_vocab()),
    "bpe_10k": len(bpe_10k_tokenizer.get_vocab())
}

for model_name, vocab_size in vocab_sizes.items():
    model = TransformerLanguageModel(
        vocab_size=vocab_size
    )
    # total parameters
    total_params = sum(
        p.numel() for p in model.parameters()
    )
    # input embedding parameters
    input_embedding_params = sum(
        p.numel() for p in model.token_embedding.parameters()
    )
    # output layer parameters
    output_params = sum(
        p.numel() for p in model.output_layer.parameters()
    )

    print("Model:", model_name)
    print("Total parameters:", total_params)
    print("Input embedding parameters:", input_embedding_params)
    print("Output layer parameters:", output_params)
    print()