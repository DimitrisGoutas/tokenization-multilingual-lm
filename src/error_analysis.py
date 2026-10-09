import torch
from torch import nn
from tokenizers import Tokenizer
import math
from model import TransformerLanguageModel
from char_tokenizer import char_vocab, encode

# load the tokenizers 
bpe_2k_tokenizer = Tokenizer.from_file(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_2k.json"
)

bpe_10k_tokenizer = Tokenizer.from_file(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_10k.json"
)

# test Chinese sentence examples for my error analysis
test_path = "/srv/data/lt2326-h26/a1/test/zh.txt"

with open(test_path, "r", encoding="utf-8") as text:
    test_sentences = text.read().splitlines()

sentences = test_sentences[:3]

# tokenizer functions to get token ids
def char_encode(sentence):
    return encode(sentence, char_vocab)

def bpe_2k_encode(sentence):
    return bpe_2k_tokenizer.encode(sentence).ids

def bpe_10k_encode(sentence):
    return bpe_10k_tokenizer.encode(sentence).ids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# character model
char_model = TransformerLanguageModel(vocab_size=len(char_vocab)).to(device)

char_model.load_state_dict(
    torch.load(
        "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/char_final.pt",
        map_location=device,
        weights_only=True
    )
)

char_model.eval()

# bpe-2k model
bpe_2k_model = TransformerLanguageModel(vocab_size=len(bpe_2k_tokenizer.get_vocab())).to(device)

bpe_2k_model.load_state_dict(
    torch.load(
        "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_2k_final.pt",
        map_location=device,
        weights_only=True
    )
)

bpe_2k_model.eval()

# bpe-10k model
bpe_10k_model = TransformerLanguageModel(
    vocab_size=len(bpe_10k_tokenizer.get_vocab())
).to(device)

bpe_10k_model.load_state_dict(
    torch.load(
        "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_10k_final.pt",
        map_location=device,
        weights_only=True
    )
)

bpe_10k_model.eval()

loss_fn = nn.CrossEntropyLoss()
# calculates the bpc and number of tokens for one sentence
def sentence_bpc(sentence, encode_function, model, vocab_size):
    token_ids = encode_function(sentence)
    # shift the tokens to create inputs and targets
    input_tokens = torch.tensor(
        token_ids[:-1],
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    target_tokens = torch.tensor(
        token_ids[1:],
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    # get the predictions without updating the model
    with torch.no_grad():
        logits = model(input_tokens)
        loss = loss_fn(logits.reshape(-1, vocab_size),target_tokens.reshape(-1))

    # convert token loss to bits per character
    number_of_predictions = len(token_ids) - 1
    total_bits = (loss.item()* number_of_predictions/ math.log(2))
    bpc = total_bits / len(sentence)

    return bpc, len(token_ids)

# comparison of tokenizers and models
for sentence in sentences:
    print("Sentence:", sentence)

    #see how each tokenizer splits the sentence
    print("Character tokenization:")
    print(list(sentence))

    print("BPE-2k tokenization:")
    print(bpe_2k_tokenizer.encode(sentence).tokens)

    print("BPE-10k tokenization:")
    print(bpe_10k_tokenizer.encode(sentence).tokens)

    # character model
    char_bpc, char_tokens = sentence_bpc(
        sentence,
        char_encode,
        char_model,
        len(char_vocab)
    )

    print("Character BPC:", char_bpc)
    print("Character tokens:", char_tokens)

    # bpe-2k model
    bpe_2k_bpc, bpe_2k_tokens = sentence_bpc(
        sentence,
        bpe_2k_encode,
        bpe_2k_model,
        len(bpe_2k_tokenizer.get_vocab())
    )

    print("BPE-2k BPC:", bpe_2k_bpc)
    print("BPE-2k tokens:", bpe_2k_tokens)

    # bpe-10k model
    bpe_10k_bpc, bpe_10k_tokens = sentence_bpc(
        sentence,
        bpe_10k_encode,
        bpe_10k_model,
        len(bpe_10k_tokenizer.get_vocab())
    )

    print("BPE-10k BPC:", bpe_10k_bpc)
    print("BPE-10k tokens:", bpe_10k_tokens)