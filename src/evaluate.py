import torch
from torch import nn 
import math 
from model import TransformerLanguageModel
from dataset import EvaluationDataset
from tokenizers import Tokenizer
from char_tokenizer import char_vocab, encode, decode, id_to_character

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_name = "char"
language = "zh"

# data for evaluation
test_data_paths = [
    "/srv/data/lt2326-h26/a1/test/" + language + ".txt"
]

if model_name == "char":
    def encode_function(sentence):
        return encode(sentence, char_vocab)

    vocab_size = len(char_vocab)
elif model_name == "bpe_2k":
    tokenizer = Tokenizer.from_file(
        "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_2k.json"
    )

    def encode_function(sentence):
        return tokenizer.encode(sentence).ids

    vocab_size = len(tokenizer.get_vocab())
elif model_name == "bpe_10k":
    tokenizer = Tokenizer.from_file(
        "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_10k.json"
    )

    def encode_function(sentence):
        return tokenizer.encode(sentence).ids

    vocab_size = len(tokenizer.get_vocab())
else:
    raise ValueError("Unknown model name: " + model_name)

# use the non-overlapping chunks 
test_dataset = EvaluationDataset(
    encode_function,
    test_data_paths
)

# load the data
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Test dataset size:", len(test_dataset))
print("Test batches:", len(test_dataloader))

model = TransformerLanguageModel(vocab_size=vocab_size)
model = model.to(device)

# load the weights that I saved from the training
model.load_state_dict(
    torch.load(
        "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/" + model_name + "_final.pt",
        map_location=device,
        weights_only=True
    )
)

model.eval()
print(model_name, "model loaded.")

# compute the loss
loss_fn = nn.CrossEntropyLoss()
total_loss = 0
total_tokens = 0
total_characters = 0

# count the characters in the original test text
with open(test_data_paths[0], "r", encoding="utf-8") as text:
    test_sentences = text.read().splitlines()

for sentence in test_sentences:
    total_characters += len(sentence)

with torch.no_grad():
    for input_tokens, target_tokens in test_dataloader:
        input_tokens = input_tokens.to(device)
        target_tokens = target_tokens.to(device)
        logits = model(input_tokens)
        # calculate the loss for this batch
        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            target_tokens.reshape(-1)
        )
        # add this batch's loss and count its tokens
        number_of_tokens = target_tokens.numel()
        total_loss += loss.item() * number_of_tokens
        total_tokens += number_of_tokens
# calculate the average loss per token
test_loss = total_loss / total_tokens
# convert the loss from nats to bits
test_loss_bits = test_loss / math.log(2)
total_bits = test_loss_bits * total_tokens
# find the tokens at the end that were not included in a full evaluation chunk
remaining_tokens = test_dataset.token_ids[total_tokens:]

print("Remaining tokens:", remaining_tokens)

if model_name == "char":
    remaining_text = decode(remaining_tokens.tolist(), id_to_character)
else:
    remaining_text = tokenizer.decode(remaining_tokens.tolist())

print("Remaining text:", remaining_text)
print("Remaining characters:", len(remaining_text))

print("All tokenized tokens:", len(test_dataset.token_ids))
# exclude the corresponding characters from the bpc calculation
evaluated_characters = total_characters - len(remaining_text)
bpc = total_bits / evaluated_characters
print("Evaluated characters:", evaluated_characters)

print("Total loss:", total_loss)
print("Total tokens:", total_tokens)
print("Total characters:", total_characters)
print("Total bits:", total_bits)
print("Average test loss:", test_loss) # score per token
print("BPC:", bpc) # score per character