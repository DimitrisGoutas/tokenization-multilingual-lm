import torch
torch.manual_seed(2326)
from torch import nn
from torch.utils.data import DataLoader
from dataset import LanguageModelDataset, EvaluationDataset
from model import TransformerLanguageModel
from tokenizers import Tokenizer
from char_tokenizer import char_vocab, encode
import matplotlib.pyplot as plt

# load the tokenizers
bpe_2k_tokenizer = Tokenizer.from_file(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_2k.json"
)

bpe_10k_tokenizer = Tokenizer.from_file(
    "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/bpe_10k.json"
)

def char_encode(sentence):
    return encode(sentence, char_vocab)

def bpe_2k_encode(sentence):
    encoding = bpe_2k_tokenizer.encode(sentence)
    return encoding.ids

def bpe_10k_encode(sentence):
    encoding = bpe_10k_tokenizer.encode(sentence)
    return encoding.ids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# training
tokenizer_name = "bpe_10k"

data_paths = [
    "/srv/data/lt2326-h26/a1/train/en.txt",
    "/srv/data/lt2326-h26/a1/train/tr.txt",
    "/srv/data/lt2326-h26/a1/train/zh.txt"
]

if tokenizer_name == "bpe_2k":
    encode_function = bpe_2k_encode
    vocab_size = len(bpe_2k_tokenizer.get_vocab())
elif tokenizer_name == "bpe_10k":
    encode_function = bpe_10k_encode
    vocab_size = len(bpe_10k_tokenizer.get_vocab())
elif tokenizer_name == "char":
    encode_function = char_encode
    vocab_size = len(char_vocab)
else:
    raise ValueError("Unknown tokenizer name: " + tokenizer_name)

batch_size = 32
num_batches = 3000
validation_interval = 250

# create the dataset
dataset = LanguageModelDataset(encode_function, data_paths)
train_dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

# validation
valid_data_paths = [
    "/srv/data/lt2326-h26/a1/valid/en.txt",
    "/srv/data/lt2326-h26/a1/valid/tr.txt",
    "/srv/data/lt2326-h26/a1/valid/zh.txt"
]
valid_dataset = EvaluationDataset(encode_function, valid_data_paths)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False
)

# create the model
model = TransformerLanguageModel(vocab_size=vocab_size)
model = model.to(device)

# loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# train the model
model.train()

train_losses = []
valid_losses = []

for batch_number, (input_tokens, target_tokens) in enumerate(train_dataloader):
    if batch_number == num_batches:
        break

    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    # make predictions
    logits = model(input_tokens)
    loss = loss_fn(logits.reshape(-1, vocab_size), target_tokens.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # calculate validation loss every 250 batches
    if (batch_number + 1) % validation_interval == 0:
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for valid_input, valid_target in valid_dataloader:
                valid_input = valid_input.to(device)
                valid_target = valid_target.to(device)
                valid_logits = model(valid_input)
                loss = loss_fn(
                    valid_logits.reshape(-1, vocab_size),
                    valid_target.reshape(-1)
                )
                valid_loss += loss.item()
        valid_loss = valid_loss / len(valid_dataloader)
        valid_losses.append(valid_loss)
        print(
            "Batch:", batch_number + 1,
            "Training loss:", train_losses[-1],
            "Validation loss:", valid_loss
        )
        model.train()

print("Training finished.")
# save the model
model_path = ( "/home/gusgoutdi@GU.GU.SE/a1_tokenization/models/" + tokenizer_name + "_final.pt")

torch.save(model.state_dict(), model_path)

print("Model saved to:", model_path)

# plot training and validation loss
plt.plot(train_losses, label="Training loss")
validation_batches = list(range(validation_interval, num_batches + 1, validation_interval))
plt.plot(validation_batches, valid_losses, label="Validation loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Final " + tokenizer_name + " training")
plt.legend()
plt.savefig("/home/gusgoutdi@GU.GU.SE/a1_tokenization/plots/final_" + tokenizer_name + "_training_curve.png")
plt.close()