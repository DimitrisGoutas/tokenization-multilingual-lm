with open("/srv/data/lt2326-h26/a1/tokenizer/balanced.txt", "r") as text:
    sentences = text.read().splitlines()

# builds a character vocabulary from the sentences
def build_char_vocab(sentences):
    characters = {}
    # 0 ID for unknown characters
    characters["<UNK>"] = 0
    token_id = 0
    for sentence in sentences:
        sentence = sentence.strip()
        for char in sentence:
            if char not in characters:
                token_id += 1
                characters[char] = token_id

    return characters

# maps IDs back to characters
def build_id_to_character(characters):
    id_to_character = {}
    for character, token_id in characters.items():
        id_to_character[token_id] = character

    return id_to_character

# convert characters to token IDs
def encode(sentence, characters):
    encoded = []
    for char in sentence:
        if char not in characters:
            char = "<UNK>"
        id_number = characters[char]
        encoded.append(id_number)
        
    return encoded

# convert token IDs back to characters
def decode(encoded, id_to_character):
    decoded = []
    for token_id in encoded:
        character = id_to_character[token_id]
        decoded.append(character)

    return "".join(decoded)

char_vocab = build_char_vocab(sentences)
id_to_character = build_id_to_character(char_vocab)
