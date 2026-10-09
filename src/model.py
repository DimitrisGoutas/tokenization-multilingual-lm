import torch
from torch import nn

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # same settings for every tokenizer, only vocab_size changes
        # so the comparison stays fair
        context_length=128
        embedding_dim=256
        num_heads=4
        num_layers=2
        feed_forward_dim=1024
        dropout=0.1

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # give each position its own vector
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        # set up one Transformer block
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        # turn the output into scores for all tokens
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids):
        token_embeddings = self.token_embedding(token_ids)
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        position_embeddings = self.position_embedding(positions)
        # combine token and position information
        x = token_embeddings + position_embeddings
        # hide future tokens
        sequence_length = token_ids.size(1)
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length,device=token_ids.device),
            diagonal=1
        ).bool()
        x = self.transformer(x, mask=mask)
        logits = self.output_layer(x)

        return logits