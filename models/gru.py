from torch import nn
import torch
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru_blocks = nn.ModuleList([GRUBlock(embedding_dim, hidden_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        hidden_list = []
        output, hd  = self.gru_blocks[0](embedded, hidden[0])
        hidden_list.append(hd)
        for i,gru_block in enumerate(self.gru_blocks[1:]):
            output, hd = gru_block(output, hidden[i+1])
            hidden_list.append(hd)
        hidden = torch.stack(hidden_list)
        logits = self.fc(output)
        return logits, hidden
    
class GRUBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(GRUBlock, self).__init__()
        self.W_i = nn.Linear(embedding_dim, 3 * hidden_dim)
        self.W_h = nn.Linear(hidden_dim, 3 * hidden_dim)

    def forward(self, x, hidden):
        # Ensure x is [Batch, 256]
        if x.dim() == 3:
            x = x.squeeze(1)

        # 1. Transform both to the gate space (size 768)
        # x_projections: [Batch, 768]
        # h_projections: [Batch, 768]
        x_projections = self.W_i(x)
        h_projections = self.W_h(hidden)

        # 2. Chunk them into the three GRU components
        i_r, i_z, i_n = x_projections.chunk(3, dim=-1)
        h_r, h_z, h_n = h_projections.chunk(3, dim=-1)

        # 3. Calculate Gates
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        
        # 4. Calculate Candidate Hidden State
        # Note: reset_gate is applied to the hidden projection part
        candidate = torch.tanh(i_n + (reset_gate * h_n))

        # 5. Final State (Interpolation)
        # hidden is [Batch, 256], update_gate is [Batch, 256], candidate is [Batch, 256]
        # All are 256 now, so the math works!
        next_hidden = (1 - update_gate) * candidate + update_gate * hidden
        
        return next_hidden, next_hidden

class OptimizedGru(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        # 1. Turn integer token IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. The GRU itself (batch_first=True is crucial based on your loop)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # 3. Map the GRU's hidden state back to the vocabulary size for prediction
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # x shape: (batch_size, 1) -> from your loop: batch[:, i:i+1]
        
        embedded = self.embedding(x)
        # embedded shape: (batch_size, 1, embedding_dim)
        
        output, hidden = self.gru(embedded, hidden)
        # output shape: (batch_size, 1, hidden_dim)
        
        # Squeeze out the sequence dimension (which is 1) before the linear layer
        output = output.squeeze(1) 
        # output shape: (batch_size, hidden_dim)
        
        y_hat = self.fc(output)
        # y_hat shape: (batch_size, vocab_size)
        
        return y_hat, hidden