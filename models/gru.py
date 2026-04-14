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

    def forward(self, x, h_prev):
        if x.dim() == 3: x = x.squeeze(1)
        # 1. Get raw linear projections
        gi = self.W_i(x)      # [batch, 3*hidden]
        gh = self.W_h(h_prev) # [batch, 3*hidden]

        # 2. Chunk into r, z, n components
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        # 3. Apply the GRU gates
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)

        # Candidate memory (n) uses the reset gate on the hidden part
        new_memory = torch.tanh(i_n + (reset_gate * h_n))

        # 4. Final interpolation
        h_next = (1 - update_gate) * h_prev + update_gate * new_memory

        # Return two values: output and the new hidden state
        return h_next, h_next