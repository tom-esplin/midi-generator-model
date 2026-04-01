from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def create_mask(sequence_length,device):
    t = torch.full((sequence_length,sequence_length),-torch.inf)
    return torch.triu(t,1).to(device)
class PositionalEncoding(nn.Module):
    """Implements the PE function.
       The forward adds the positional encoding to the input encoding
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = 1 / (10000 ** (torch.arange(0., d_model, 2) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
    

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        # Implement Multi-head attention mechanism

        # Make h attention heads (linear layers) for q, k, and v
        self.attention_heads = nn.ModuleList([
            nn.ModuleList([nn.Linear(d_model, d_model // h),
             nn.Linear(d_model, d_model // h),
             nn.Linear(d_model, d_model // h)])
            for i in range(h)])
        self.final_linear = nn.Linear(d_model,d_model)

    def forward(self, q_seq_embedding, kv_seq_embedding, mask):
        # While q_seq_embedding == kv_seq_embedding in our lab, because we are performing self-attention,
        # we have separated them as two input parameters because it is not the only form of attention.
        outputs = []
        for item in self.attention_heads:
           query_tensor = item[0](q_seq_embedding)
           key_tensor = item[1](kv_seq_embedding)
           value_tensor = item[2](kv_seq_embedding)
           outputs.append(F.scaled_dot_product_attention(query_tensor, key_tensor, value_tensor, mask))
        return self.final_linear(torch.cat(outputs,2))
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d_model,d_ff),nn.ReLU(),nn.Linear(d_ff,d_model))
        self.multi_headed_attention = MultiHeadedAttention(h,d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        output = self.dropout(self.multi_headed_attention(x,x,mask)) + x
        norm_output = self.layer_norm1(output)
        output = norm_output + self.dropout(self.ff(norm_output))
        return self.layer_norm2(output)


class TransformerDecoder(nn.Module):
    "N layer decoder-only transformer."

    def __init__(self, vocab_size, N=4, d_model=256, d_ff=512, h=4, dropout=.1):
        """
        vocab_size - number of tokens in vocabulary
        N - number of DecoderLayer modules
        d_model - embedding dimension
        d_ff - feed-forward hidden dimension
        h - number of attention heads
        dropout - dropout probability: used in PositionalEncoding, PositionalFeedForward, and AddAndNorm
        """
        super().__init__()
        self.embedder = nn(vocab_size,d_model)
        self.pos_encoder = PositionalEncoding(d_model,dropout)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model,d_ff,h,dropout) for i in range(N)])
        self.final_linear = nn.Linear(d_model,vocab_size)
        # Initialize parameters w/ xavier for better performance
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask):
        embed_x = self.pos_encoder(self.embedder(x))

        decode_output = embed_x
        for layer in self.decoder_layers:
            decode_output = layer(decode_output, mask)

        return self.final_linear(decode_output)