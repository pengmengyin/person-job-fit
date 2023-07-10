import torch
from torch import nn
from torch.nn import MultiheadAttention
from transformers import BertModel

class BERT_BiLSTM_Attention(nn.Module):
    def __init__(self, bert_model, hidden_size, dropout_prob):
        super(BERT_BiLSTM_Attention, self).__init__()
        self.hidden_dim = hidden_size
        self.activate = nn.ReLU()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.bilstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.bilstm_2 = nn.LSTM(2500, 2500, bidirectional=True, batch_first=True)
        self.muti_head_attention = MultiheadAttention(hidden_size,8)
        self.multihead_attention = nn.MultiheadAttention(2*768, num_heads=8)
        self.multihead_attention_2 = nn.MultiheadAttention(5000, num_heads=8)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=500, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=500, out_channels=500, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(4*hidden_size, 500)
        self.fc1 = nn.Linear(2*hidden_size, 5)
        self.fc2 = nn.Linear(2*hidden_size, 500)
        self.fc3 = nn.Linear(1500, 500)
        self.fc4 = nn.Linear(2500, 2500)
        self.fc5 = nn.Linear(2500, 2)

    def forward(self, input_ids1, input_ids2, token_type_ids1, token_type_ids2, attention_mask1, attention_mask2):
        bert_outputs = self.bert(input_ids=input_ids1, token_type_ids=token_type_ids1, attention_mask=attention_mask1)[
            0]
        bert_outputs1 = self.bert(input_ids=input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)[
            0]
        _,(lstm_output,_) = self.bilstm(bert_outputs)
        _,(lstm_output1,_)  = self.bilstm(bert_outputs1)

        lstm_output = lstm_output.permute(1, 0, 2).contiguous().view(-1, self.hidden_dim*2)
        lstm_output1 = lstm_output1.permute(1, 0, 2).contiguous().view(-1, self.hidden_dim*2)
        sequence_output,_ = self.multihead_attention(lstm_output.unsqueeze(0),lstm_output.unsqueeze(0),lstm_output.unsqueeze(0))
        sequence_output1,_ = self.multihead_attention(lstm_output1.unsqueeze(0), lstm_output1.unsqueeze(0), lstm_output1.unsqueeze(0))
        sequence_output=sequence_output.squeeze(0)
        sequence_output1=sequence_output1.squeeze(0)

        bert_output = bert_outputs.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        conv1_output = nn.functional.relu(self.conv1(bert_output))  # [batch_size, 128, seq_len]
        conv2_output = nn.functional.relu(self.conv2(conv1_output))  # [batch_size, 64, seq_len]
        pool_output = self.pool(conv2_output).squeeze(-1)  # [batch_size, 64]

        bert_output1 = bert_outputs1.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        conv1_output1 = nn.functional.relu(self.conv1(bert_output1))  # [batch_size, 128, seq_len]
        conv2_output1 = nn.functional.relu(self.conv2(conv1_output1))  # [batch_size, 64, seq_len]
        pool_output1 = self.pool(conv2_output1).squeeze(-1)  # [batch_size, 64]

        # fc = self.fc1(fc+sequence_output)
        # fc1 = self.fc1(fc1+sequence_output1)
        # fc = self.fc(concat)
        # fc1 = self.fc(concat1)
#0.755
        sequence_output = self.fc2(self.activate(sequence_output))
        sequence_output1 = self.fc2(self.activate(sequence_output1))
        lstm_output = self.fc2(self.activate(lstm_output))
        lstm_output1 = self.fc2(self.activate(lstm_output1))
        concat = torch.cat([sequence_output,lstm_output,pool_output],dim=1)
        concat1 = torch.cat([sequence_output1,lstm_output1,pool_output1],dim=1)
        fc = self.fc3(self.activate(concat))
        fc1 = self.fc3(self.activate(concat1))
        fc = torch.cat([fc,fc1,fc-fc1,fc+fc1,torch.abs(fc-fc1)],dim=1)
        # _,(lstm_output,_) = self.bilstm_2(fc.reshape(-1,1,2500))
        # lstm_output = lstm_output.permute(1, 0, 2).contiguous().view(-1, 5000)
        # sequence_output, _ = self.multihead_attention_2(lstm_output.unsqueeze(0), lstm_output.unsqueeze(0),
        #                                               lstm_output.unsqueeze(0))
        # sequence_output = sequence_output.squeeze(0)
        fc = self.fc4(self.activate(fc))
        fc = self.fc4(self.activate(fc))
        fc = self.fc5(self.activate(fc))
        return fc


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = input_dim // num_heads
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        query = self.query_layer(x).view(batch_size, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)
        key = self.key_layer(x).view(batch_size, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)
        value = self.value_layer(x).view(batch_size, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_probs, value)
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, seq_len, input_dim)
        output = self.output_layer(weighted_values)
        return output