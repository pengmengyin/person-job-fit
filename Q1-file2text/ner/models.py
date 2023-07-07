import torch
from TorchCRF import CRF

from transformers import BertModel


class bertLinear(torch.nn.Module):
    def __init__(self, config):
        super(bertLinear, self).__init__()
        self.model_name = "bertLinear"
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(in_features=self.config.bert_hid_size, out_features=self.config.label_num)

    def forward(self, bert_inputs):
        outputs = self.bert(bert_inputs, attention_mask=bert_inputs.ne(0).float())
        sequence_output, cls_output = outputs[0], outputs[1]
        sequence_output = self.dropout(sequence_output)
        outputs = self.linear(sequence_output)
        return outputs


class bertLSTM(torch.nn.Module):
    def __init__(self, config):
        super(bertLSTM, self).__init__()
        self.model_name = "bertLSTM"
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)
        self.lstm = torch.nn.LSTM(input_size=config.bert_hid_size, hidden_size=config.bert_hid_size, batch_first=True)
        self.linear = torch.nn.Linear(in_features=self.config.bert_hid_size, out_features=self.config.label_num)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, bert_inputs):
        outputs = self.bert(bert_inputs, attention_mask=bert_inputs.ne(0).float())
        sequence_output, cls_output = outputs[0], outputs[1]
        sequence_output = self.dropout(sequence_output)
        outputs, hidden = self.lstm(sequence_output)
        outputs = self.linear(outputs)
        return outputs


class bertBiLSTM(torch.nn.Module):
    def __init__(self, config):
        super(bertBiLSTM, self).__init__()
        self.model_name = "bertBiLSTM"
        self.config = config
        #加载预训练模型
        self.bert = BertModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)
        self.lstm = torch.nn.LSTM(
            input_size=config.bert_hid_size
            , hidden_size=config.bert_hid_size // 2
            , batch_first=True
            , bidirectional=True
        )
        self.linear1 = torch.nn.Linear(in_features=self.config.bert_hid_size, out_features=self.config.bert_hid_size)
        self.linear = torch.nn.Linear(in_features=self.config.bert_hid_size, out_features=self.config.label_num)
        #将全连接层的参数按照一定概率进行丢弃，防止过拟合
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, bert_inputs):
        outputs = self.bert(bert_inputs, attention_mask=bert_inputs.ne(0).float())
        #bert输出的last_hiden_state的size为(batch * lens * 768)  ,,,,(batch * 1 * 768)
        sequence_output, cls_output = outputs[0], outputs[1]
        sequence_output = self.dropout(sequence_output)
        #(batch * lens * 768)
        outputs, hidden = self.lstm(sequence_output)
        #(batch * lens * 768)  * (768 * config.label_num)
        #(batch * lens * config.label_num)                      (batch,label_nums)

        outputs = self.linear1(outputs)
        outputs = self.linear1(outputs)
        outputs = self.linear(outputs)
        return outputs


class bertCNN(torch.nn.Module):
    def __init__(self, config):
        super(bertCNN, self).__init__()
        self.model_name = "bertCNN"
        self.config = config
        #BertModel.from_pretrained，加载BERT模型
        self.bert = BertModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)
        self.base = torch.nn.Sequential(  #是 torch.nn 中的一种序列容器，参数会按照定义好的序列自动传递下去
            torch.nn.Dropout2d(0.1),  #随机将张量中的整个通道设置为零，概率0.1
            torch.nn.Conv2d(in_channels=self.config.bert_hid_size, out_channels=self.config.conv_hid_size,
                            kernel_size=1), #对有多个输入平面组成的输入信号进行二维卷积
                            #in_channels:输入图像通道数
                            #out_channels：卷积产生的通道数
                            #kernel_size：卷积核尺寸
            torch.nn.GELU()#激活函数
        )
        self.conv_ = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=self.config.conv_hid_size
                             , out_channels=self.config.conv_hid_size
                             , groups=self.config.conv_hid_size
                             , kernel_size=3, dilation=d, padding=d) for d in config.dilation]
                            #group：控制分组卷积，默认不分组，为1组。
                            #dilation：扩张操作：控制kernel点（卷积核点）的间距
                            #padding：填充操作：孔子padding_mode的数目

        )

        self.linear = torch.nn.Linear(in_features=self.config.conv_hid_size * len(config.dilation),
                                      out_features=self.config.label_num)
        self.dropout = torch.nn.Dropout(0.5) #将张量中的所有元素随机设置为0，概率0.5

    def forward(self, bert_inputs):
        sequence_output = self.bert(bert_inputs, bert_inputs.ne(0).byte())[0]
        sequence_output = sequence_output.unsqueeze(1).permute(0, 3, 1, 2)
        sequence_output = self.base(sequence_output)
        conv_outputs = list()
        for conv in self.conv_:
            conv_output = conv(sequence_output)
            conv_output = torch.nn.functional.gelu(conv_output)
            conv_outputs.append(conv_output)
        conv_output = torch.cat(conv_outputs, dim=1).permute(0, 2, 3, 1).squeeze(1)
        conv_output = self.linear(conv_output)
        return conv_output


class bertBiLSTM_CRF(torch.nn.Module):
    def __init__(self, config):
        super(bertBiLSTM_CRF, self).__init__()
        self.model_name = "bertBiLSTM_CRF"
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)
        self.lstm = torch.nn.LSTM(input_size=config.bert_hid_size, hidden_size=config.bert_hid_size, batch_first=True)
        self.linear = torch.nn.Linear(in_features=self.config.bert_hid_size, out_features=self.config.label_num)
        self.dropout = torch.nn.Dropout(0.1)
        self.crf = CRF(self.config.label_num)


    def forward(self, bert_inputs):
        outputs = self.bert(bert_inputs, attention_mask=bert_inputs.ne(0).float())
        sequence_output, cls_output = outputs[0], outputs[1]
        sequence_output = self.dropout(sequence_output)
        outputs, hidden = self.lstm(sequence_output)
        outputs = self.linear(outputs)
        return outputs

    def compute_loss(self, bert_inputs, bert_labels):
        out = self.forward(bert_inputs)
        loss = -self.crf(out, bert_labels)
        return loss

    def decode(self, inputs):
        out = self.forward(inputs)
        predict_index = self.crf.decode(out)
        return predict_index
