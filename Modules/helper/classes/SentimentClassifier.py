from torch import nn 
import transformers

class SentimentClassifier(nn.Modules):
    def __init__(self, numClasses, pretrainedModel):
        super(SentimentClassifier, self).__init__()
        self.numClasses = numClasses
        self.pretrainedModel = pretrainedModel
        self.BERTmodel = transformers.BertModel.from_pretrained(self.pretrainedModel)
        self.dropOut = nn.Dropout(p=0.3)
        self.BiLSTM = nn.LSTM(
            self.BERTmodel.config.hidden_size,
            1024,
            bidirectional=True,
            batch_first=True
        )
        self.linear = nn.Linear(2*1024, self.numClasses)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        BERToutput = self.BERTmodel(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.dropOut(BERToutput["pooler_output"])
        output, _ = self.BiLSTM(output)
        output = self.linear(output)
        return self.softmax(output)