import torch

class SentClasDataset:
    def __init__(self, text, sentLabel, tokenizer, max_len):
        self.text = text 
        self.sentLabel = sentLabel
        self.tokenizer = tokenizer 
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            str(self.text[item]),
            max_legth=self.max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "text": str(self.text[item]),
            "input_ids": encoding["input_ids"].reshape(-1,),
            "attention_mask": encoding["attention_mask"].reshape(-1,),
            "sentLabels": torch.tensor(self.sentLabel[item], dtype=torch.long)
        }