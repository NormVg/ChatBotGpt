from torch.utils.data import Dataset
import json

class ChatData(Dataset):
    def __init__(self, path:list, tokenizer):
        

        self.X = []
        for file in path:
            doc = json.load(open(file,"r",encoding="utf-8"))
            for i in doc :
                data = f"<startofstring> {i['q']} <AI>: {i['a']} <endofstring>"
                self.X.append(data)

        
        self.X_encoded = tokenizer(self.X,max_length=40, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])
    
