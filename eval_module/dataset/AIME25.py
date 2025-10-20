import json

class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data=[]

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                temp={  
                        'mark':{},
                        'input':{
                            'question':item['problem'],
                        },
                        "label":{
                            "answer":item['answer'],
                        },
                    }
                self.data.append(temp)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


