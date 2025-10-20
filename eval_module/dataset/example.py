class Dataset:
    def __init__(self):
        ...
        self.data_list=[]
    
    def __len__(self):
        ...
        return len(self.data_list)
    
    def __getitem__(self, index):

        data=self.data_list[index]
        data={
                'mark':{
                    'type':'test'
                },
                'input':{
                    'x':1
                },
                "label":{
                    'label':1
                }
            }
        return data