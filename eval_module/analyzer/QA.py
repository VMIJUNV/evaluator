class Analyzer:
    def __init__(self):
        ...
    def analyse(self,output,label):

        output_answer=output['answer']
        label_answer=str(label['answer'])

        EM=0
        if output_answer==label_answer:
            EM=1

        res={
            'EM':EM,
        }
        return res