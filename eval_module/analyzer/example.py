class Analyzer:
    def __init__(self):
        ...
    def analyse(self,output,label):
        """
        参数:
            output (dict): 来自于method的输出，表示方法的推理结果。
            label (dict): 来自dataset模块的标签，表示数据的正确标签。

        返回:
            res (dict): 用于传递给后续的summarizer模块，表示分析的结果。
        """
        res={
            'acc':1
        }
        return res
