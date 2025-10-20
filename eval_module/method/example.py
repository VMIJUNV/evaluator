class Method:
    def __init__(self):
        ...
    
    def inference(self,inp):
        """
        参数:
            inp (dict): 来自于method的输出，表示方法的输入数据。

        返回:
            out (dict): 用于传递给后续的analyzer模块，表示推理的结果。
        """
        out={
            'pred':1
        }
        return out
