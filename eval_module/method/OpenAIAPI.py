from openai import OpenAI

class Method:
    def __init__(self,api_key,base_url,model_args={}):
        self.api_key = api_key
        self.base_url = base_url
        self.model_args = model_args
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def inference(self,batch_input):
        batch_out=[]
        for input in batch_input:
            question=input['question']
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": question}
            ]
            response = self.client.chat.completions.create(
                messages=messages,
                **self.model_args
            )
            out={
                "answer":response.choices[0].message.content
            }
            batch_out.append(out)
        return batch_out
