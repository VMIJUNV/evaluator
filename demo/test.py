from dream_evaluator.Evaluator import Evaluator
from dream_evaluator.utils import auto_load_eval_modules



if __name__ == '__main__':

    eval_config={
        "project_name": 'test',
        "output_path": "test/output/",

        "resume": False,
        "max_version": 3,
        "mode": "two-step",
        "save_record": True
    }

    eval_modules={
        "dataset": {
            "cls": "AIME25",
            "args": {
                "data_path": f"test/AIME25_3.jsonl"
            }
        },
        "method": {
            "cls": 'OpenAIAPI',
            'args':{
                'api_key': '',
                'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'model_args':{
                    'model': 'qwen-flash',
                    'max_tokens': 512
                }
            }
        },
        "analyzer": {
            "cls": "QA",
            "args": {}
        },
        "summarizer": {
            "cls": "QA",
            "args": {}
        }
    }

    eval_modules = auto_load_eval_modules(eval_modules,eval_modules_path='eval_module')
    evaluator = Evaluator(eval_config,eval_modules)
    evaluator.eval()
