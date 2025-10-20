import importlib.util
from pathlib import Path

def get_class_from_module(module_path, class_name):

    spec = importlib.util.spec_from_file_location("temp_module", module_path)
    module = importlib.util.module_from_spec(spec)
    
    spec.loader.exec_module(module)
    
    if hasattr(module, class_name):
        return getattr(module, class_name)


current_file = Path(__file__).resolve()
eval_modules_default_path = current_file.parent.parent / 'eval_module'


def auto_load_eval_modules(eval_modules,eval_modules_path=eval_modules_default_path):
    eval_modules_path=Path(eval_modules_path)
    
    eval_module_type_list=['dataset','method','analyzer','summarizer']
    eval_module_cls_name_dict={
        'dataset':'Dataset',
        'method':'Method',
        'analyzer':'Analyzer',
        'summarizer':'Summarizer'
    }

    for eval_module_type, eval_module_config in eval_modules.items():
        if eval_module_type not in eval_module_type_list:
            continue

        eval_module_cls_name=eval_module_cls_name_dict[eval_module_type]
        eval_module_name=eval_module_config['cls']
        if isinstance(eval_module_name,str):
            eval_module_cls_path = eval_modules_path / eval_module_type / f'{eval_module_name}.py'
            eval_module_cls = get_class_from_module(eval_module_cls_path, eval_module_cls_name)
            eval_modules[eval_module_type]['cls']=eval_module_cls

        eval_modules[eval_module_type]['args']=eval_module_config['args']

    return eval_modules
