from pathlib import Path
from tqdm import tqdm
import logging

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from .Recorder import Recorder

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class Evaluator:
    def __init__(self,eval_config,eval_modules):
        self.eval_config = eval_config

        self.output_path=Path(self.eval_config['output_path'])
        self.project_name=self.eval_config['project_name']

        self.eval_config={
            'log_level':self.eval_config.get('log_level','INFO'),
            'resume':self.eval_config.get('resume',False),
            'max_version':self.eval_config.get('max_version',5),
            'mode':self.eval_config.get('mode','one-step'),
            'num_threads':self.eval_config.get('num_threads',1),
            'save_record':self.eval_config.get('save_record',True),        
            'inference_record_key':self.eval_config.get('inference_record_key',['index','output']),
            'analysis_record_key':self.eval_config.get('analysis_record_key',['index','mark','analysis']),
        }
    
        logger.setLevel(self.eval_config['log_level'])
        logger.debug(f"Evaluator config:{self.eval_config}")

        self.modules=eval_modules

        self.dataset_cls=self.modules['dataset']['cls']
        self.method_cls=self.modules['method']['cls']
        self.analyzer_cls=self.modules['analyzer']['cls']
        self.summarizer_cls=self.modules['summarizer']['cls']

        self.dataset_args=self.modules['dataset']['args']
        self.method_args=self.modules['method']['args']
        self.analyzer_args=self.modules['analyzer']['args']
        self.summarizer_args=self.modules['summarizer']['args']

        self.dataset = self.dataset_cls(**self.dataset_args)
        self.all_tasks=list(range(len(self.dataset)))

        self.recorder=Recorder()
        self.file_manage()


    def file_manage(self):
        self.project_path=self.output_path / self.project_name
        self.project_path.mkdir(parents=True, exist_ok=True)

        version_dirs = []
        version_nums = []
        for item in self.project_path.iterdir():
            if item.is_dir() and item.name.startswith('version_'):
                version_dirs.append(item)
                version_nums.append(int(item.name.split('_')[-1]))

        if version_nums==[]:
            self.current_version=0
        else:
            version_nums.sort()
            if self.eval_config['resume']:
                version_nums.append(version_nums[-1]+1)
            overflow_num=len(version_nums)-self.eval_config['max_version']
            if overflow_num>0:
                overflow_version_nums=version_nums[:overflow_num]
                for item in version_dirs:
                    if int(item.name.split('_')[-1]) in overflow_version_nums:
                        shutil.rmtree(item)
            self.current_version=version_nums[-1]

        self.save_path=self.project_path / f"version_{self.current_version}"
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.inference_records_path=self.save_path / "inference_records.jsonl"
        self.analysis_records_path=self.save_path / "analysis_records.jsonl"
        self.summary_path=self.save_path
        

    def eval_init(self):
        self.method = self.method_cls(**self.method_args)
        self.analyzer = self.analyzer_cls(**self.analyzer_args)
        self.load_analysis_records()

        logger.info(f"Dataset size:{len(self.dataset)} Analysis completed:{len(self.analysis_records)}")

    def eval_inference_init(self):
        self.method = self.method_cls(**self.method_args)
        self.load_inference_records()

        logger.info(f"Dataset size:{len(self.dataset)} Inference completed:{len(self.inference_records)}")

    def eval_analysis_init(self):
        del self.method
        self.analyzer = self.analyzer_cls(**self.analyzer_args)
        self.load_inference_records()
        self.load_analysis_records()

        logger.info(f"Dataset size:{len(self.dataset)} Analysis completed:{len(self.analysis_records)}")


    def load_inference_records(self):
        if self.eval_config['save_record']:
            self.inference_records={}
            records=self.recorder.read_records(self.inference_records_path)
            for record in records:
                index=record['index']
                self.inference_records[index]=record
        else:
            if not hasattr(self, 'inference_records'):
                self.inference_records = {}

    def load_analysis_records(self):
        if self.eval_config['save_record']:
            self.analysis_records={}
            records=self.recorder.read_records(self.analysis_records_path)
            for record in records:
                index=record['index']
                self.analysis_records[index]=record
        else:
            if not hasattr(self, 'analysis_records'):
                self.analysis_records = {}

    def add_inference_record(self,record):
        record=dict((k, record[k]) for k in self.eval_config['inference_record_key'])

        if self.eval_config['save_record']:
            self.recorder.add_record(self.inference_records_path,record)
        else:
            self.inference_records[record['index']]=record

    def add_analysis_record(self,record):
        record=dict((k, record[k]) for k in self.eval_config['analysis_record_key'])

        if self.eval_config['save_record']:
            self.recorder.add_record(self.analysis_records_path,record)
        else:
            self.analysis_records[record['index']]=record

    def inference_single_record(self,record):
        input=record['input']
        output=self.method.inference(input)
        return output

    def analysis_single_record(self,record):
        output=record['output']
        label=record['label']
        analysis=self.analyzer.analyse(output,label)
        return analysis

    def summary_records(self):
        self.summarizer = self.summarizer_cls(**self.summarizer_args)
        self.load_analysis_records()
        self.summarizer.summary(self.analysis_records,self.summary_path)

    def inference_single_task(self,index):
        if index in self.inference_records:
            return
        
        data = self.dataset[index]
        record={
            'index':index,
            'mark':data['mark'],
            'input':data['input'],
            'label':data['label'],
        }
        record['output'] = self.inference_single_record(record)

        self.add_inference_record(record)
        
    def analysis_single_task(self,index):
        if index in self.analysis_records:
            return
        if index not in self.inference_records:
            return
        
        data = self.dataset[index]
        record={
            'index':index,
            'mark':data['mark'],
            'input':data['input'],
            'label':data['label'],
        }
        inference_record=self.inference_records[index]
        record.update(inference_record)

        record["analysis"]=self.analysis_single_record(record)
        
        self.add_analysis_record(record)


    def eval_single_task(self,index):
        if index in self.analysis_records:
            return
        
        data = self.dataset[index]
        record={
            'index':index,
            'mark':data['mark'],
            'input':data['input'],
            'label':data['label'],
        }
        record['output']=self.inference_single_record(record)
        record["analysis"]=self.analysis_single_record(record)
        
        self.add_analysis_record(record)

    def executor(self,task_func,task_list,num_threads=8):
        if num_threads<=1:
            for task in tqdm(task_list):
                task_func(task)
        else:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(task_func,task) for task in task_list]
                pbar = tqdm(as_completed(futures), total=len(futures))
                for future in pbar:
                    future.result()

    def eval(self):
        mode=self.eval_config['mode']
        num_threads=self.eval_config['num_threads']

        if mode=="one-step":
            logger.info(f"Start the evaluation.")
            self.eval_init()
            self.executor(self.eval_single_task,self.all_tasks,num_threads=num_threads)
        elif mode=="two-step":
            logger.info("Start the inference step.")
            self.eval_inference_init()
            self.executor(self.inference_single_task,self.all_tasks,num_threads=num_threads)
            logger.info("Start the analysis step.")
            self.eval_analysis_init()
            self.executor(self.analysis_single_task,self.all_tasks,num_threads=num_threads)
        
        logger.info("Start summarizing and analyzing the results.")
        self.summary_records()
