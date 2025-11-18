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

            'batch_size':self.eval_config.get('batch_size',1),
            'inference_batch_size':self.eval_config.get('inference_batch_size',1),
            'analysis_batch_size':self.eval_config.get('analysis_batch_size',1),
            'threads':self.eval_config.get('threads',1),
            'inference_threads':self.eval_config.get('inference_threads',1),
            'analysis_threads':self.eval_config.get('analysis_threads',1),

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

    def add_inference_record(self,batch_record):
        for record in batch_record:
            record=dict((k, record[k]) for k in self.eval_config['inference_record_key'])
            if self.eval_config['save_record']:
                self.recorder.add_record(self.inference_records_path,record)
            else:
                self.inference_records[record['index']]=record

    def add_analysis_record(self,batch_record):
        for record in batch_record:
            record=dict((k, record[k]) for k in self.eval_config['analysis_record_key'])
            if self.eval_config['save_record']:
                self.recorder.add_record(self.analysis_records_path,record)
            else:
                self.analysis_records[record['index']]=record

    def inference_batch_record(self,batch_record):
        batch_input=[record['input'] for record in batch_record]
        batch_output=self.method.inference(batch_input)
        return batch_output
    def analysis_batch_record(self,batch_record):
        batch_output=[record['output'] for record in batch_record]
        batch_label=[record['label'] for record in batch_record]
        batch_analysis=self.analyzer.analyse(batch_output,batch_label)
        return batch_analysis

    def summary_records(self):
        self.summarizer = self.summarizer_cls(**self.summarizer_args)
        self.load_analysis_records()
        self.summarizer.summary(self.analysis_records,self.summary_path)

    def inference_batch_task(self,index_list):
        batch_index=[]
        for index in index_list:
            if index in self.inference_records:
                continue
            batch_index.append(index)
        
        if len(batch_index)==0:
            return
        
        batch_records=[]
        for index in batch_index:
            data = self.dataset[index]
            record={
                'index':index,
                'mark':data['mark'],
                'input':data['input'],
                'label':data['label'],
            }
            batch_records.append(record)
        
        batch_output=self.inference_batch_record(batch_records)
        for record,output in zip(batch_records,batch_output):
            record['output']=output
        
        self.add_inference_record(batch_records)
        
    def analysis_batch_task(self,index_list):
        batch_index=[]
        for index in index_list:
            if index in self.analysis_records:
                continue
            if index not in self.inference_records:
                continue
            batch_index.append(index)
        
        if len(batch_index)==0:
            return
        if index not in self.inference_records:
            return
        
        batch_records=[]
        for index in batch_index:
            data = self.dataset[index]
            record={
                'index':index,
                'mark':data['mark'],
                'input':data['input'],
                'label':data['label'],
            }
            inference_record=self.inference_records[index]
            record.update(inference_record)
            batch_records.append(record)
        
        batch_analysis=self.analysis_batch_record(batch_records)
        for record,analysis in zip(batch_records,batch_analysis):
            record["analysis"]=analysis
        
        self.add_analysis_record(batch_records)


    def eval_batch_task(self,index_list):
        batch_index=[]
        for index in index_list:
            if index in self.analysis_records:
                continue
            batch_index.append(index)
        
        if len(batch_index)==0:
            return
        
        batch_records=[]
        for index in batch_index:
            data = self.dataset[index]
            record={
                'index':index,
                'mark':data['mark'],
                'input':data['input'],
                'label':data['label'],
            }
            batch_records.append(record)
        
        batch_output=self.inference_batch_record(batch_records)
        for record,output in zip(batch_records,batch_output):
            record['output']=output
        
        batch_analysis=self.analysis_batch_record(batch_records)
        for record,analysis in zip(batch_records,batch_analysis):
            record["analysis"]=analysis
        
        self.add_analysis_record(batch_records)

    def executor(self,task_func,task_list,num_threads=1,batch_size=1):
        new_task_list = [task_list[i:i + batch_size] for i in range(0, len(task_list), batch_size)]
        if num_threads<=1:
            for batch_task in tqdm(new_task_list):
                task_func(batch_task)
        else:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(task_func,batch_task) for batch_task in new_task_list]
                pbar = tqdm(as_completed(futures), total=len(futures))
                for future in pbar:
                    future.result()

    def eval(self):
        mode=self.eval_config['mode']
        threads=self.eval_config['threads']
        inference_threads=self.eval_config['inference_threads']
        analysis_threads=self.eval_config['analysis_threads']
        batch_size=self.eval_config['batch_size']
        inference_batch_size=self.eval_config['inference_batch_size']
        analysis_batch_size=self.eval_config['analysis_batch_size']

        if mode=="one-step":
            logger.info(f"Start the evaluation.")
            self.eval_init()
            self.executor(self.eval_batch_task,self.all_tasks,num_threads=threads,batch_size=batch_size)
        elif mode=="two-step":
            logger.info("Start the inference step.")
            self.eval_inference_init()
            self.executor(self.inference_batch_task,self.all_tasks,num_threads=inference_threads,batch_size=inference_batch_size)
            logger.info("Start the analysis step.")
            self.eval_analysis_init()
            self.executor(self.analysis_batch_task,self.all_tasks,num_threads=analysis_threads,batch_size=analysis_batch_size)
        
        logger.info("Start summarizing and analyzing the results.")
        self.summary_records()
