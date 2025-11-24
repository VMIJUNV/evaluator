from pathlib import Path
from tqdm import tqdm
import logging
import yaml
import shutil
import json
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from .module import Module
from .recorder import Recorder
from .config import EvaluatorConfig

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class Evaluator:
    @classmethod
    def load_from_dict(cls, config_dict: Dict[str, Any]):
        config=config_dict.get('evaluator',{})
        modules=config_dict.get('modules',{})
        config = EvaluatorConfig.load_from_dict(config)
        modules = Module.load_from_dict(modules)
        return cls(config,modules,config_dict)

    @classmethod
    def load_from_yaml(cls, config_path: str):
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls.load_from_dict(config_dict)

    def __init__(self,config:EvaluatorConfig,modules:Module,config_dict:Dict[str, Any]):
        self.config = config
        self.modules = modules
        self.output_path=Path(self.config.output_path)

        self.file_manage()
        if self.config.save_log:
            file_handler = logging.FileHandler(self.log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.setLevel(self.config.log_level)
        logger.info("Config:\n%s", json.dumps(config_dict, indent=4, ensure_ascii=False))

        self.dataset = self.modules.create_dataset()
        self.all_tasks=list(range(len(self.dataset)))
        self.recorder=Recorder()


    def file_manage(self):
        self.output_path.mkdir(parents=True, exist_ok=True)

        versions = []
        for item in self.output_path.iterdir():
            item_name=item.name.split('_')
            if len(item_name)==2 and item_name[0]=='version' and item_name[1].isdigit():
                versions.append({
                    'version':int(item_name[-1]),
                    'path':item
                })

        if len(versions)==0:
            self.current_version = {
                'version':0,
                'path':self.output_path / "version_0"
            }
        else:
            versions.sort(key=lambda x:x['version'])
            if self.config.resume:
                versions.append({
                    'version':versions[-1]['version']+1,
                    'path':self.output_path / f"version_{versions[-1]['version']+1}"
                })
                overflow_num=len(versions)-self.config.max_version
                if overflow_num > 0:
                    overflow_versions=versions[:overflow_num]
                    for version in overflow_versions:
                        shutil.rmtree(version['path'])
            self.current_version=versions[-1]

        self.save_path = self.current_version['path']
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.inference_records_path=self.save_path / "inference.jsonl"
        self.analysis_records_path=self.save_path / "analysis.jsonl"
        self.log_path=self.save_path / "log.txt"
        self.summary_path = self.save_path
        

    def eval_init(self):
        self.inference = self.modules.create_inference()
        self.analyzer = self.modules.create_analyzer()
        self.load_analysis_records()
        logger.info(f"Dataset size:{len(self.dataset)} Analysis completed:{len(self.analysis_records)}")

    def eval_inference_init(self):
        self.inference = self.modules.create_inference()
        self.load_inference_records()
        logger.info(f"Dataset size:{len(self.dataset)} Inference completed:{len(self.inference_records)}")

    def eval_analysis_init(self):
        del self.inference
        self.analyzer = self.modules.create_analyzer()
        self.load_inference_records()
        self.load_analysis_records()
        logger.info(f"Dataset size:{len(self.dataset)} Analysis completed:{len(self.analysis_records)}")


    def load_inference_records(self):
        if self.config.record_inference:
            self.inference_records={}
            records=self.recorder.read_records(self.inference_records_path)
            for record in records:
                index=record['index']
                self.inference_records[index]=record
        else:
            if not hasattr(self, 'inference_records'):
                self.inference_records = {}

    def load_analysis_records(self):
        if self.config.record_analysis:
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
            record=dict((k, record[k]) for k in self.config.inference_record_key)
            if self.config.record_inference:
                self.recorder.add_record(self.inference_records_path,record)
            else:
                self.inference_records[record['index']]=record

    def add_analysis_record(self,batch_record):
        for record in batch_record:
            record=dict((k, record[k]) for k in self.config.analysis_record_key)
            if self.config.record_analysis:
                self.recorder.add_record(self.analysis_records_path,record)
            else:
                self.analysis_records[record['index']]=record

    def batch_inference(self,batch_record):
        batch_input=[record['input'] for record in batch_record]
        batch_output=self.inference.inference(batch_input)
        return batch_output
    
    def batch_analysis(self,batch_record):
        batch_output=[record['output'] for record in batch_record]
        batch_label=[record['label'] for record in batch_record]
        batch_analysis=self.analyzer.analyse(batch_output,batch_label)
        return batch_analysis

    def summary_records(self):
        self.summarizer = self.modules.create_summarizer()
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
        
        batch_output=self.batch_inference(batch_records)
        for record,output in zip(batch_records,batch_output):
            record['output'] = output
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
        
        batch_records=[]
        for index in batch_index:
            data = self.dataset[index]
            record={
                'index':index,
                'mark':data['mark'],
                'input':data['input'],
                'label':data['label'],
            }
            inference_record = self.inference_records[index]
            record.update(inference_record)
            batch_records.append(record)
        
        batch_analysis=self.batch_analysis(batch_records)
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
        
        batch_output=self.batch_inference(batch_records)
        for record,output in zip(batch_records,batch_output):
            record['output']=output
        self.add_inference_record(batch_records)
        
        batch_analysis=self.batch_analysis(batch_records)
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

        if self.config.mode=="one step":
            logger.info(f"Start the evaluation.")
            self.eval_init()
            self.executor(self.eval_batch_task,self.all_tasks,num_threads=self.config.threads,batch_size=self.config.batch_size)
        elif self.config.mode=="two step":
            logger.info("Start the inference step.")
            self.eval_inference_init()
            self.executor(self.inference_batch_task,self.all_tasks,num_threads=self.config.inference_threads,batch_size=self.config.inference_batch_size)
            logger.info("Start the analysis step.")
            self.eval_analysis_init()
            self.executor(self.analysis_batch_task,self.all_tasks,num_threads=self.config.analysis_threads,batch_size=self.config.analysis_batch_size)
        elif self.config.mode=="only inference":
            logger.info("Start the inference step.")
            self.eval_inference_init()
            self.executor(self.inference_batch_task,self.all_tasks,num_threads=self.config.inference_threads,batch_size=self.config.inference_batch_size)
        elif self.config.mode=="only analysis":
            logger.info("Start the analysis step.")
            self.eval_analysis_init()
            self.executor(self.analysis_batch_task,self.all_tasks,num_threads=self.config.analysis_threads,batch_size=self.config.analysis_batch_size)
        
        if self.config.summary:
            logger.info("Start summarizing and analyzing the results.")
            self.summary_records()
