from dataclasses import dataclass, field, fields
from typing import List, Dict, Any,Literal

@dataclass
class EvaluatorConfig:
    output_path: str
    mode: Literal['one step','two step','only inference','only analysis'] = 'one step'
    batch_size: int = 1
    inference_batch_size: int = 1
    analysis_batch_size: int = 1
    threads: int = 1
    inference_threads: int = 1
    analysis_threads: int = 1
    resume: bool = False
    max_version: int = 10
    record_inference: bool = True
    record_analysis: bool = True
    summary: bool = True
    inference_record_key: List[str] = field(default_factory=lambda: ['index', 'mark','input','output'])
    analysis_record_key: List[str] = field(default_factory=lambda: ['index', 'mark','output','label','analysis'])
    save_log: bool = True
    log_level: str = 'INFO'

    @classmethod
    def load_from_dict(cls, config_dict: Dict[str, Any]):
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)
