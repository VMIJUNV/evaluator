from abc import ABC, abstractmethod
from dataclasses import dataclass,field
from typing import Type, Dict, Any
import importlib.util

class BaseDataset(ABC):
    @abstractmethod
    def __len__(self):
        ...
    @abstractmethod
    def __getitem__(self, index):
        ...

class BaseInference(ABC):
    @abstractmethod
    def inference(self):
        ...

class BaseAnalyzer(ABC):
    @abstractmethod
    def analyse(self):
        ...

class BaseSummarizer(ABC):
    @abstractmethod
    def summary(self):
        ...


def get_class_from_module(module_path, class_name):
    spec = importlib.util.spec_from_file_location("temp_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, class_name):
        return getattr(module, class_name)


@dataclass
class Module():
    dataset_cls: Type[BaseDataset] = None
    inference_cls: Type[BaseInference] = None
    analyzer_cls: Type[BaseAnalyzer] = None
    summarizer_cls: Type[BaseSummarizer] = None

    dataset_init_kwargs: dict = field(default_factory=dict)
    inference_init_kwargs: dict = field(default_factory=dict)
    analyzer_init_kwargs: dict = field(default_factory=dict)
    summarizer_init_kwargs: dict = field(default_factory=dict)

    def create_dataset(self):
        return self.dataset_cls(**self.dataset_init_kwargs)
    def create_inference(self):
        return self.inference_cls(**self.inference_init_kwargs)
    def create_analyzer(self):
        return self.analyzer_cls(**self.analyzer_init_kwargs)
    def create_summarizer(self):
        return self.summarizer_cls(**self.summarizer_init_kwargs)
    

    @classmethod
    def load_from_dict(cls, module_dict: Dict[str, Any]):
        dataset_cls_path = module_dict.get('dataset_cls_path', None)
        dataset_cls = module_dict.get('dataset_cls')
        if isinstance(dataset_cls, str):
            dataset_cls = get_class_from_module(dataset_cls_path, dataset_cls)
        dataset_init_kwargs = module_dict.get('dataset_init_kwargs',{})

        inference_cls_path = module_dict.get('inference_cls_path', None)
        inference_cls = module_dict.get('inference_cls')
        if isinstance(inference_cls, str):
            inference_cls = get_class_from_module(inference_cls_path, inference_cls)
        inference_init_kwargs = module_dict.get('inference_init_kwargs',{})

        analyzer_cls_path = module_dict.get('analyzer_cls_path', None)
        analyzer_cls = module_dict.get('analyzer_cls')
        if isinstance(analyzer_cls, str):
            analyzer_cls = get_class_from_module(analyzer_cls_path, analyzer_cls)
        analyzer_init_kwargs = module_dict.get('analyzer_init_kwargs',{})

        summarizer_cls_path = module_dict.get('summarizer_cls_path', None)
        summarizer_cls = module_dict.get('summarizer_cls')
        if isinstance(summarizer_cls, str):
            summarizer_cls = get_class_from_module(summarizer_cls_path, summarizer_cls)
        summarizer_init_kwargs = module_dict.get('summarizer_init_kwargs',{})

        return cls(
            dataset_cls=dataset_cls,
            inference_cls=inference_cls,
            analyzer_cls=analyzer_cls,
            summarizer_cls=summarizer_cls,
            dataset_init_kwargs=dataset_init_kwargs,
            inference_init_kwargs=inference_init_kwargs,
            analyzer_init_kwargs=analyzer_init_kwargs,
            summarizer_init_kwargs=summarizer_init_kwargs,
        )
    