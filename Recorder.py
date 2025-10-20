import threading
import json


class Recorder:
    def __init__(self):
        self.add_lock = threading.Lock()

    def read_records(self,records_path):
        records_path.touch(exist_ok=True)
        records=[]
        with open(records_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                records.append(item)
        return records
    
    def add_record(self,records_path,record):
        with self.add_lock:
            with open(records_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')