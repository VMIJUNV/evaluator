import json
import pandas as pd

class Summarizer:
    def __init__(self):
        ...

    def summary(self,analysis_records,save_path):
        if len(analysis_records)==0:
            return None
        mark=[]
        analysis=[]
        for analysis_record in analysis_records.values():
            mark.append(analysis_record['mark'])
            analysis.append(analysis_record['analysis'])
        
        df_mark=pd.DataFrame(mark)
        df_analysis=pd.DataFrame(analysis)

        groups={
            'all':df_mark.index
        }
        mark_column_names = df_mark.columns
        for mark_column_name in mark_column_names:
            group_indices = df_mark.groupby(mark_column_name).groups
            group_indices_list = {f"{mark_column_name}_{k}": v for k, v in group_indices.items()}
            groups.update(group_indices_list)

        summary={
            "groups":[k for k,_ in groups.items()],
        }

        for group_name, group_indices in groups.items():
            df_analysis_group=df_analysis.loc[group_indices]

            df_metrics=df_analysis_group[['EM']]
            means = df_metrics.mean(numeric_only=True)
            summary[group_name]={
                'count':len(group_indices),
                'metrics':means.to_dict(),
            }
            
        with open(save_path / 'summary.json','w') as f:
            json.dump(summary,f,ensure_ascii=False,indent=4)
        
        return summary['all']