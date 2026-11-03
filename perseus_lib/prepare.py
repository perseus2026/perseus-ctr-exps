"""Prepare a converted local dataset with the exact selected training recipe."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--converted',type=Path,required=True,help='Conversion root containing hub/ and basis/')
    p.add_argument('--workdir',type=Path,required=True,help='New prepared directory; must not exist')
    a=p.parse_args();cfg=a.config.expanduser().resolve();source=a.converted.expanduser().resolve();out=a.workdir.expanduser().resolve()
    if out.exists():p.error('workdir already exists; select a fresh prepared directory')
    if not cfg.is_file() or not (source/'hub').is_dir() or not (source/'basis/train/samples.pq').is_file():p.error('config or converted dataset missing')
    import polars as pl
    from perseus import core
    data=yaml.safe_load(cfg.read_text());core.Config.model_validate(data)
    for split in ['train','test']:
        context=pl.scan_parquet(source/f'basis/{split}/samples.pq').collect_schema()['context']
        names={f.name for f in context.fields}
        missing=set(k for k,v in data['features'].items() if v['located_in'].get('context'))-names
        if missing:p.error(f'{split}: converted context is missing {sorted(missing)}')
    out.mkdir(parents=True);shutil.copy2(cfg,out/'config.yaml');(out/'basis').symlink_to(source/'basis',target_is_directory=True)
    env=os.environ.copy();env['internal_storage_event_hub']=str(source/'hub')
    subprocess.run([sys.executable,'-m','perseus','train','prepare-dataset','--workdir',str(out)],check=True,env=env)
    (out/'preparation.json').write_text(json.dumps({'converter_contract':'one source row per sample; candidate_action constant0; no target in context','config':str(cfg),'converted':str(source)},indent=2)+'\n')
    print(f'Prepared: {out}')

if __name__=='__main__':main()
