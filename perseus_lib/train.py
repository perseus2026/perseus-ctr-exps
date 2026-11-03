"""Stage one immutable recipe/seed and launch portable local training."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True,help='Prepared dataset root containing checkpoint/ and dataset/')
    p.add_argument('--workdir',type=Path,required=True,help='New output directory; must not exist')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--processes',type=int,default=1,help='Training processes on this machine, not allocated devices')
    p.add_argument('--microbatch',type=int)
    p.add_argument('--effective-batch',type=int,help='Defaults to recipe batch x accumulation for one process')
    p.add_argument('--workers',type=int,help='Data-loader workers PER training process')
    p.add_argument('--precision',choices=['no','fp16','bf16'])
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='Validate paths/config and print resolved settings; write nothing')
    a=p.parse_args()
    cfg=a.config.expanduser().resolve();prepared=a.prepared.expanduser().resolve();work=a.workdir.expanduser().resolve()
    if not cfg.is_file():p.error(f'Configuration missing: {cfg}')
    if not (prepared/'checkpoint/config.yaml').is_file() or not (prepared/'dataset').is_dir():p.error('prepared must contain checkpoint/config.yaml and dataset/')
    if work.exists():p.error('workdir already exists; select a new directory to avoid replacing a run')
    if a.processes<1 or (a.workers is not None and a.workers<0):p.error('processes must be positive and workers nonnegative')
    if a.cpu and a.processes!=1:p.error('this launcher supports one CPU process; distributed mode is multi-GPU')
    data=yaml.safe_load(cfg.read_text());t=data['training']
    target=a.effective_batch if a.effective_batch is not None else t['dataloader']['batch_size']*(t.get('accumulation') or 1)
    micro=a.microbatch if a.microbatch is not None else t['dataloader']['batch_size']
    if target<1 or micro<1 or target%(micro*a.processes):p.error('effective batch must be a positive multiple of microbatch x processes')
    t['seed']=a.seed;t['dataloader']['batch_size']=micro;t['test_dataloader']['batch_size']=micro;t['accumulation']=target//(micro*a.processes)
    if a.workers is not None:
        t['dataloader']['num_workers']=a.workers;t['test_dataloader']['num_workers']=a.workers
    precision=a.precision or t.get('mixed_precision','no');t['mixed_precision']=precision
    if a.cpu:
        os.environ['ACCELERATE_USE_CPU']='true'
        t['dataloader']['pin_memory']=False;t['test_dataloader']['pin_memory']=False
    from perseus import core
    normalized=core.Config.model_validate(data)
    old=core.Config.model_validate(yaml.safe_load((prepared/'checkpoint/config.yaml').read_text()))
    current_dump=normalized.model_dump(); old_dump=old.model_dump()
    for key in ['events','features','encoders','max_events_per_sequence','embedding_dim','prepare']:
        if current_dump.get(key)!=old_dump.get(key):p.error(f'prepared data contract differs for {key}; prepare this recipe separately')
    for location in ['event','context','artifacts']:
        if list(normalized.feature_to_encoder(**{location:True}))!=list(old.feature_to_encoder(**{location:True})):p.error(f'prepared {location} feature order differs')
    if normalized.task.type!=old.task.type:p.error('prepared task type differs')
    cmd=[sys.executable,'-m','accelerate.commands.launch','--num_processes',str(a.processes),'--num_machines','1','--mixed_precision',precision,'--dynamo_backend','no']
    if a.cpu:cmd+=['--cpu']
    elif a.processes>1:cmd+=['--multi_gpu']
    cmd+=['-m','perseus','train','fit-model','--fixed-padding','--workdir',str(work)]
    report={'seed':a.seed,'processes':a.processes,'microbatch':micro,'accumulation':t['accumulation'],'effective_batch':target,'workers_per_process':t['dataloader']['num_workers'],'mixed_precision':precision,'config_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),'command':cmd}
    print(json.dumps(report,indent=2),flush=True);print(shlex.join(cmd),flush=True)
    if a.dry_run:return
    work.mkdir(parents=True)
    shutil.copytree(prepared/'checkpoint',work/'checkpoint')
    (work/'dataset').symlink_to(prepared/'dataset',target_is_directory=True)
    (work/'checkpoint/config.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    (work/'launch.json').write_text(json.dumps(report,indent=2)+'\n')
    with (work/'train.log').open('w') as log:
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=os.environ.copy())
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        result=process.wait()
    raise SystemExit(result)

if __name__=='__main__':main()
