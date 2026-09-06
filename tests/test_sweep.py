"""Exercise process scheduling, isolation and real offline worker outputs."""
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from mindvirus.config import Config, ModelConfig
from mindvirus.probes import hand_battery


def module():
    assert importlib.util.find_spec('mindvirus.sweep') is not None, 'GPU sweep launcher is missing'
    return importlib.import_module('mindvirus.sweep')


def inputs(tmp_path):
    configs=[]
    for name in ['llama-fixture','gemma-fixture']:
        p=tmp_path/f'{name}.yaml'
        cfg=Config(ModelConfig('fake',name), ModelConfig('fake',name),
                   n_agents=2,rounds=1,probe_every=1,battery_task='relation',probe_batch_size=2)
        p.write_text(yaml.safe_dump(cfg.to_dict())); configs.append(p)
    battery=tmp_path/'battery.json'
    hand_battery('honesty-absolutism',task='relation').save(battery)
    return configs,battery


def test_all_workers_start_before_waiting_and_receive_distinct_devices(tmp_path,monkeypatch):
    m=module(); configs,battery=inputs(tmp_path); events=[]
    class Process:
        def __init__(self,cmd,**kwargs):
            self.device=kwargs['env']['CUDA_VISIBLE_DEVICES']
            events.append(('start',self.device))
            self.pid=100+len(events)
        def wait(self,**kwargs):
            events.append(('wait',self.device)); return 0
    monkeypatch.setattr(m.subprocess,'Popen',Process)
    out=m.run_sweep(configs,['2','5'],battery,tmp_path/'sweep',seeds=[7,9])
    assert events==[('start','2'),('start','5'),('wait','2'),('wait','5')]
    manifest=json.loads((out/'manifest.json').read_text())
    assert [w['device'] for w in manifest['workers']]==['2','5']
    jobs=[j for w in manifest['workers'] for j in w['jobs']]
    assert len(jobs)==len({j['output'] for j in jobs})==4
    assert [j['seed'] for j in jobs]==[7,9,7,9]
    assert all(not Path(j['output']).is_absolute() for j in jobs)


def test_worker_failure_is_reported_as_failure(tmp_path,monkeypatch):
    m=module(); configs,battery=inputs(tmp_path)
    class Process:
        def __init__(self,cmd,**kwargs): self.pid=1; self.code=7 if kwargs['env']['CUDA_VISIBLE_DEVICES']=='1' else 0
        def wait(self,**kwargs): return self.code
    monkeypatch.setattr(m.subprocess,'Popen',Process)
    out=tmp_path/'sweep'
    with pytest.raises(RuntimeError,match='worker'):
        m.run_sweep(configs,['0','1'],battery,out)
    status=json.loads((out/'status.json').read_text())
    assert status['state']=='failed'
    assert [w['exit_code'] for w in status['workers']]==[0,7]


@pytest.mark.parametrize('devices,seeds',[(['0','0'],[0]),(['0'],[0]),(['0','1'],[1,1])])
def test_invalid_assignments_never_create_outputs(tmp_path,devices,seeds):
    m=module(); configs,battery=inputs(tmp_path)
    with pytest.raises(ValueError):m.run_sweep(configs,devices,battery,tmp_path/'sweep',seeds=seeds)
    assert not (tmp_path/'sweep').exists()


def test_sweep_never_overwrites_previous_run(tmp_path):
    m=module(); configs,battery=inputs(tmp_path)
    output=tmp_path/'existing';output.mkdir();(output/'sentinel').write_text('original')
    with pytest.raises(FileExistsError):m.run_sweep(configs,['0','1'],battery,output)
    assert (output/'sentinel').read_text()=='original'


def test_two_real_offline_workers_keep_complete_separate_results(tmp_path):
    m=module();configs,battery=inputs(tmp_path)
    out=m.run_sweep(configs,['0','1'],battery,tmp_path/'sweep',personas=1,repeats=1)
    manifest=json.loads((out/'manifest.json').read_text())
    pids=[]
    for worker in manifest['workers']:
        result=json.loads((out/f"worker-{worker['index']}"/'status.json').read_text())
        assert result['state']=='completed'
        pids.append(result['pid'])
        run=out/worker['jobs'][0]['output']
        rows=[json.loads(line) for line in (run/'observations.jsonl').read_text().splitlines()]
        assert len(rows)==32 and len({r['trial_id'] for r in rows})==32
        summary=json.loads((run/'summary.json').read_text())
        assert summary['empirical_model_run'] is False and summary['contagion_established'] is False
    assert len(set(pids))==2


def test_worker_reuses_model_and_keeps_seeded_control_outputs_separate(tmp_path,monkeypatch):
    m=module();configs,battery=inputs(tmp_path)
    out=m.run_sweep(configs[:1],['0'],battery,tmp_path/'sweep',stage='simulation',
                    seeds=[7,9],dry_run=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    builds=[]
    from mindvirus.backends import FakeBackend
    def build(*args,**kwargs):
        builds.append(True)
        return FakeBackend(default='<journal>My view.</journal><post>My post.</post>',
                           logprobs={'A':.6,'B':.4})
    monkeypatch.setattr(m,'build_backend',build)
    m._run_worker(out/'manifest.json',0)
    status=json.loads((out/'worker-0/status.json').read_text())
    assert len(builds)==1
    assert len(status['jobs'])==4
    for job in status['jobs']:
        cfg=yaml.safe_load((out/job['result']/'config.yaml').read_text())
        assert cfg['seed']==job['seed']
        assert cfg['n_patient_zero']==(1 if job['arm']=='seeded' else 0)
        assert (out/job['result']/'probes.jsonl').exists()


def test_frozen_battery_tampering_fails_before_model_load(tmp_path,monkeypatch):
    m=module();configs,battery=inputs(tmp_path)
    out=m.run_sweep(configs[:1],['0'],battery,tmp_path/'sweep',dry_run=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    (out/'battery.json').write_text('{}')
    monkeypatch.setattr(m,'build_backend',lambda *a,**k:pytest.fail('must not load'))
    with pytest.raises(ValueError,match='changed'):m._run_worker(out/'manifest.json',0)
    assert json.loads((out/'worker-0/status.json').read_text())['state']=='failed'


def test_timeout_terminates_owned_workers_and_records_failure(tmp_path,monkeypatch):
    m=module();configs,battery=inputs(tmp_path);processes=[]
    class Process:
        def __init__(self,*args,**kwargs):
            self.pid=100+len(processes);self.terminated=False;processes.append(self)
        def wait(self,**kwargs):
            if not self.terminated:raise m.subprocess.TimeoutExpired('worker',1)
            return -15
        def poll(self):return None
        def terminate(self):self.terminated=True
    monkeypatch.setattr(m.subprocess,'Popen',Process)
    with pytest.raises(m.subprocess.TimeoutExpired):
        m.run_sweep(configs,['0','1'],battery,tmp_path/'sweep',timeout_seconds=1)
    assert all(p.terminated for p in processes)
    status=json.loads((tmp_path/'sweep/status.json').read_text())
    assert status['state']=='failed'
    assert [w['exit_code'] for w in status['workers']]==[-15,-15]


def test_worker_cannot_reexecute_or_overwrite_an_existing_job(tmp_path,monkeypatch):
    m=module();configs,battery=inputs(tmp_path)
    out=m.run_sweep(configs[:1],['0'],battery,tmp_path/'sweep',personas=1,repeats=1,dry_run=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    m._run_worker(out/'manifest.json',0)
    before={str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):m._run_worker(out/'manifest.json',0)
    after={str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    assert after==before
