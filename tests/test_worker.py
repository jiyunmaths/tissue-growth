import time
from queue import Empty
from tissue_growth.config import Config
from tissue_growth.worker import Worker
from tissue_growth.solver import Simulation


def wait_for(worker,kind,timeout=45,action=None):
    end=time.monotonic()+timeout
    while time.monotonic() < end:
        try:
            item=worker.notices.get(timeout=0.1)
        except Empty:
            continue
        assert item['type'] != 'error',item
        if item['type']==kind and (action is None or item.get('action')==action):
            return item
    raise AssertionError(f"Timed out waiting for {kind}: worker alive={worker.process.is_alive()}")


def test_worker_controls_and_checkpoint(tmp_path):
    directory=tmp_path/'run'
    worker=Worker(Config(backend='scipy',n=8,t_end=100,dt=0.001),directory)
    try:
        wait_for(worker,'ready')
        worker.send('growth',rate=0.03)
        wait_for(worker,'ack',action='growth')
        worker.send('perturb',parameters={'fraction':0.3})
        wait_for(worker,'ack',action='perturb')
        worker.send('run')
        wait_for(worker,'ack',action='run')
        time.sleep(0.1)
        worker.send('pause')
        paused=wait_for(worker,'ack',action='pause')
        assert paused['time'] > 0
        worker.send('save')
        saved=wait_for(worker,'ack',action='save')
        assert saved['time'] == paused['time']
        restored=Simulation.load(directory/'checkpoint.npz')
        try:
            assert restored.growth_rate == 0.03
            assert any(e['type']=='activator_depletion' for e in restored.events)
        finally:
            restored.close()
    finally:
        worker.close()
    assert not worker.process.is_alive()
    assert (directory/'summary.json').exists()
