"""Spawned solver process. GUI events are applied between accepted steps."""
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import time
import traceback
from .config import Config
from .recording import Recorder
from .solver import Simulation


def worker_main(config_data, directory, commands, frames, notices, restart=None):
    sim = recorder = None
    try:
        sim = Simulation.load(restart) if restart else Simulation(Config.from_dict(config_data))
        recorder = Recorder(directory, sim)
        recorder.record(sim, force=True)
        notices.put({"type":"ready", "points":sim.ops.points, "triangles":sim.ops.triangles,
                     "config":sim.config.to_dict(),"directory":str(directory)})
        running = False
        completed = sim.t >= sim.config.t_end-1e-12
        dirty, last_frame = True, 0.0
        def publish():
            packet = {"c":sim.c.copy(),"diagnostics":sim.diagnostics(),"running":running,
                      "complete":completed,"growth_rate":sim.growth_rate}
            try:
                frames.get_nowait()
            except Empty:
                pass
            try:
                frames.put_nowait(packet)
            except Full:
                pass
        while True:
            # Bound command handling so a flood cannot permanently starve stepping.
            for _ in range(32):
                try:
                    command = commands.get_nowait()
                except Empty:
                    break
                action = command.get("action")
                try:
                    if action == "shutdown":
                        recorder.finish(sim)
                        return
                    elif action == "run":
                        running = not completed
                    elif action == "pause":
                        running = False
                    elif action == "growth":
                        sim.set_growth_rate(command["rate"])
                    elif action == "perturb":
                        if completed:
                            raise ValueError("Run has ended. Reset or use a longer t_end before perturbing.")
                        sim.perturb(**command["parameters"])
                    elif action == "save":
                        sim.checkpoint(Path(directory)/"checkpoint.npz")
                    else:
                        raise ValueError(f"Unknown command: {action}")
                    sim.events.append({"time":sim.t,"type":"control","action":action})
                    recorder.record(sim, force=action in {"save","pause","growth","perturb"})
                    notices.put({"type":"ack","action":action,"time":sim.t})
                    dirty = True
                except (ValueError, KeyError) as exc:
                    notices.put({"type":"command_error","message":str(exc)})
            if running:
                sim.step()
                recorder.record(sim)
                completed = sim.t >= sim.config.t_end-1e-12
                if completed:
                    running = False
                    recorder.record(sim, force=True)
                    sim.checkpoint(Path(directory)/"checkpoint.npz")
                    notices.put({"type":"complete","time":sim.t})
                    dirty = True
            now = time.monotonic()
            if dirty or (running and now-last_frame >= 0.25):
                publish()
                last_frame, dirty = now, False
            if not running:
                time.sleep(0.025)
    except Exception:
        notices.put({"type":"error","message":traceback.format_exc()})
    finally:
        if recorder is not None and sim is not None:
            try:
                recorder.finish(sim)
            except Exception:
                notices.put({"type":"error","message":traceback.format_exc()})
        if sim is not None:
            sim.close()
        # Never wait on an abandoned visualization consumer at process exit.
        frames.cancel_join_thread()


class Worker:
    def __init__(self, config, directory, restart=None):
        context = mp.get_context("spawn")
        self.commands = context.Queue(maxsize=64)
        self.frames = context.Queue(maxsize=1)
        self.notices = context.Queue()
        self.process = context.Process(target=worker_main, args=(config.to_dict(),str(directory),self.commands,self.frames,self.notices,restart), daemon=False)
        self.process.start()
        self.closed = False

    def send(self, action, **kwargs):
        if not self.process.is_alive():
            raise RuntimeError("Simulation worker is not running. Reset to create a new worker.")
        try:
            self.commands.put_nowait({"action":action, **kwargs})
        except Full as exc:
            raise RuntimeError("Command queue is full; wait for pending controls") from exc

    def close(self):
        if self.closed:
            return
        if self.process.is_alive():
            try:
                self.send("shutdown")
            except RuntimeError:
                pass
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)
        for queue in (self.commands,self.frames,self.notices):
            queue.cancel_join_thread()
            queue.close()
        self.closed = True
