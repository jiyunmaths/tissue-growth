import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import sys
from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Growing tissue pattern laboratory")
    sub = parser.add_subparsers(dest="command",required=True)
    for name in ("dashboard","run","benchmark"):
        p = sub.add_parser(name)
        p.add_argument("--config",type=Path)
        p.add_argument("--backend",choices=["fenicsx","scipy"])
        p.add_argument("--geometry",choices=["square","sphere"])
        p.add_argument("--pc",choices=["jacobi","gamg","auto"],help="FEniCSx solver strategy (SciPy always uses Jacobi)")
        p.add_argument("--n",type=int,help="Square subdivisions per side or sphere subdivisions per octahedron edge")
        p.add_argument("--dt",type=float)
        p.add_argument("--t-end",type=float)
        p.add_argument("--output",type=Path,default=Path("runs") if name == "dashboard" else None)
        if name != "benchmark":
            p.add_argument("--restart",type=Path)
        if name == "dashboard":
            p.add_argument("--host",default="127.0.0.1")
            p.add_argument("--port",type=int,default=8080)
        if name == "benchmark":
            p.add_argument("--sizes",type=int,nargs="+",default=[32,64,128])
            p.add_argument("--steps",type=int,default=100)
    args = parser.parse_args()
    try:
        config = Config.read(args.config) if args.config else Config()
        changes = {key:getattr(args,key) for key in ("backend","geometry","pc","n","dt","t_end") if getattr(args,key) is not None}
        config = replace(config,**changes)
        if getattr(args,"restart",None):
            if args.config or changes:
                parser.error("--restart restores its configuration; do not combine it with configuration overrides")
            from .solver import Simulation
            # Read metadata without constructing a second solver in the dashboard process.
            import numpy as np
            with np.load(args.restart,allow_pickle=False) as checkpoint:
                config = Config.from_dict(json.loads(str(checkpoint["metadata"]))["config"])
        if args.command == "dashboard":
            from .dashboard import Dashboard
            print(f"Open http://{args.host}:{args.port} in your browser",flush=True)
            Dashboard(config,args.output,str(args.restart) if args.restart else None).start(args.host,args.port)
        elif args.command == "run":
            from .solver import Simulation
            from .recording import Recorder
            from datetime import datetime, timezone
            import uuid
            directory = args.output or Path("runs")/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:6])
            if directory.exists():
                parser.error(f"Output already exists: {directory}. Choose a new run directory.")
            sim = Simulation.load(args.restart) if args.restart else Simulation(config)
            recorder = None
            try:
                recorder = Recorder(directory,sim)
                recorder.record(sim,force=True)
                start = perf_counter()
                while sim.step():
                    recorder.record(sim)
                recorder.finish(sim)
                print(json.dumps({"output":str(directory.resolve()),"wall_seconds":perf_counter()-start,**sim.diagnostics()},indent=2))
            finally:
                if recorder is not None:
                    recorder.finish(sim)
                sim.close()
        else:
            if args.steps < 1:
                parser.error("--steps must be positive")
            from .solver import Simulation
            results = []
            for n in args.sizes:
                cfg = replace(config,n=n,t_end=max(config.t_end,(args.steps+2)*config.dt))
                sim = Simulation(cfg)
                try:
                    sim.step()  # Exclude first solver/preconditioner setup from warm timing.
                    start = perf_counter()
                    for _ in range(args.steps):
                        sim.step()
                    wall = perf_counter()-start
                    row = {"backend":cfg.backend,"geometry":cfg.geometry,"pc":cfg.pc,"dt":cfg.dt,
                           "growth_rate":cfg.growth_rate,"n":n,"measured_steps":args.steps,"warm_wall_seconds":wall,
                           "warm_ms_per_step":1000*wall/args.steps,**sim.diagnostics()}
                    results.append(row)
                    memory = f"{row['worker_rss_mb']:.1f} MB" if row["worker_rss_mb"] is not None else "RSS unavailable"
                    print(f"n={n}: {row['dofs']} DOFs, {row['warm_ms_per_step']:.3f} ms/step, {memory}",flush=True)
                finally:
                    sim.close()
            text = json.dumps(results,indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(text)
            else:
                print(text)
    except KeyboardInterrupt:
        return 130
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        parser.exit(1,f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
