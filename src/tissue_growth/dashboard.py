import asyncio
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
import uuid
import numpy as np
from .worker import Worker


CSS = """
body{margin:0;background:#f0f3f5;color:#183344;font-family:Inter,system-ui,sans-serif}
.shell{display:grid;grid-template-columns:300px minmax(0,1fr);height:calc(100vh - 64px);overflow:hidden}
.sidebar{background:#fff;border-right:1px solid #dce5e8;padding:22px;overflow:auto}
.eyebrow{font-size:11px;letter-spacing:2px;color:#428b84;font-weight:700;margin-bottom:10px}
.panel-title{font-size:15px;font-weight:650;margin:22px 0 12px}
.note{font-size:12px;line-height:1.6;color:#627787;margin:8px 0 15px}
.row{display:flex;gap:8px;align-items:center;margin:10px 0}
.btn{background:#eaf0f3;color:#1c3d50;border:0;border-radius:7px;padding:10px 13px;cursor:pointer;font-size:13px;font-weight:600}
.btn.primary{background:#087f78;color:white}.btn:disabled{opacity:.4;cursor:default}
.workspace{padding:20px;display:flex;flex-direction:column;gap:14px;min-width:0;overflow:auto}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.metric{background:white;border:1px solid #e1e8ec;border-radius:10px;padding:13px 16px}
.metric-label{font-size:11px;color:#66808e;text-transform:uppercase;letter-spacing:1px}
.metric-value{font-size:23px;font-weight:550;margin-top:3px;font-variant-numeric:tabular-nums}
.view-card{background:#122632;border-radius:12px;overflow:hidden;position:relative;min-height:400px;flex:1;display:flex;flex-direction:column}
.view-toolbar{display:flex;justify-content:space-between;align-items:center;color:#e8f4f6;padding:10px 16px;font-size:13px}
.chart-card{background:#fff;border:1px solid #e1e8ec;border-radius:10px;padding:12px 16px}
.chart-card svg{display:block;width:100%;height:110px}
.view-toolbar{flex-shrink:0}
.status{font-size:12px;color:#56717f;white-space:pre-wrap;overflow-wrap:anywhere}
.error{color:#aa3232;background:#fff1ed;padding:10px;border-radius:6px;white-space:pre-wrap;max-height:160px;overflow:auto;font-size:12px}
@media(max-width:850px){.shell{grid-template-columns:245px 1fr}.sidebar{padding:12px}.metrics{grid-template-columns:repeat(2,1fr)}}
"""


def chart_svg(history, key="std_u", label="Activator spatial standard deviation"):
    if len(history) < 2:
        return '<svg viewBox="0 0 760 110" role="img"><text x="20" y="55" fill="#718493" font-size="13">Run the simulation to trace changes over time.</text></svg>'
    times = np.array([d["time"] for d in history])
    values = np.array([d[key] for d in history])
    high = max(float(values.max())*1.1, 0.001)
    x = 50 + 680*(times-times[0])/max(times[-1]-times[0],1e-12)
    y = 78-65*values/high
    points = " ".join(f"{a:.2f},{b:.2f}" for a,b in zip(x,y))
    return f'''<svg viewBox="0 0 760 110" role="img" aria-label="{label} over simulation time">
    <path d="M50 10V80H735" fill="none" stroke="#d8e3e8"/>
    <polyline points="{points}" fill="none" stroke="#078b80" stroke-width="2.5"/>
    <g font-size="11" fill="#67808c"><text x="0" y="18">{high:.3g}</text><text x="27" y="81">0</text>
    <text x="50" y="100">{times[0]:.1f}</text><text x="700" y="100">{times[-1]:.1f}</text>
    <text x="330" y="103">Simulation time</text></g></svg>'''


class Dashboard:
    def __init__(self, config, output_root, restart=None, organization=None):
        import pyvista as pv
        from trame.app import get_server
        from trame.ui.vuetify3 import SinglePageLayout
        from trame.widgets import html, vtk, client, vuetify3 as v3
        self.config, self.output_root, self.restart_path = config, Path(output_root), restart
        self.organization = organization
        aim1 = organization is not None
        self.spatial = spatial = aim1 and organization.get("model") == "spatial"
        spherical = config.geometry == "sphere"
        size_name = "radius" if spherical else "side length"
        self.worker = None
        self.task = None
        self.grid = self.actor = self.latest = None
        self.history = deque(maxlen=500)
        self.server = get_server(client_type="vue3")
        self.state = state = self.server.state
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background("#122632")
        self.plotter.view_xy()
        self.plotter.enable_parallel_projection()
        state.update({"busy":True,"running":False,"complete":False,"status":"Starting simulation worker…", "error":"",
                      "field":"u", "fields":[{"title":"Activator u","value":"u"},{"title":"Inhibitor v","value":"v"}], "color_range":"", "growth_rate":config.growth_rate,"injury_x":0.5,"injury_y":0.5,"injury_radius":0.12,
                      "injury_fraction":0.5,"sim_time":"0.00","length":"—","amplitude":"—","memory":"—",
                      "step_info":"Waiting for initial assembly", "chart_svg":chart_svg([]),"run_directory":"", "backend":config.backend})
        if aim1:
            state.update({"field": "occupancy", "fields": [
                {"title": "Total cells R + D", "value": "occupancy"},
                {"title": "Renewing cells R", "value": "renewing"},
                {"title": "Differentiated cells D", "value": "differentiated"},
                {"title": "Niche signal N", "value": "niche"},
                {"title": "Differentiated fraction", "value": "fraction"}],
                "initial_condition": organization["initial_condition"],
                "initial_conditions": ["mosaic", "random", "low", "high", "segregated", "imposed_pattern"],
                "population": "both", "populations": ["both", "renewing", "differentiated"],
                "injury_radius": 0.35, "injury_fraction": 0.6, "turnover": "—",
                "color_mode": "fixed", "color_modes": ["fixed", "auto"], "turnover_info": ""})
        if spatial:
            supply = organization["parameters"]["substrate_supply"]
            state.update({"field": "renewing_fraction", "erase_signals": True,
                "morphology": {1.24: "niches", 1.65: "bands"}.get(supply, "custom"),
                "morphologies": ["niches", "bands", "custom"],
                "initial_conditions": ["near_uniform", "random", "low", "high", "mosaic", "segregated", "imposed_pattern"],
                "fields": [{"title": "Renewing-cell fraction", "value": "renewing_fraction"},
                           *state.fields, {"title": "Niche precursor", "value": "substrate"}]})
        with SinglePageLayout(self.server) as layout:
            layout.title.set_text("Tissue Growth · Aim 1" if aim1 else "Tissue Growth · Pattern Laboratory")
            with layout.toolbar:
                v3.VSpacer()
                v3.VChip("{{ backend }} · CPU", size="small", variant="tonal", color="teal", classes="mr-3")
                v3.VChip("Research prototype", size="small", variant="outlined")
            with layout.content:
                client.Style(CSS)
                with html.Div(classes="shell"):
                    with html.Div(classes="sidebar"):
                        html.Div("AIM 1" if aim1 else "EXPERIMENT 01", classes="eyebrow")
                        html.Div("Establishment & repair" if aim1 else "Growth & spatial organization", style="font-size:22px;line-height:1.25;font-weight:600")
                        html.P("Local niche activation and precursor depletion establish renewing-rich islands or bands in differentiated-rich tissue. Patterns arise from local rules on a fixed surface." if spatial else "Renewing cells, differentiated cells, and niche feedback on a fixed surface. Explore uniform homeostasis and recovery after cell loss." if aim1 else "Two interacting signals on an expanding tissue. Explore growth history and recovery after a local signal perturbation.",classes="note")
                        with html.Div(classes="row"):
                            html.Button("Run", classes="btn primary", click=lambda: self.command("run"), disabled=("busy || running || complete",))
                            html.Button("Pause",classes="btn",click=lambda:self.command("pause"),disabled=("busy || !running",))
                            html.Button("Reset",classes="btn",click=self.reset,disabled=("busy",))
                        if aim1:
                            if spatial:
                                v3.VSelect(v_model=("morphology",), items=("morphologies",), label="Spatial preset", density="compact", hide_details=True, variant="outlined")
                                html.P("Choose niches or connected bands, then Reset. The presets differ in precursor supply; custom uses the loaded configuration. Neither stores a target map.", classes="note")
                            html.Div("Initial condition", classes="panel-title")
                            v3.VSelect(v_model=("initial_condition",), items=("initial_conditions",), label="Starting tissue", density="compact", hide_details=True, variant="outlined")
                            html.P("Choose a start, then press Reset and Run. The same feedback parameters apply to every start. Allow the tissue to settle before applying a repair experiment.", classes="note")
                        else:
                            html.Div("Growth intervention",classes="panel-title")
                            v3.VTextField(v_model=("growth_rate",), label="Linear growth rate (1 / time)", type="number", min=0, step=0.005, density="compact", hide_details=True, variant="outlined")
                            html.Button("Apply growth rate",classes="btn",style="margin-top:10px",click=self.apply_growth,disabled=("busy || complete",))
                            html.P(f"Isotropic expansion on a {config.geometry}, capped at {size_name} {config.max_length:g}. Changes apply at the next step boundary.",classes="note")
                        html.Div("Local cell depletion" if aim1 else "Local signal depletion",classes="panel-title")
                        if aim1:
                            v3.VSelect(v_model=("population",), items=("populations",), label="Cells to deplete", density="compact", hide_details=True, variant="outlined")
                            if spatial:
                                v3.VCheckbox(v_model=("erase_signals",), label="Also deplete niche signals", hide_details=True, density="compact")
                        with html.Div(classes="row", style="flex-direction:column;align-items:stretch" if spherical else ""):
                            v3.VTextField(v_model=("injury_x",),label="Longitude / 360°" if spherical else "Center x",type="number",min=0,max=1,step=0.1,density="compact",hide_details=True,variant="outlined")
                            v3.VTextField(v_model=("injury_y",),label="Colatitude / 180°" if spherical else "Center y",type="number",min=0,max=1,step=0.1,density="compact",hide_details=True,variant="outlined")
                        v3.VSlider(v_model=("injury_radius",),label="Width (radians)" if spherical else "Radius",min=0.02,max=0.8 if aim1 else 0.4,step=0.01,thumb_label=True,hide_details=True,color="teal")
                        v3.VSlider(v_model=("injury_fraction",),label="Fraction",min=0,max=1,step=0.05,thumb_label=True,hide_details=True,color="teal")
                        html.Button("Apply perturbation",classes="btn",click=self.perturb,disabled=("busy || complete",))
                        html.P(("Longitude wraps from 0 to 1; colatitude runs from north pole (0) to south pole (1). Width is angular distance along the surface. " if spherical else "Coordinates and radius are fractions of the current side length. ")+("Signal depletion removes niche activity and precursor within the same lesion. Allow time for the spatial organization to rebuild." if spatial else "Removes the selected cells locally; niche signal is retained." if aim1 else "Depletes activator u; this is not a cell-ablation model."),classes="note")
                        html.Div("Save & reproduce",classes="panel-title")
                        html.Button("Save snapshot" if aim1 else "Save checkpoint",classes="btn",click=lambda:self.command("save"),disabled=("busy",))
                        html.Div("{{ run_directory }}",classes="status",style="margin-top:12px")
                        html.P("Checkpoints, diagnostics, and intervention logs are saved on the machine running the solver.",classes="note")
                        html.Div("{{ error }}",v_if=("error",),classes="error",role="alert")
                    with html.Div(classes="workspace"):
                        with html.Div(classes="metrics"):
                            for label, variable in ([("Simulation time","sim_time"),("Mean cell density","length"),("Lineage contrast · SD" if spatial else "Spatial variation · CV","amplitude"),("Replacement flux","turnover")] if aim1 else [("Simulation time","sim_time"),(size_name.capitalize(),"length"),("Pattern amplitude σ(u)","amplitude"),("Worker RAM · MB","memory")]):
                                with html.Div(classes="metric"):
                                    html.Div(label,classes="metric-label")
                                    html.Div("{{ "+variable+" }}",classes="metric-value")
                        with html.Div(classes="view-card"):
                            with html.Div(classes="view-toolbar"):
                                html.Span("Field range · {{ color_range }}" if aim1 else "Concentration · {{ color_range }} (auto scale)")
                                if aim1:
                                    v3.VSelect(v_model=("color_mode",), items=("color_modes",), label="Color scale", density="compact", hide_details=True, variant="solo", style="max-width:130px")
                                v3.VSelect(v_model=("field",),items=("fields",),label="Displayed field",density="compact",hide_details=True,variant="solo",style="max-width:240px")
                            self.view = vtk.VtkLocalView(self.plotter.ren_win, ref="tissue_view", style="flex:1;min-height:300px;width:100%")
                        with html.Div(classes="chart-card"):
                            html.Div("Spatial contrast in renewing-cell fraction" if spatial else "Spatial cell-density variation" if aim1 else "Spatial pattern amplitude",style="font-size:13px;font-weight:600")
                            html.Div(v_html=("chart_svg",))
                        if aim1:
                            html.Div("{{ turnover_info }}", classes="status", id="turnover-status")
                        html.Div("{{ status }}",classes="status",id="run-status")
                        html.Div("{{ step_info }}",classes="status")
        state.change("field")(self.change_field)
        if aim1:
            state.change("color_mode")(self.change_field)
        self.server.controller.on_server_ready.add(self.on_ready)
        self.server.controller.on_server_exited.add(self.close)

    def on_ready(self, **_):
        self.start_worker(self.restart_path)
        self.task = asyncio.create_task(self.poll())

    def start_worker(self, restart=None):
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:6]
        directory = self.output_root/name
        options = None if self.organization is None else {**self.organization, "initial_condition": self.state.initial_condition}
        if self.spatial:
            parameters = dict(options["parameters"])
            if self.state.morphology in {"niches", "bands"}:
                parameters["substrate_supply"] = {"niches": 1.24, "bands": 1.65}[self.state.morphology]
            options["parameters"] = parameters
        self.worker = Worker(self.config,directory,restart,organization=options)
        self.state.run_directory = str(directory.resolve())

    def command(self, action, **kwargs):
        try:
            self.worker.send(action, **kwargs)
            self.state.status = f"Queued: {action}"
            self.state.error = ""
        except (RuntimeError, ValueError, AttributeError) as exc:
            self.state.error = str(exc)

    def apply_growth(self):
        try:
            self.command("growth",rate=float(self.state.growth_rate))
        except (ValueError, TypeError):
            self.state.error = "Enter a numeric growth rate"

    def perturb(self):
        try:
            parameters = {"x":float(self.state.injury_x),"y":float(self.state.injury_y),"radius":float(self.state.injury_radius),"fraction":float(self.state.injury_fraction)}
            if self.organization is not None:
                parameters["population"] = self.state.population
                if self.spatial:
                    parameters["erase_signals"] = bool(self.state.erase_signals)
            self.command("perturb",parameters=parameters)
        except (ValueError, TypeError):
            self.state.error = "Enter numeric perturbation coordinates"

    async def reset(self):
        self.state.update({"busy":True,"running":False,"complete":False,"error":"","status":"Resetting; saving the previous run…"})
        self.state.flush()
        previous_worker, self.worker = self.worker, None
        await asyncio.to_thread(previous_worker.close)
        self.history.clear()
        self.latest = None
        self.state.chart_svg = chart_svg([])
        self.start_worker()

    def change_field(self, field=None, **_):
        if self.grid is not None:
            self.render_frame()

    def render_frame(self):
        if self.grid is None or self.latest is None:
            return
        field = self.state.field
        index = 0 if field == "u" else 1
        if self.organization is not None:
            r, d, n = self.latest["c"][:3]
            fields = {"renewing": r, "differentiated": d, "niche": n,
                      "occupancy": r+d, "fraction": d/np.maximum(r+d, 1e-15),
                      "renewing_fraction": r/np.maximum(r+d, 1e-15)}
            if self.spatial:
                fields["substrate"] = self.latest["c"][3]
            values = fields[field]
        else:
            values = self.latest["c"][index]
        self.grid.point_data["concentration"] = values
        self.grid.set_active_scalars("concentration")
        low, high = float(values.min()),float(values.max())
        # Keep a finite color range for spatially homogeneous states.
        if high-low < 1e-6:
            low,high = low-5e-4,high+5e-4
        if self.organization is not None and self.state.color_mode == "fixed":
            low, high = 0.0, (1.0 if field in {"niche", "fraction"} else 1.5*self.organization["parameters"]["carrying_capacity"])
            if self.spatial:
                high = {"niche": 4.0, "substrate": 2.0, "fraction": 1.0, "renewing_fraction": 1.0}.get(field, high)
        self.actor.mapper.scalar_range = (low,high)
        self.state.color_range = f"{low:.4g} – {high:.4g}"
        self.view.update()

    async def poll(self):
        import pyvista as pv
        while True:
            try:
                if self.worker is not None:
                    while True:
                        try:
                            message = self.worker.notices.get_nowait()
                        except Empty:
                            break
                        kind = message["type"]
                        if kind == "ready":
                            self.points = message["points"]
                            faces = np.column_stack([np.full(len(message["triangles"]),3),message["triangles"]]).ravel()
                            self.grid = pv.PolyData(self.points.copy()*self.config.length,faces)
                            self.grid.point_data["concentration"] = np.ones(len(self.points))
                            self.plotter.clear()
                            self.actor = self.plotter.add_mesh(self.grid,scalars="concentration",cmap="viridis",show_edges=False,show_scalar_bar=False,smooth_shading=self.organization is not None)
                            self.plotter.view_xy()
                            # Fixed camera extent makes growth visible; it does not auto-fit each frame.
                            maximum = message["config"]["max_length"]
                            if message["config"]["geometry"] == "sphere":
                                # View the default depletion center (-x) and leave
                                # room for the entire sphere at its maximum radius.
                                self.plotter.camera.focal_point = (0, 0, 0)
                                self.plotter.camera.position = (-3*maximum, -maximum, maximum)
                                self.plotter.camera.up = (0, 0, 1)
                                self.plotter.camera.parallel_scale = maximum*(1.65 if self.organization is not None else 1.16)
                            else:
                                self.plotter.camera.focal_point = (maximum/2,maximum/2,0)
                                self.plotter.camera.position = (maximum/2,maximum/2,maximum*3)
                                self.plotter.camera.up = (0,1,0)
                                self.plotter.camera.parallel_scale = maximum*0.58
                            self.plotter.reset_camera_clipping_range()
                            self.state.update({"busy":False,"status":"Ready · press Run to begin","backend":message["config"]["backend"]})
                            self.view.update()
                            self.view.push_camera()
                        elif kind == "error":
                            self.state.update({"busy":False,"running":False,"error":message["message"],"status":"Simulation failed · inspect the error and reset"})
                        elif kind == "command_error":
                            self.state.error = message["message"]
                        elif kind == "ack":
                            self.state.status = f"{message['action'].capitalize()} applied at t = {message['time']:.3f}"
                        elif kind == "complete":
                            self.state.update({"complete":True,"running":False,"status":"Run complete · checkpoint saved"})
                    frame = None
                    while True:
                        try:
                            frame = self.worker.frames.get_nowait()
                        except Empty:
                            break
                    if frame is not None:
                        if self.latest is None:
                            self.state.growth_rate = frame["growth_rate"]
                        self.latest = frame
                        d = frame["diagnostics"]
                        metric = "std_renewing_fraction" if self.spatial else "cv_occupancy" if self.organization is not None else "std_u"
                        if not self.history or d["time"] != self.history[-1]["time"] or d[metric] != self.history[-1][metric]:
                            self.history.append(d)
                        if self.organization is not None:
                            self.state.update({"sim_time": f"{d['time']:.2f}", "length": f"{d['mean_occupancy']:.4f}",
                                "amplitude": f"{d[metric]:.4g}", "turnover": f"{d['flux_proliferation']:.4f}",
                                "running": frame["running"], "complete": frame["complete"],
                                "chart_svg": chart_svg(self.history, metric, "Renewing-cell fraction standard deviation" if self.spatial else "Cell-density coefficient of variation"),
                                "turnover_info": f"Differentiated fraction {d['differentiated_fraction']:.3f} · cumulative proliferation {d['cumulative_proliferation']:.3f} · cumulative cell loss {d['cumulative_renewing_loss']+d['cumulative_differentiated_loss']+d['cumulative_extrusion']:.3f}",
                                "step_info": f"Fixed {self.config.geometry} · {d['dofs']:,} unknowns · {d['step_ms']:.1f} ms / step · {d['rejected']} rejected steps"})
                        else:
                            self.state.update({"sim_time":f"{d['time']:.2f}","length":f"{d['length']:.3f}","amplitude":f"{d['std_u']:.4f}","memory":f"{d['worker_rss_mb']:.0f}" if d["worker_rss_mb"] is not None else "N/A",
                                           "running":frame["running"],"complete":frame["complete"],"chart_svg":chart_svg(self.history),
                                           "step_info":f"{d['dofs']:,} DOFs · {d['step_ms']:.1f} ms / last step · dt {d['last_dt']:.4g} · balance residual {d['balance_error']:.1e} · {d['rejected']} rejected steps"})
                        if self.grid is not None:
                            self.grid.points = self.points*d["length"]
                            self.plotter.reset_camera_clipping_range()
                            self.render_frame()
                    if not self.worker.process.is_alive() and not self.state.error:
                        self.state.update({"busy":False,"running":False,"error":f"Worker exited (code {self.worker.process.exitcode}). Reset to restart."})
                self.state.flush()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state.error = f"Dashboard update failed: {exc}"
                self.state.flush()
            await asyncio.sleep(0.1)

    def close(self, **_):
        if self.task is not None:
            self.task.cancel()
        if self.worker is not None:
            self.worker.close()
        self.plotter.close()

    def start(self, host="127.0.0.1",port=8080):
        try:
            self.server.start(host=host,port=port,open_browser=False)
        finally:
            self.close()
