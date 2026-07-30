"""Tkinter GUI for the MMI photonic link power budget calculator.

All calculation logic lives in power_budget.py — this module only handles
input widgets, layout, and displaying results.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import power_budget as pb

# (display label, params-dict key) for parameters that feed the optical
# cascade — these are the only ones worth sweeping against Output power /
# Transmission, since the rest (responsivity, drive params, etc.) don't
# affect those two curves at all.
SWEEP_PARAMS = [
    ("Laser power (dBm)", "p_laser"),
    ("Edge coupler loss in (dB)", "l_in"),
    ("Edge coupler loss out (dB)", "l_out"),
    ("Core width (um)", "core_width"),
    ("Core height (um)", "core_height"),
    ("Core bulk loss (dB/cm)", "alpha_core"),
    ("Cladding bulk loss (dB/cm)", "alpha_clad"),
    ("Waveguide length before MMI (um)", "length1"),
    ("Waveguide length after MMI (um)", "length2"),
    ("MMI # output ports", "n_ports"),
    ("MMI width (um)", "mmi_width"),
    ("MMI length (um)", "mmi_custom_length"),
    ("Modulator length (um)", "mod_length"),
    ("Modulator splitting ratio γ", "splitting_ratio"),
]


class PowerBudgetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MMI Modulation Power Calculator")

        self.mod_efficiency_mode = tk.StringVar(value="Capacitive")
        self.include_out_coupler = tk.BooleanVar(value=True)
        self.mmi_access_mode = tk.StringVar(value="Direct")
        self.mmi_length_mode = tk.StringVar(value="Derived")
        self._row_widgets = {}

        root.minsize(700, 500)

        container = ttk.Frame(root, padding=10)
        container.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(0, weight=1)

        self.inputs_frame = ttk.Frame(container)
        self.inputs_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.inputs_frame.columnconfigure(1, weight=1)

        results_frame = ttk.Frame(container)
        results_frame.grid(row=0, column=1, sticky="nsew")
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        self._build_results(results_frame)

        self.mod_efficiency_mode.trace_add("write", self._clear_results)
        self.include_out_coupler.trace_add("write", self._clear_results)
        self.mmi_access_mode.trace_add("write", self._clear_results)
        self.mmi_length_mode.trace_add("write", self._clear_results)

        self._build_inputs()

        button_row = ttk.Frame(container)
        button_row.grid(row=1, column=0, columnspan=2, pady=10)
        ttk.Button(button_row, text="Calculate", command=self.calculate).grid(
            row=0, column=0, padx=(0, 6)
        )
        ttk.Button(button_row, text="Snapshot results", command=self._snapshot_results).grid(
            row=0, column=1, padx=(6, 6)
        )
        ttk.Button(button_row, text="Parameter sweep", command=self._open_sweep_window).grid(
            row=0, column=2, padx=(6, 6)
        )
        ttk.Button(button_row, text="3D parameter sweep", command=self._open_3d_sweep_window).grid(
            row=0, column=3
        )

    # ------------------------------------------------------------------
    # Input construction
    # ------------------------------------------------------------------

    def _track(self, row, widget):
        self._row_widgets.setdefault(row, []).append(widget)

    def _add_section(self, row, text):
        label = ttk.Label(self.inputs_frame, text=text, font=("", 10, "bold"))
        label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 2))
        self._track(row, label)
        return row + 1

    def _add_field(self, row, label, default=""):
        lbl = ttk.Label(self.inputs_frame, text=label)
        lbl.grid(row=row, column=0, sticky="w")
        var = tk.StringVar(value=default)
        var.trace_add("write", self._clear_results)
        entry = ttk.Entry(self.inputs_frame, textvariable=var, width=14)
        entry.grid(row=row, column=1, sticky="ew")
        self._track(row, lbl)
        self._track(row, entry)
        return var

    def _add_output_row(self, parent, row, label_text):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w")
        var = tk.StringVar(value="—")
        ttk.Label(parent, textvariable=var).grid(row=row, column=1, sticky="e", padx=(12, 0))
        return var

    def _build_results(self, parent):
        ttk.Label(parent, text="Results", font=("", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

        self.budget_tree = ttk.Treeview(
            parent, columns=("stage", "power"), show="headings", height=8
        )
        self.budget_tree.heading("stage", text="Stage")
        self.budget_tree.heading("power", text="Power (dBm)")
        self.budget_tree.column("stage", width=230, anchor="w", stretch=True)
        self.budget_tree.column("power", width=120, anchor="e", stretch=True)
        self.budget_tree.grid(row=2, column=0, columnspan=2, sticky="nsew")
        parent.rowconfigure(2, weight=1)

        row = 3
        ttk.Label(parent, text="Summary", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 2)
        )
        row += 1
        self.v_out_power = self._add_output_row(parent, row, "Output power (dBm):")
        row += 1
        self.v_total_loss = self._add_output_row(parent, row, "Total loss (dB):")
        row += 1
        self.v_available_power = self._add_output_row(parent, row, "Available power (dB):")
        row += 1
        self.v_power_budget = self._add_output_row(parent, row, "Power budget (dB):")
        row += 1
        self.v_photocurrent = self._add_output_row(parent, row, "Photocurrent (uA):")
        row += 1
        self.v_mod_eff = self._add_output_row(parent, row, "Modulation efficiency (pJ/bit):")
        row += 1
        self.v_confinement = self._add_output_row(parent, row, "Confinement factor Γ:")
        row += 1
        self.v_n_eff = self._add_output_row(parent, row, "Effective index:")
        row += 1
        self.v_alpha_out = self._add_output_row(parent, row, "Derived α (dB/cm):")
        row += 1
        self.v_er_out = self._add_output_row(parent, row, "Derived ER (dB):")
        row += 1
        self.v_mmi_length_out = self._add_output_row(parent, row, "MMI length used (um):")
        row += 1
        self.v_mmi_excess_out = self._add_output_row(parent, row, "MMI excess loss (dB):")
        row += 1
        self.v_mmi_modes_out = self._add_output_row(parent, row, "MMI guided modes:")
        row += 1

        self.error_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.error_var, foreground="red",
                  wraplength=340, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )

    def _build_inputs(self):
        row = 0
        f = self.inputs_frame

        row = self._add_section(row, "Laser")
        self.v_p_laser = self._add_field(row, "Laser power (dBm):", "10")
        row += 1

        row = self._add_section(row, "Edge coupler")
        self.v_l_in = self._add_field(row, "Loss in (dB):", "1.7")
        row += 1
        self.out_coupler_toggle = ttk.Checkbutton(
            f, text="Include output edge coupler",
            variable=self.include_out_coupler, command=self._refresh_out_coupler_fields
        )
        self.out_coupler_toggle.grid(row=row, column=0, columnspan=2, sticky="w")
        self._track(row, self.out_coupler_toggle)
        row += 1
        self.out_coupler_row = row
        self.v_l_out = self._add_field(row, "Loss out (dB):", "1.7")
        row += 1

        row = self._add_section(row, "Waveguide")
        self.v_wavelength = self._add_field(row, "Wavelength (nm):", "1550")
        row += 1
        self.v_n_core = self._add_field(row, "Core index:", "3.45")
        row += 1
        self.v_n_clad = self._add_field(row, "Cladding index:", "1.44")
        row += 1
        self.v_core_width = self._add_field(row, "Core width (um):", "0.48")
        row += 1
        self.v_core_height = self._add_field(row, "Core height (um):", "0.22")
        row += 1
        self.v_alpha_core = self._add_field(row, "Core bulk loss (dB/cm):", "0.01")
        row += 1
        self.v_alpha_clad = self._add_field(row, "Cladding bulk loss (dB/cm):", "0.001")
        row += 1
        self.v_length1 = self._add_field(row, "Length before MMI (um):", "5")
        row += 1
        self.v_length2 = self._add_field(row, "Length after MMI (um):", "5")
        row += 1

        row = self._add_section(row, "MMI")
        self.v_n_ports = self._add_field(row, "# output ports:", "2")
        row += 1
        self.v_mmi_width = self._add_field(row, "Width (um):", "3.0")
        row += 1

        access_label = ttk.Label(f, text="Access coupling:")
        access_label.grid(row=row, column=0, sticky="w")
        self._track(row, access_label)
        access_menu = ttk.OptionMenu(
            f, self.mmi_access_mode, self.mmi_access_mode.get(),
            "Direct", "Tapered", command=lambda _=None: self._refresh_mmi_fields()
        )
        access_menu.grid(row=row, column=1, sticky="e")
        self._track(row, access_menu)
        row += 1
        self.mmi_taper_row = row
        self.v_taper_width = self._add_field(row, "Taper tip width (um):", "1.0")
        row += 1

        length_mode_label = ttk.Label(f, text="Length:")
        length_mode_label.grid(row=row, column=0, sticky="w")
        self._track(row, length_mode_label)
        length_menu = ttk.OptionMenu(
            f, self.mmi_length_mode, self.mmi_length_mode.get(),
            "Derived", "Custom", command=lambda _=None: self._refresh_mmi_fields()
        )
        length_menu.grid(row=row, column=1, sticky="e")
        self._track(row, length_menu)
        row += 1
        self.mmi_custom_length_row = row
        self.v_mmi_custom_length = self._add_field(row, "Custom length (um):", "10.0")
        row += 1

        row = self._add_section(row, "MZM")
        self.v_mod_length = self._add_field(row, "Length (um):", "500")
        row += 1
        self.v_splitting_ratio = self._add_field(row, "Splitting ratio γ (0.5 = ideal):", "0.48")
        row += 1

        row = self._add_section(row, "Photodetector")
        self.v_responsivity = self._add_field(row, "Responsivity (A/W):", "0.9")
        row += 1
        self.v_sensitivity = self._add_field(row, "Sensitivity (dBm):", "-20")
        row += 1

        row = self._add_section(row, "Modulation efficiency")
        mode_label = ttk.Label(f, text="Mode:")
        mode_label.grid(row=row, column=0, sticky="w")
        self._track(row, mode_label)
        mode_menu = ttk.OptionMenu(
            f, self.mod_efficiency_mode, self.mod_efficiency_mode.get(),
            "Capacitive", "Power",
            command=lambda _=None: self._refresh_mod_eff_fields()
        )
        mode_menu.grid(row=row, column=1, sticky="e")
        self._track(row, mode_menu)
        row += 1
        self.mod_cap_row_start = row
        self.v_cap_per_um = self._add_field(row, "Capacitance density (fF/um):", "0.1")
        row += 1
        self.v_vpp = self._add_field(row, "Drive swing (V):", "2.0")
        row += 1
        self.mod_cap_row_end = row
        self.v_p_drive = self._add_field(row, "Drive power (W):", "0.01")
        row += 1
        self.v_bit_rate = self._add_field(row, "Bit rate (bits/s):", "25e9")
        row += 1
        self.mod_power_row_end = row

        self._refresh_mod_eff_fields()
        self._refresh_out_coupler_fields()
        self._refresh_mmi_fields()

    def _set_rows_visible(self, start, end, visible):
        for r in range(start, end):
            for widget in self._row_widgets.get(r, []):
                if visible:
                    widget.grid()
                else:
                    widget.grid_remove()

    def _refresh_mod_eff_fields(self):
        capacitive = self.mod_efficiency_mode.get() == "Capacitive"
        self._set_rows_visible(self.mod_cap_row_start, self.mod_cap_row_end, capacitive)
        self._set_rows_visible(self.mod_cap_row_end, self.mod_power_row_end, not capacitive)

    def _refresh_out_coupler_fields(self):
        self._set_rows_visible(self.out_coupler_row, self.out_coupler_row + 1,
                                self.include_out_coupler.get())

    def _refresh_mmi_fields(self):
        self._set_rows_visible(self.mmi_taper_row, self.mmi_taper_row + 1,
                                self.mmi_access_mode.get() == "Tapered")
        self._set_rows_visible(self.mmi_custom_length_row, self.mmi_custom_length_row + 1,
                                self.mmi_length_mode.get() == "Custom")

    # ------------------------------------------------------------------
    # Calculation
    # ------------------------------------------------------------------

    def _snapshot_results(self):
        popup = tk.Toplevel(self.root)
        popup.title("Results snapshot")
        popup.minsize(360, 300)

        frame = ttk.Frame(popup, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Results", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        tree = ttk.Treeview(frame, columns=("stage", "power"), show="headings", height=8)
        tree.heading("stage", text="Stage")
        tree.heading("power", text="Power (dBm)")
        tree.column("stage", width=230, anchor="w", stretch=True)
        tree.column("power", width=120, anchor="e", stretch=True)
        for item in self.budget_tree.get_children():
            tree.insert("", tk.END, values=self.budget_tree.item(item)["values"])
        tree.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(2, 10))

        summary = [
            ("Output power (dBm):", self.v_out_power.get()),
            ("Total loss (dB):", self.v_total_loss.get()),
            ("Available power (dB):", self.v_available_power.get()),
            ("Power budget (dB):", self.v_power_budget.get()),
            ("Photocurrent (uA):", self.v_photocurrent.get()),
            ("Modulation efficiency (pJ/bit):", self.v_mod_eff.get()),
        ]
        row = 2
        for label, value in summary:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            ttk.Label(frame, text=value).grid(row=row, column=1, sticky="e", padx=(12, 0))
            row += 1

    def _open_sweep_window(self):
        popup = tk.Toplevel(self.root)
        popup.title("Parameter sweep")
        popup.minsize(640, 520)

        frame = ttk.Frame(popup, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        param_labels = [label for label, _key in SWEEP_PARAMS]
        key_by_label = dict(SWEEP_PARAMS)

        controls = ttk.Frame(frame)
        controls.grid(row=0, column=0, sticky="w", pady=(0, 8))

        param_var = tk.StringVar(value=param_labels[0])
        ttk.Label(controls, text="Parameter:").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(controls, param_var, param_var.get(), *param_labels).grid(
            row=0, column=1, padx=(4, 16)
        )

        ttk.Label(controls, text="Start:").grid(row=0, column=2, sticky="w")
        start_var = tk.StringVar()
        ttk.Entry(controls, textvariable=start_var, width=10).grid(row=0, column=3, padx=(4, 16))

        ttk.Label(controls, text="Stop:").grid(row=0, column=4, sticky="w")
        stop_var = tk.StringVar()
        ttk.Entry(controls, textvariable=stop_var, width=10).grid(row=0, column=5, padx=(4, 16))

        ttk.Label(controls, text="Points:").grid(row=0, column=6, sticky="w")
        points_var = tk.StringVar(value="21")
        ttk.Entry(controls, textvariable=points_var, width=6).grid(row=0, column=7, padx=(4, 16))

        ttk.Label(controls, text="Y-axis:").grid(row=0, column=8, sticky="w")
        units_var = tk.StringVar(value="dB")
        ttk.OptionMenu(controls, units_var, units_var.get(), "dB", "Linear",
                       command=lambda _=None: redraw()).grid(row=0, column=9, padx=(4, 16))

        show_out_var = tk.BooleanVar(value=True)
        show_trans_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Output power", variable=show_out_var,
                         command=lambda: redraw()).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )
        ttk.Checkbutton(controls, text="Transmission", variable=show_trans_var,
                         command=lambda: redraw()).grid(
            row=1, column=3, columnspan=3, sticky="w", pady=(6, 0)
        )

        error_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=error_var, foreground="red").grid(
            row=1, column=0, sticky="w"
        )

        fig = Figure(figsize=(6, 4.5), dpi=100)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=3, column=0, sticky="w")

        point_label_var = tk.StringVar(value="Click a point on the graph to inspect its value.")
        ttk.Label(frame, textvariable=point_label_var).grid(
            row=4, column=0, sticky="w", pady=(4, 0)
        )
        selected_annotation = {"out": None, "trans": None}

        def prefill_range(*_args):
            key = key_by_label[param_var.get()]
            if key == "mmi_custom_length":
                self.mmi_length_mode.set("Custom")
            try:
                current = self._current_params()[key]
            except (tk.TclError, ValueError):
                return
            span = abs(current) * 0.5 if current != 0 else 1.0
            start_var.set(f"{current - span:.4g}")
            stop_var.set(f"{current + span:.4g}")

        param_var.trace_add("write", prefill_range)
        prefill_range()

        last_data = {"xs": None, "p_pd_dbm": None, "total_loss": None}

        def redraw():
            if last_data["xs"] is None:
                return
            xs = last_data["xs"]
            if units_var.get() == "dB":
                out_vals = last_data["p_pd_dbm"]
                trans_vals = [-tl for tl in last_data["total_loss"]]
                out_label, trans_label = "Output power (dBm)", "Transmission (dB)"
            else:
                out_vals = [pb.dbm_to_mw(v) for v in last_data["p_pd_dbm"]]
                trans_vals = [pb.db_to_linear(tl) for tl in last_data["total_loss"]]
                out_label, trans_label = "Output power (mW)", "Transmission (linear)"

            ax1.clear()
            ax2.clear()
            selected_annotation["out"] = None
            selected_annotation["trans"] = None
            point_label_var.set("Click a point on the graph to inspect its value.")

            if show_out_var.get():
                ax1.plot(xs, out_vals, color="tab:blue", marker="o", markersize=3, label=out_label)
                ax1.set_ylabel(out_label, color="tab:blue", labelpad=10)
                ax1.tick_params(axis="y", labelcolor="tab:blue")
                ax1.yaxis.set_visible(True)
            else:
                ax1.yaxis.set_visible(False)

            if show_trans_var.get():
                ax2.plot(xs, trans_vals, color="tab:red", marker="s", markersize=3, label=trans_label)
                ax2.yaxis.set_label_position("right")
                ax2.set_ylabel(trans_label, color="tab:red", labelpad=18)
                ax2.tick_params(axis="y", labelcolor="tab:red")
                ax2.yaxis.set_visible(True)
            else:
                ax2.yaxis.set_visible(False)

            ax1.set_xlabel(param_var.get())
            ax1.grid(True, alpha=0.3)
            # Fixed fractional margins instead of tight_layout(): reserved space
            # for both twin-axis labels stays valid at any window size, so
            # nothing gets clipped regardless of how the popup is resized.
            fig.subplots_adjust(left=0.16, right=0.82, bottom=0.12, top=0.95)
            canvas.draw()

        def on_click(event):
            if last_data["xs"] is None or event.xdata is None:
                return
            if event.inaxes not in (ax1, ax2):
                return

            xs = last_data["xs"]
            idx = min(range(len(xs)), key=lambda i: abs(xs[i] - event.xdata))

            for key in ("out", "trans"):
                if selected_annotation[key] is not None:
                    selected_annotation[key].remove()
                    selected_annotation[key] = None

            if units_var.get() == "dB":
                out_vals = last_data["p_pd_dbm"]
                trans_vals = [-tl for tl in last_data["total_loss"]]
            else:
                out_vals = [pb.dbm_to_mw(v) for v in last_data["p_pd_dbm"]]
                trans_vals = [pb.db_to_linear(tl) for tl in last_data["total_loss"]]

            summary_parts = [f"x = {xs[idx]:.4g}"]
            if show_out_var.get():
                y = out_vals[idx]
                selected_annotation["out"] = ax1.annotate(
                    f"({xs[idx]:.4g}, {y:.4g})", xy=(xs[idx], y),
                    xytext=(10, 10), textcoords="offset points", color="tab:blue",
                    bbox={"boxstyle": "round", "fc": "white", "ec": "tab:blue", "alpha": 0.9},
                    arrowprops={"arrowstyle": "->", "color": "tab:blue"},
                )
                summary_parts.append(f"output = {y:.4g}")
            if show_trans_var.get():
                y = trans_vals[idx]
                selected_annotation["trans"] = ax2.annotate(
                    f"({xs[idx]:.4g}, {y:.4g})", xy=(xs[idx], y),
                    xytext=(10, -20), textcoords="offset points", color="tab:red",
                    bbox={"boxstyle": "round", "fc": "white", "ec": "tab:red", "alpha": 0.9},
                    arrowprops={"arrowstyle": "->", "color": "tab:red"},
                )
                summary_parts.append(f"transmission = {y:.4g}")

            point_label_var.set("Selected point: " + ", ".join(summary_parts))
            canvas.draw_idle()

        canvas.mpl_connect("button_press_event", on_click)

        def run_sweep():
            error_var.set("")
            try:
                key = key_by_label[param_var.get()]
                start = float(start_var.get())
                stop = float(stop_var.get())
                n_points = int(points_var.get())
                if n_points < 2:
                    raise ValueError("Points must be >= 2")

                base_params = self._current_params()
                step = (stop - start) / (n_points - 1)
                xs = [start + i * step for i in range(n_points)]
                p_pd_dbm_vals = []
                total_loss_vals = []
                for x in xs:
                    params = dict(base_params)
                    params[key] = x
                    result = self._compute(params)
                    p_pd_dbm_vals.append(result["p_pd_dbm"])
                    total_loss_vals.append(result["total_loss"])
            except (tk.TclError, ValueError) as exc:
                error_var.set(f"Input error: {exc}")
                return
            except ZeroDivisionError:
                error_var.set("Input error: division by zero for this sweep range")
                return

            last_data["xs"] = xs
            last_data["p_pd_dbm"] = p_pd_dbm_vals
            last_data["total_loss"] = total_loss_vals
            redraw()

        ttk.Button(controls, text="Run sweep", command=run_sweep).grid(row=0, column=10, padx=(8, 0))

    def _open_3d_sweep_window(self):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers the 3d projection

        popup = tk.Toplevel(self.root)
        popup.title("3D parameter sweep")
        popup.minsize(700, 620)

        frame = ttk.Frame(popup, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        param_labels = [label for label, _key in SWEEP_PARAMS]
        key_by_label = dict(SWEEP_PARAMS)

        controls = ttk.Frame(frame)
        controls.grid(row=0, column=0, sticky="w", pady=(0, 8))

        ttk.Label(controls, text="Param 1:").grid(row=0, column=0, sticky="w")
        param1_var = tk.StringVar(value=param_labels[0])
        ttk.OptionMenu(controls, param1_var, param1_var.get(), *param_labels).grid(
            row=0, column=1, padx=(4, 12)
        )
        ttk.Label(controls, text="Start:").grid(row=0, column=2, sticky="w")
        start1_var = tk.StringVar()
        ttk.Entry(controls, textvariable=start1_var, width=8).grid(row=0, column=3, padx=(4, 8))
        ttk.Label(controls, text="Stop:").grid(row=0, column=4, sticky="w")
        stop1_var = tk.StringVar()
        ttk.Entry(controls, textvariable=stop1_var, width=8).grid(row=0, column=5, padx=(4, 8))
        ttk.Label(controls, text="Points:").grid(row=0, column=6, sticky="w")
        points1_var = tk.StringVar(value="12")
        ttk.Entry(controls, textvariable=points1_var, width=5).grid(row=0, column=7, padx=(4, 12))

        ttk.Label(controls, text="Param 2:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        param2_var = tk.StringVar(value=param_labels[1])
        ttk.OptionMenu(controls, param2_var, param2_var.get(), *param_labels).grid(
            row=1, column=1, padx=(4, 12), pady=(6, 0)
        )
        ttk.Label(controls, text="Start:").grid(row=1, column=2, sticky="w", pady=(6, 0))
        start2_var = tk.StringVar()
        ttk.Entry(controls, textvariable=start2_var, width=8).grid(row=1, column=3, padx=(4, 8), pady=(6, 0))
        ttk.Label(controls, text="Stop:").grid(row=1, column=4, sticky="w", pady=(6, 0))
        stop2_var = tk.StringVar()
        ttk.Entry(controls, textvariable=stop2_var, width=8).grid(row=1, column=5, padx=(4, 8), pady=(6, 0))
        ttk.Label(controls, text="Points:").grid(row=1, column=6, sticky="w", pady=(6, 0))
        points2_var = tk.StringVar(value="12")
        ttk.Entry(controls, textvariable=points2_var, width=5).grid(row=1, column=7, padx=(4, 12), pady=(6, 0))

        ttk.Label(controls, text="Z:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        z_var = tk.StringVar(value="Output power")
        ttk.OptionMenu(controls, z_var, z_var.get(), "Output power", "Transmission",
                       command=lambda _=None: redraw()).grid(row=2, column=1, padx=(4, 12), pady=(6, 0))
        ttk.Label(controls, text="Units:").grid(row=2, column=2, sticky="w", pady=(6, 0))
        units_var = tk.StringVar(value="dB")
        ttk.OptionMenu(controls, units_var, units_var.get(), "dB", "Linear",
                       command=lambda _=None: redraw()).grid(row=2, column=3, padx=(4, 12), pady=(6, 0))

        error_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=error_var, foreground="red").grid(
            row=1, column=0, sticky="w"
        )

        fig = Figure(figsize=(6.5, 5.5), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=3, column=0, sticky="w")

        def prefill(param_var, start_var, stop_var):
            key = key_by_label[param_var.get()]
            if key == "mmi_custom_length":
                self.mmi_length_mode.set("Custom")
            try:
                current = self._current_params()[key]
            except (tk.TclError, ValueError):
                return
            span = abs(current) * 0.5 if current != 0 else 1.0
            start_var.set(f"{current - span:.4g}")
            stop_var.set(f"{current + span:.4g}")

        param1_var.trace_add("write", lambda *_a: prefill(param1_var, start1_var, stop1_var))
        param2_var.trace_add("write", lambda *_a: prefill(param2_var, start2_var, stop2_var))
        prefill(param1_var, start1_var, stop1_var)
        prefill(param2_var, start2_var, stop2_var)

        last_data = {"X": None, "Y": None, "p_pd_dbm": None, "total_loss": None, "labels": None}

        def redraw():
            if last_data["X"] is None:
                return
            x_grid, y_grid = last_data["X"], last_data["Y"]

            if z_var.get() == "Output power":
                if units_var.get() == "dB":
                    z_grid = last_data["p_pd_dbm"]
                    z_label = "Output power (dBm)"
                else:
                    z_grid = [[pb.dbm_to_mw(v) for v in row] for row in last_data["p_pd_dbm"]]
                    z_label = "Output power (mW)"
            else:
                if units_var.get() == "dB":
                    z_grid = [[-tl for tl in row] for row in last_data["total_loss"]]
                    z_label = "Transmission (dB)"
                else:
                    z_grid = [[pb.db_to_linear(tl) for tl in row] for row in last_data["total_loss"]]
                    z_label = "Transmission (linear)"

            ax.clear()
            ax.plot_surface(np.array(x_grid), np.array(y_grid), np.array(z_grid),
                             cmap="viridis", edgecolor="none", antialiased=True)
            ax.set_xlabel(last_data["labels"][0])
            ax.set_ylabel(last_data["labels"][1])
            ax.set_zlabel(z_label)
            fig.subplots_adjust(left=0.02, right=0.95, bottom=0.05, top=0.95)
            canvas.draw()

        def run_sweep():
            error_var.set("")
            try:
                key1 = key_by_label[param1_var.get()]
                key2 = key_by_label[param2_var.get()]
                start1, stop1 = float(start1_var.get()), float(stop1_var.get())
                start2, stop2 = float(start2_var.get()), float(stop2_var.get())
                n1, n2 = int(points1_var.get()), int(points2_var.get())
                if n1 < 2 or n2 < 2:
                    raise ValueError("Points must be >= 2")

                base_params = self._current_params()
                step1 = (stop1 - start1) / (n1 - 1)
                step2 = (stop2 - start2) / (n2 - 1)
                xs = [start1 + i * step1 for i in range(n1)]
                ys = [start2 + j * step2 for j in range(n2)]

                x_grid = [[x for x in xs] for _ in ys]
                y_grid = [[y for _ in xs] for y in ys]
                p_pd_grid, loss_grid = [], []
                for y in ys:
                    p_row, loss_row = [], []
                    for x in xs:
                        params = dict(base_params)
                        params[key1] = x
                        params[key2] = y
                        result = self._compute(params)
                        p_row.append(result["p_pd_dbm"])
                        loss_row.append(result["total_loss"])
                    p_pd_grid.append(p_row)
                    loss_grid.append(loss_row)
            except (tk.TclError, ValueError) as exc:
                error_var.set(f"Input error: {exc}")
                return
            except ZeroDivisionError:
                error_var.set("Input error: division by zero for this sweep range")
                return

            last_data["X"] = x_grid
            last_data["Y"] = y_grid
            last_data["p_pd_dbm"] = p_pd_grid
            last_data["total_loss"] = loss_grid
            last_data["labels"] = (param1_var.get(), param2_var.get())
            redraw()

        ttk.Button(controls, text="Run 3D sweep", command=run_sweep).grid(
            row=2, column=6, columnspan=2, padx=(4, 0), pady=(6, 0)
        )

    def _clear_results(self, *args):
        for item in self.budget_tree.get_children():
            self.budget_tree.delete(item)
        for var in (self.v_out_power, self.v_total_loss, self.v_photocurrent,
                    self.v_mod_eff, self.v_available_power, self.v_power_budget,
                    self.v_confinement, self.v_n_eff, self.v_alpha_out, self.v_er_out,
                    self.v_mmi_length_out, self.v_mmi_excess_out, self.v_mmi_modes_out):
            var.set("—")
        self.error_var.set("")

    def calculate(self):
        self._clear_results()
        try:
            self._run_calculation()
        except (tk.TclError, ValueError) as exc:
            self.error_var.set(f"Input error: {exc}")
        except ZeroDivisionError:
            self.error_var.set(
                "Input error: division by zero (check splitting ratio, indices, "
                "dimensions, C, or bit rate values)"
            )

    def _f(self, var):
        return float(var.get())

    def _current_params(self):
        return {
            "p_laser": self._f(self.v_p_laser),
            "l_in": self._f(self.v_l_in),
            "l_out": self._f(self.v_l_out) if self.include_out_coupler.get() else 0.0,
            "wavelength_nm": self._f(self.v_wavelength),
            "n_core": self._f(self.v_n_core),
            "n_clad": self._f(self.v_n_clad),
            "core_width": self._f(self.v_core_width),
            "core_height": self._f(self.v_core_height),
            "alpha_core": self._f(self.v_alpha_core),
            "alpha_clad": self._f(self.v_alpha_clad),
            "length1": self._f(self.v_length1),
            "length2": self._f(self.v_length2),
            "n_ports": int(self._f(self.v_n_ports)),
            "mmi_width": self._f(self.v_mmi_width),
            "mmi_access_mode": self.mmi_access_mode.get(),
            "taper_width": self._f(self.v_taper_width),
            "mmi_length_mode": self.mmi_length_mode.get(),
            "mmi_custom_length": self._f(self.v_mmi_custom_length),
            "mod_length": self._f(self.v_mod_length),
            "splitting_ratio": self._f(self.v_splitting_ratio),
            "responsivity": self._f(self.v_responsivity),
            "sensitivity": self._f(self.v_sensitivity),
            "mod_mode": self.mod_efficiency_mode.get(),
            "cap_per_um": self._f(self.v_cap_per_um),
            "vpp": self._f(self.v_vpp),
            "p_drive": self._f(self.v_p_drive),
            "bit_rate": self._f(self.v_bit_rate),
        }

    def _compute(self, p):
        """Pure calculation from a params dict -> result dict. No widget access,
        so this can be reused for both the normal Calculate button and sweeps."""
        wavelength_um = p["wavelength_nm"] / 1000
        confinement, n_eff = pb.strip_confinement(
            p["n_core"], p["n_clad"], p["core_width"], p["core_height"], wavelength_um
        )
        alpha = pb.material_loss_db_per_cm(confinement, p["alpha_core"], p["alpha_clad"])

        n_ports = int(round(p["n_ports"]))
        _, n_eff_vertical = pb.slab_mode(p["n_core"], p["n_clad"], p["core_height"], wavelength_um)
        access_width = p["taper_width"] if p["mmi_access_mode"] == "Tapered" else p["core_width"]

        if p["mmi_length_mode"] == "Derived":
            mmi_length, mmi_excess, num_modes = pb.mmi_derive_length_and_loss(
                n_eff_vertical, p["n_clad"], p["mmi_width"], n_ports, access_width, wavelength_um
            )
        else:
            mmi_length = p["mmi_custom_length"]
            mmi_excess, num_modes = pb.mmi_excess_loss_db(
                n_eff_vertical, p["n_clad"], p["mmi_width"], n_ports, access_width,
                wavelength_um, mmi_length
            )

        l_prop1 = pb.propagation_loss_db(alpha, p["length1"])
        l_mmi = pb.mmi_total_loss_db(n_ports, mmi_excess)
        l_prop2 = pb.propagation_loss_db(alpha, p["length2"])
        l_mod_il = pb.propagation_loss_db(alpha, p["mod_length"])
        er_db = pb.er_db_from_splitting_ratio(p["splitting_ratio"])
        l_er_penalty = pb.er_power_penalty_db(er_db)

        stages = pb.compute_results(
            p["p_laser"], p["l_in"], l_prop1, l_mmi, l_prop2,
            l_mod_il, l_er_penalty, p["l_out"]
        )

        p_pd_dbm = stages[-2][1]  # power at photodetector, before the total-loss entry
        total_loss = stages[-1][1]
        photocurrent = pb.pd_metrics(p["responsivity"], p_pd_dbm)

        if p["mod_mode"] == "Capacitive":
            cap_fF = pb.capacitance_fF(p["cap_per_um"], p["mod_length"])
            e_bit_pJ = pb.modulation_efficiency_capacitive_pJ(cap_fF, p["vpp"])
        else:
            e_bit_pJ = pb.modulation_efficiency_power_pJ(p["p_drive"], p["bit_rate"])

        available = pb.available_power_db(p["p_laser"], p["sensitivity"])
        power_budget = pb.power_budget_db(p["p_laser"], p["sensitivity"], total_loss)

        return {
            "stages": stages,
            "p_pd_dbm": p_pd_dbm,
            "total_loss": total_loss,
            "photocurrent": photocurrent,
            "e_bit_pJ": e_bit_pJ,
            "available": available,
            "power_budget": power_budget,
            "confinement": confinement,
            "n_eff": n_eff,
            "alpha": alpha,
            "er_db": er_db,
            "mmi_length": mmi_length,
            "mmi_excess": mmi_excess,
            "mmi_num_modes": num_modes,
        }

    def _run_calculation(self):
        result = self._compute(self._current_params())

        for name, value in result["stages"][:-1]:
            self.budget_tree.insert("", tk.END, values=(name, f"{value:.3f}"))

        self.v_out_power.set(f"{result['p_pd_dbm']:.3f}")
        self.v_total_loss.set(f"{result['total_loss']:.3f}")
        self.v_photocurrent.set(f"{result['photocurrent'] * 1e6:.4f}")
        self.v_mod_eff.set(f"{result['e_bit_pJ']:.4f}")
        self.v_available_power.set(f"{result['available']:.3f}")
        self.v_power_budget.set(f"{result['power_budget']:.3f}")
        self.v_confinement.set(f"{result['confinement']:.4f}")
        self.v_n_eff.set(f"{result['n_eff']:.4f}")
        self.v_alpha_out.set(f"{result['alpha']:.4f}")
        self.v_er_out.set(f"{result['er_db']:.3f}")
        self.v_mmi_length_out.set(f"{result['mmi_length']:.4f}")
        self.v_mmi_excess_out.set(f"{result['mmi_excess']:.4f}")
        self.v_mmi_modes_out.set(str(result["mmi_num_modes"]))


def main():
    root = tk.Tk()
    PowerBudgetGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
