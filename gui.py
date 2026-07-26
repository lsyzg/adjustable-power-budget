"""Tkinter GUI for the MMI photonic link power budget calculator.

All calculation logic lives in power_budget.py — this module only handles
input widgets, layout, and displaying results.
"""

import tkinter as tk
from tkinter import ttk

import power_budget as pb


class PowerBudgetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MMI Power Budget Calculator")

        self.mod_efficiency_mode = tk.StringVar(value="Capacitive (CV^2)")
        self.include_out_coupler = tk.BooleanVar(value=True)
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

        self._build_inputs()

        calc_btn = ttk.Button(container, text="Calculate", command=self.calculate)
        calc_btn.grid(row=1, column=0, columnspan=2, pady=10)

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

        ttk.Label(parent, text="Power budget (stage → dBm)").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 2)
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
        self.v_qe_penalty = self._add_output_row(parent, row, "Quantum efficiency penalty (dB):")
        row += 1
        self.v_photocurrent = self._add_output_row(parent, row, "Photocurrent (uA):")
        row += 1
        self.v_mod_eff = self._add_output_row(parent, row, "Modulation efficiency (pJ/bit):")
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
        self.v_wavelength = self._add_field(row, "Wavelength (nm):", "1310")
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
        self.v_alpha = self._add_field(row, "alpha (dB/cm):", "2.0")
        row += 1
        self.v_length1 = self._add_field(row, "Length before MMI (um):", "5000")
        row += 1
        self.v_length2 = self._add_field(row, "Length after MMI (um):", "5000")
        row += 1

        row = self._add_section(row, "MMI (1xN)")
        self.v_n_ports = self._add_field(row, "Number of output ports N:", "2")
        row += 1
        self.v_mmi_excess = self._add_field(row, "Excess loss (dB):", "0.3")
        row += 1

        row = self._add_section(row, "Modulator (inline on post-MMI waveguide)")
        self.v_mod_length = self._add_field(row, "Length (um):", "500")
        row += 1
        self.v_mod_er = self._add_field(row, "Extinction ratio ER (dB):", "6.0")
        row += 1

        row = self._add_section(row, "Photodetector")
        self.v_responsivity = self._add_field(row, "Responsivity R (A/W):", "0.9")
        row += 1
        self.v_sensitivity = self._add_field(row, "Receiver sensitivity (dBm):", "-20")
        row += 1

        row = self._add_section(row, "Modulation efficiency")
        mode_label = ttk.Label(f, text="Mode:")
        mode_label.grid(row=row, column=0, sticky="w")
        self._track(row, mode_label)
        mode_menu = ttk.OptionMenu(
            f, self.mod_efficiency_mode, self.mod_efficiency_mode.get(),
            "Capacitive (CV^2)", "Power-based",
            command=lambda _=None: self._refresh_mod_eff_fields()
        )
        mode_menu.grid(row=row, column=1, sticky="e")
        self._track(row, mode_menu)
        row += 1
        self.mod_cap_row_start = row
        self.v_cap_fF = self._add_field(row, "Capacitance C (fF):", "50")
        row += 1
        self.v_vpp = self._add_field(row, "Drive swing Vpp (V):", "2.0")
        row += 1
        self.mod_cap_row_end = row
        self.v_p_drive = self._add_field(row, "Drive power (W):", "0.01")
        row += 1
        self.v_bit_rate = self._add_field(row, "Bit rate (bits/s):", "25e9")
        row += 1
        self.mod_power_row_end = row

        self._refresh_mod_eff_fields()
        self._refresh_out_coupler_fields()

    def _set_rows_visible(self, start, end, visible):
        for r in range(start, end):
            for widget in self._row_widgets.get(r, []):
                if visible:
                    widget.grid()
                else:
                    widget.grid_remove()

    def _refresh_mod_eff_fields(self):
        capacitive = self.mod_efficiency_mode.get() == "Capacitive (CV^2)"
        self._set_rows_visible(self.mod_cap_row_start, self.mod_cap_row_end, capacitive)
        self._set_rows_visible(self.mod_cap_row_end, self.mod_power_row_end, not capacitive)

    def _refresh_out_coupler_fields(self):
        self._set_rows_visible(self.out_coupler_row, self.out_coupler_row + 1,
                                self.include_out_coupler.get())

    # ------------------------------------------------------------------
    # Calculation
    # ------------------------------------------------------------------

    def _clear_results(self, *args):
        for item in self.budget_tree.get_children():
            self.budget_tree.delete(item)
        for var in (self.v_out_power, self.v_total_loss, self.v_qe_penalty,
                    self.v_photocurrent, self.v_mod_eff, self.v_available_power,
                    self.v_power_budget):
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
                "Input error: division by zero (check ER, C, alpha/length, or bit rate values)"
            )

    def _f(self, var):
        return float(var.get())

    def _run_calculation(self):
        p_laser = self._f(self.v_p_laser)

        l_in = self._f(self.v_l_in)

        alpha = self._f(self.v_alpha)
        length1 = self._f(self.v_length1)
        l_prop1 = pb.propagation_loss_db(alpha, length1)

        n_ports = self._f(self.v_n_ports)
        mmi_excess = self._f(self.v_mmi_excess)
        l_mmi = pb.mmi_total_loss_db(n_ports, mmi_excess)

        length2 = self._f(self.v_length2)
        l_prop2 = pb.propagation_loss_db(alpha, length2)

        l_out = self._f(self.v_l_out) if self.include_out_coupler.get() else 0.0

        mod_length = self._f(self.v_mod_length)
        l_mod_il = pb.propagation_loss_db(alpha, mod_length)
        er_db = self._f(self.v_mod_er)
        l_er_penalty = pb.er_power_penalty_db(er_db)

        stages = pb.compute_power_budget(
            p_laser, l_in, l_prop1, l_mmi, l_prop2, l_mod_il, l_er_penalty, l_out
        )

        responsivity = self._f(self.v_responsivity)
        wavelength_nm = self._f(self.v_wavelength)
        eta_qe = pb.qe_from_responsivity(responsivity, wavelength_nm)
        qe_penalty = pb.qe_penalty_db(eta_qe)

        p_pd_dbm = stages[-2][1]  # power at photodetector, before the total-loss entry
        photocurrent = pb.pd_metrics(responsivity, p_pd_dbm)

        if self.mod_efficiency_mode.get() == "Capacitive (CV^2)":
            e_bit_pJ = pb.modulation_efficiency_capacitive_pJ(self._f(self.v_cap_fF), self._f(self.v_vpp))
        else:
            e_bit_pJ = pb.modulation_efficiency_power_pJ(self._f(self.v_p_drive), self._f(self.v_bit_rate))

        total_loss = stages[-1][1]

        for name, value in stages[:-1]:
            self.budget_tree.insert("", tk.END, values=(name, f"{value:.3f}"))

        self.v_out_power.set(f"{p_pd_dbm:.3f}")
        self.v_total_loss.set(f"{total_loss:.3f}")
        self.v_qe_penalty.set(f"{qe_penalty:.4f}")
        self.v_photocurrent.set(f"{photocurrent * 1e6:.4f}")
        self.v_mod_eff.set(f"{e_bit_pJ:.4f}")

        p_sensitivity = self._f(self.v_sensitivity)
        available = pb.available_power_db(p_laser, p_sensitivity)
        power_budget = pb.power_budget_db(p_laser, p_sensitivity, total_loss)

        self.v_available_power.set(f"{available:.3f}")
        self.v_power_budget.set(f"{power_budget:.3f}")


def main():
    root = tk.Tk()
    PowerBudgetGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
