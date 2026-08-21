"""Schematic renderer for the photonic link structure.

A faithful recreation of pictures/MMI.png's layout -- NOT to scale, same
fixed visual layout every time -- generalized to the actual number of
MZM arm-pairs (n_ports/2) and labeled with the real dimension values for
whichever parameters are currently entered in the calculator, drawn onto
a plain tkinter.Canvas. MMI bodies are plain rectangles (not tapered).
"""

BG = "white"
LINE = "black"
LINE_WIDTH = 2

LASER_W, LASER_H = 170, 70
EDGE_COUPLER_W, EDGE_COUPLER_H = 24, 44
ICON_GAP = 20          # gap between the substrate box and the laser/photodetector pills
WG_STUB_W = 90
MMI_BODY_W = 170
MMI_ARM_MARGIN = 34     # body half-height beyond the outermost arm
ARM_SEG_W = 130
ELECTRODE_W, ELECTRODE_H = 130, 46
ARM_GAP = 46            # vertical gap between the 2 arms of one MZM
MZM_GAP = 36            # extra vertical gap between different MZMs
SUBSTRATE_MARGIN = 130
OUTER_MARGIN = 34


FONT_BIG = ("", 18)
FONT_PILL = ("", 12)   # Laser/Photodetector are drawn with this by default
FONT_MED = ("", 11)
FONT_SMALL = ("", 10)
FONT_TINY = ("", 8)     # edge coupler / port labels


def _core_geometry(params, result):
    n_ports = max(2, int(round(params["n_ports"] / 2)) * 2)
    return {
        "n_ports": n_ports,
        "num_mzms": n_ports // 2,
        "core_width_um": params["core_width"],
        "core_height_um": params["core_height"],
        "length1_um": params["length1"],
        "mmi_width_um": params["mmi_width"],
        "mmi_length_um": result["mmi_length"],
        "length2_um": params["length2"],
        "mod_length_um": params["mod_length"],
        "recomb_width_um": params["recomb_width"],
        "recomb_length_um": result["recomb_length"],
    }


def draw_structure(canvas, params, result):
    canvas.delete("all")
    geo = _core_geometry(params, result)
    num_mzms = geo["num_mzms"]

    def label(x, y, text, font=FONT_SMALL, anchor="center"):
        canvas.create_text(x, y, text=text, font=font, anchor=anchor)

    def pill(x0, y0, x1, y1, text):
        r = (y1 - y0) / 2
        canvas.create_arc(x0, y0, x0 + 2 * r, y1, start=90, extent=180, style="arc", width=LINE_WIDTH)
        canvas.create_arc(x1 - 2 * r, y0, x1, y1, start=-90, extent=180, style="arc", width=LINE_WIDTH)
        canvas.create_line(x0 + r, y0, x1 - r, y0, width=LINE_WIDTH)
        canvas.create_line(x0 + r, y1, x1 - r, y1, width=LINE_WIDTH)
        label((x0 + x1) / 2, (y0 + y1) / 2, text, font=FONT_PILL)

    def rect(x0, y0, x1, y1, text=None, font=FONT_SMALL):
        canvas.create_rectangle(x0, y0, x1, y1, outline=LINE, width=LINE_WIDTH)
        if text:
            label((x0 + x1) / 2, (y0 + y1) / 2, text, font=font)

    # --- vertical layout: stack MZM arm-pairs, each pair ARM_GAP apart,
    # MZM-to-MZM separated by an extra MZM_GAP ---
    pair_span = ARM_GAP
    block_h = num_mzms * pair_span + (num_mzms - 1) * MZM_GAP
    mid_y = OUTER_MARGIN + SUBSTRATE_MARGIN + block_h / 2 + MMI_ARM_MARGIN

    arm_ys = []
    top = mid_y - block_h / 2
    for m in range(num_mzms):
        y0 = top + m * (pair_span + MZM_GAP)
        arm_ys.append((y0, y0 + pair_span))

    mmi_body_h = block_h + 2 * MMI_ARM_MARGIN
    substrate_h = mmi_body_h + 2 * SUBSTRATE_MARGIN

    # --- horizontal layout ---
    cx = OUTER_MARGIN
    laser_x0 = cx
    cx += LASER_W + ICON_GAP
    substrate_x0 = cx
    edge1_x0 = cx
    cx += EDGE_COUPLER_W
    wg1_x0 = cx
    cx += WG_STUB_W
    mmi1_x0 = cx
    cx += MMI_BODY_W
    arm1_x0 = cx
    cx += ARM_SEG_W
    electrode_x0 = cx
    cx += ELECTRODE_W
    arm2_x0 = cx
    cx += ARM_SEG_W
    mmi2_x0 = cx
    cx += MMI_BODY_W
    wg2_x0 = cx
    cx += WG_STUB_W
    edge2_x0 = cx
    cx += EDGE_COUPLER_W
    substrate_x1 = cx
    photo_x0 = cx + ICON_GAP

    substrate_y0 = OUTER_MARGIN
    substrate_y1 = OUTER_MARGIN + substrate_h

    # Substrate outline
    rect(substrate_x0, substrate_y0, substrate_x1, substrate_y1)
    label(substrate_x0 + 10, substrate_y1 - 14, "", font=("", 11, "italic"), anchor="w")

    # Laser / Photodetector
    pill(laser_x0, mid_y - LASER_H / 2, laser_x0 + LASER_W, mid_y + LASER_H / 2, "Laser")
    pill(photo_x0, mid_y - LASER_H / 2, photo_x0 + LASER_W + 80, mid_y + LASER_H / 2, "Photodetector")
    canvas.create_line(laser_x0 + LASER_W, mid_y, edge1_x0, mid_y, width=LINE_WIDTH)
    canvas.create_line(edge2_x0 + EDGE_COUPLER_W, mid_y, photo_x0, mid_y, width=LINE_WIDTH)

    # Edge couplers
    rect(edge1_x0, mid_y - EDGE_COUPLER_H / 2, edge1_x0 + EDGE_COUPLER_W, mid_y + EDGE_COUPLER_H / 2)
    rect(edge2_x0, mid_y - EDGE_COUPLER_H / 2, edge2_x0 + EDGE_COUPLER_W, mid_y + EDGE_COUPLER_H / 2)
    label(edge1_x0 + EDGE_COUPLER_W / 2, mid_y - EDGE_COUPLER_H / 2 - 14, "", font=FONT_TINY)
    label(edge2_x0 + EDGE_COUPLER_W / 2, mid_y - EDGE_COUPLER_H / 2 - 14, "", font=FONT_TINY)

    # Waveguide stubs (real: length1_um x core_width_um)
    canvas.create_line(wg1_x0, mid_y, mmi1_x0, mid_y, width=LINE_WIDTH)
    canvas.create_line(mmi2_x0 + MMI_BODY_W, mid_y, wg2_x0 + WG_STUB_W, mid_y, width=LINE_WIDTH)
    wg_label = f"waveguide: {geo['length1_um']:g}um x {geo['core_width_um']:g}um"
    label(wg1_x0 + WG_STUB_W / 2 + 110, mid_y + 75, wg_label, font=FONT_SMALL)
    label(wg2_x0 + WG_STUB_W / 2 - 110, mid_y + 75, wg_label, font=FONT_SMALL)

    # --- Splitter / recombiner MMI bodies (plain rectangles) ---
    def mmi_block(x0, real_len_um, real_w_um, count):
        top_body = mid_y - mmi_body_h / 2
        rect(x0, top_body, x0 + MMI_BODY_W, mid_y + mmi_body_h / 2)
        label(x0 + MMI_BODY_W / 2, mid_y, "MMI", font=FONT_BIG)
        label(x0 + MMI_BODY_W / 2, top_body - 25,
              f"{real_len_um:g}um x {real_w_um:g}um, {count} ports", font=FONT_MED)

    mmi_block(mmi1_x0, geo["mmi_length_um"], geo["mmi_width_um"], geo["n_ports"])
    mmi_block(mmi2_x0, geo["recomb_length_um"], geo["recomb_width_um"], 2)

    # --- Arms: splitter output -> Electrode -> recombiner input, one row
    # per MZM, fanning out from/into the MMI faces ---
    for (y0, y1) in arm_ys:
        for arm_y in (y0, y1):
            canvas.create_line(mmi1_x0 + MMI_BODY_W, mid_y, arm1_x0, arm_y, width=LINE_WIDTH)
            canvas.create_line(arm1_x0, arm_y, electrode_x0, arm_y, width=LINE_WIDTH)
            canvas.create_line(electrode_x0 + ELECTRODE_W, arm_y, arm2_x0 + ARM_SEG_W, arm_y, width=LINE_WIDTH)
            canvas.create_line(arm2_x0 + ARM_SEG_W, arm_y, mmi2_x0, mid_y, width=LINE_WIDTH)
        label(arm1_x0 + ARM_SEG_W / 2, y0 - 10, "output port", font=FONT_TINY)
        label(arm2_x0 + ARM_SEG_W / 2, y0 - 10, "input port", font=FONT_TINY)
        label(arm1_x0 + ARM_SEG_W / 2, y1 - 10, "output port", font=FONT_TINY)
        label(arm2_x0 + ARM_SEG_W / 2, y1 - 10, "input port", font=FONT_TINY)
        rect(electrode_x0, y0 - ELECTRODE_H / 2, electrode_x0 + ELECTRODE_W, y1 + ELECTRODE_H / 2,
             f"{geo['mod_length_um']:g}um", font=FONT_MED)

    note = "Note: number of ports varies and is always a multiple of 2"
    label((substrate_x0 + substrate_x1) / 2, substrate_y1 + 20, note, font=FONT_MED)

    # --- Center the whole drawing within the canvas's current size ---
    canvas.update_idletasks()
    bbox = canvas.bbox("all")
    if bbox is not None:
        bx0, by0, bx1, by1 = bbox
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        dx = (canvas_w - (bx1 - bx0)) / 2 - bx0
        dy = (canvas_h - (by1 - by0)) / 2 - by0
        canvas.move("all", dx, dy)
        canvas.configure(scrollregion=canvas.bbox("all"))
