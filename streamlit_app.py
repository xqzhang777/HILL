import streamlit as st
import numpy as np
from bokeh.plotting import figure
from bokeh.models import CustomJS, Span, LinearColorMapper
from bokeh.models import HoverTool, CrosshairTool
from bokeh.events import MouseMove, DoubleTap
from bokeh.layouts import gridplot

def main(args):
    st.set_page_config(page_title="Helical Indexing", layout="wide")
    st.server.server_util.MESSAGE_SIZE_LIMIT = 2e8  # default is 5e7 (50MB)

    query_params = st.experimental_get_query_params()

    title = "Helical indexing using layer lines"
    st.title(title)

    import itertools
    counter = itertools.count() # to generate unique keys required by streamlit widgets

    col1, col2, col3, col4 = st.beta_columns((1., 0.5, 0.25, 3.0))

    with col1:
        with st.beta_expander(label="README", expanded=False):
            st.write("This Web app considers a biological helical structure as the product of a continous helix and a set of parallel planes, and based on the covolution theory, the Fourier Transform (FT) of a helical structure would be the convolution of the FT of the continous helix and the FT of the planes.  \nThe FT of a continous helix consists of equally spaced layer planes (3D) or layerlines (2D projection) that can be described by Bessel functions of increasing orders (0, +/-1, +/-2, ...) from the Fourier origin (i.e. equator). The spacing between the layer planes/lines is determined by the helical pitch (i.e. the shift along the helical axis for a 360 ° turn of the helix). If the structure has additional cyclic symmetry (for example, C6) around the helical axis, only the layer plane/line orders of integer multiplier of the symmetry (e.g. 0, +/-6, +/-12, ...) are visible. The primary peaks of the layer lines in the power spectra form a pattern similar to a X symbol.  \nThe FT of the parallel planes consists of equally spaced points along the helical axis (i.e. meridian) with the spacing being determined by the helical rise.  \nThe convolution of these two components (X-shaped pattern of layer lines and points along the meridian) generates the layer line patterns seen in the power spectra of the projection images of helical structures. The helical indexing task is thus to identify the helical rise, pitch (or twist), and cyclic symmetry that would predict a layer line pattern to explain the observed the layer lines in the power spectra. This Web app allows you to interactively change the helical parameters and superimpose the predicted layer liines on the power spectra to complete the helical indexing task.  \n  \nPS: power spectra; PD: phase difference between the two sides of meridian; YP: Y-axis power spectra profile; LL: layer lines; m: indices of the X-patterns along the meridian; Jn: Bessel order")
        
        show_initial_message(message=args.message) # only show once on the first run

        # make radio display horizontal
        st.markdown('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        data, apix, radius_auto, mask_radius, is_pwr, input_params, (image_container, image_label) = obtain_input_image(col1, args, query_params, counter)
        input_mode, (uploaded_filename, url, emdid) = input_params

        if is_pwr:
            label = f"Add phases from another image"
        else:
            label = f"Replace amplitudes with another image"
        input_image2 = st.checkbox(label=label, value=False)        
        if input_image2:
            data2, apix2, radius_auto2, mask_radius2, is_pwr2, input_params2, _ = obtain_input_image(col1, args, query_params, counter)
            input_mode2, (uploaded_filename2, url2, emdid2) = input_params2

        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu). Report problems to Wen Jiang (jiang12 at purdue.edu)*")

    with col2:
        pitch_or_twist = st.beta_container()
        value = float(query_params["rise"][0]) if "rise" in query_params else data_example.rise
        rise = st.number_input('Rise (Å)', value=value, min_value=-180.0, max_value=180.0, step=1.0, format="%.3f")
        value = int(query_params["csym"][0]) if "csym" in query_params else data_example.csym
        csym = st.number_input('Csym', value=value, min_value=1, max_value=16, step=1)

        if input_image2: value = max(radius_auto*apix, radius_auto2*apix2)
        else: value = radius_auto*apix
        value = float(query_params["radius"][0]) if "radius" in query_params else value
        if value<=1: value = 100.0
        helical_radius = st.number_input('Radius (Å)', value=value, min_value=1.0, max_value=1000.0, step=10., format="%.1f")
        
        tilt = st.number_input('Out-of-plane tilt (°)', value=0.0, min_value=-90.0, max_value=90.0, step=1.0)
        value = float(query_params["resx"][0]) if "resx" in query_params else round(3*apix, 0)
        cutoff_res_x = st.number_input('Resolution limit - X (Å)', value=value, min_value=2*apix, step=1.0)
        value = float(query_params["resy"][0]) if "resy" in query_params else round(3*apix, 0)
        cutoff_res_y = st.number_input('Resolution limit - Y (Å)', value=value, min_value=2*apix, step=1.0)
        with st.beta_expander(label="Filters", expanded=False):
            log_xform = st.checkbox(label="Log(amplitude)", value=True)
            hp_fraction = st.number_input('Fourier high-pass (%)', value=0.4, min_value=0.0, max_value=100.0, step=0.1, format="%.2f") / 100.0
            lp_fraction = st.number_input('Fourier low-pass (%)', value=0.0, min_value=0.0, max_value=100.0, step=10.0, format="%.2f") / 100.0
            ny, nx = data.shape
            pnx = st.number_input('FFT X-dim size (pixels)', value=max(min(nx,ny), 512), min_value=min(nx, 128), step=2)
            pny = st.number_input('FFT Y-dim size (pixels)', value=max(max(nx,ny), 1024), min_value=min(ny, 512), step=2)
        with st.beta_expander(label="Simulation", expanded=False):
            ball_radius = st.number_input('Gaussian radius (Å)', value=0.0, min_value=0.0, max_value=helical_radius, step=5.0, format="%.1f")
            show_simu = True if ball_radius > 0 else False
            if show_simu:
                az = st.number_input('Azimuthal angle (°)', value=0, min_value=0, max_value=360, step=1, format="%.1f")
                noise = st.number_input('Noise (sigma)', value=0., min_value=0., step=1., format="%.1f")
                movie_frames = st.number_input('Tilt movie frame #', value=0, min_value=0, max_value=1000, step=1)
                use_plot_size = st.checkbox('Use plot size', value=False)

    if is_pwr:
        pwr = resize_rescale_power_spectra(data, nyquist_res=2*apix, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                output_size=(pny, pnx), log=log_xform, low_pass_fraction=lp_fraction, high_pass_fraction=hp_fraction)
        phase = None
    else:
        pwr, phase = compute_power_spectra(data, apix=apix, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                output_size=(pny, pnx), log=log_xform, low_pass_fraction=lp_fraction, high_pass_fraction=hp_fraction)
    
    if input_image2:
        if is_pwr2:
            pwr2 = resize_rescale_power_spectra(data2, nyquist_res=2*apix2, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                    output_size=(pny, pnx), log=log_xform, low_pass_fraction=lp_fraction, high_pass_fraction=hp_fraction)
            phase2 = None
        else:
            pwr2, phase2 = compute_power_spectra(data2, apix=apix2, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                    output_size=(pny, pnx), log=log_xform, low_pass_fraction=lp_fraction, high_pass_fraction=hp_fraction)
    else:
        pwr2 = None
        phase2 = None

    with col3:
        st.subheader("Display:")
        value = int(query_params["show_pitch"][0]) if "show_pitch" in query_params else True
        show_pitch = st.checkbox(label="Pitch", value=value)
        with pitch_or_twist:
            if show_pitch:
                value = float(query_params["pitch"][0]) if "pitch" in query_params else data_example.pitch
                pitch = st.number_input('Pitch (Å)', value=value, min_value=1.0, max_value=max(pny,pnx)*apix, step=1.0, format="%.2f")
                if pitch < cutoff_res_y:
                    st.warning(f"pitch is too small. it should be > {cutoff_res_y} (Limit FFT X-dim to resolution (Å))")
                    return
                twist = 360./(pitch/rise)
                st.markdown(f"*(twist = {twist:.2f} °)*")
            else:
                value = float(query_params["twist"][0]) if "twist" in query_params else data_example.twist
                twist = st.number_input('Twist (°)', value=value, min_value=-180.0, max_value=180.0, step=1.0, format="%.2f")
                pitch = (360./twist)*rise
                st.markdown(f"*(pitch = {pitch:.2f} Å)*")

        show_phase = False
        show_phase_diff = False
        show_yprofile = False
        show_pwr = st.checkbox(label="PS", value=True)
        if show_pwr:
            show_yprofile = st.checkbox(label="YP", value=False)
        if not is_pwr:
            show_phase = st.checkbox(label="Phase", value=False)
            show_phase_diff = st.checkbox(label="PD", value=True)
        
        show_pwr2 = False
        show_phase2 = False
        show_phase_diff2 = False
        show_yprofile2 = False
        if input_image2:
            show_pwr2 = st.checkbox(label="PS2", value=False if show_pwr else True)
            if show_pwr2:
                show_yprofile2 = st.checkbox(label="YP2", value=False)
            if not is_pwr2:
                show_phase2 = st.checkbox(label="Phase2", value=False)
                show_phase_diff2 = st.checkbox(label="PD2", value=False if show_phase_diff else True)

        if show_pwr or show_pwr2:
            show_pseudocolor = st.checkbox(label="Color", value=True)
            value=False if input_mode==0 or is_pwr or (input_image2 and (input_mode2==0  or is_pwr2)) else True
            show_LL = st.checkbox(label="LL", value=value)
            if show_LL:
                m_groups = compute_layer_line_positions(twist=twist, rise=rise, csym=csym, radius=helical_radius, tilt=tilt, cutoff_res=cutoff_res_y)
                ng = len(m_groups)
                st.subheader("m=")
                show_choices = {}
                lgs = sorted(m_groups.keys())[::-1]
                for lgi, lg in enumerate(lgs):
                    value = True if lg in [0, 1] else False
                    show_choices[lg] = st.checkbox(label=str(lg), value=value)

    if show_simu:
        proj = simulate_helix(twist, rise, csym, helical_radius=helical_radius, ball_radius=ball_radius, 
                ny=data.shape[0], nx=data.shape[1], apix=apix, tilt=tilt, az0=az)
        if noise>0:
            sigma = np.std(proj[np.nonzero(proj)])
            proj = proj + np.random.normal(loc=0.0, scale=noise*sigma, size=proj.shape)
        fraction_x = mask_radius/(proj.shape[1]//2*apix)
        tapering_image = generate_tapering_filter(image_size=proj.shape, fraction_start=[0.9, fraction_x], fraction_slope=0.1)
        proj = proj * tapering_image
        with image_container:
            st.image([data, proj], use_column_width=True, caption=[image_label, "Simulated"], clamp=True)

    with col4:
        if not (show_pwr or show_pwr2): return

        items = [ (show_pwr, pwr, "Power Spectra", show_phase, show_phase_diff, phase, "Phase Diff Across Meridian", show_yprofile), 
                  (show_pwr2, pwr2, "Power Spectra - 2", show_phase2, show_phase_diff2, phase2, "Phase Diff Across Meridian - 2", show_yprofile2)
                ]
        if show_simu and show_pwr:
            if use_plot_size:
                apix_simu = min(cutoff_res_y, cutoff_res_x)/2
                proj = simulate_helix(twist, rise, csym, helical_radius=helical_radius, ball_radius=ball_radius, 
                        ny=pny, nx=pnx, apix=apix_simu, tilt=tilt, az0=az)
                if noise>0:
                    sigma = np.std(proj[np.nonzero(proj)])
                    proj = proj + np.random.normal(loc=0.0, scale=noise*sigma, size=proj.shape)
                fraction_x = mask_radius/(proj.shape[1]//2*apix_simu)
                tapering_image = generate_tapering_filter(image_size=proj.shape, fraction_start=[0.9, fraction_x], fraction_slope=0.1)
                proj = proj * tapering_image
            else:
                apix_simu = apix
            proj_pwr, proj_phase = compute_power_spectra(proj, apix=apix_simu, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                    output_size=(pny, pnx), log=log_xform, low_pass_fraction=lp_fraction, high_pass_fraction=hp_fraction)
            items += [(show_pwr or show_pwr2, proj_pwr, "Simulated Power Spectra", show_phase or show_phase2, show_phase_diff or show_phase_diff2, proj_phase, "Simulated Phase Diff Across Meridian", show_yprofile or show_yprofile2)]

        figs = []
        figs_skip_yprofile = []
        for item in items:
            show_pwr_work, pwr_work, title_pwr_work, show_phase_work, show_phase_diff_work, phase_work, title_phase_work, show_yprofile_work = item
            if show_pwr_work:
                tooltips = [("Res r", "Å"), ('Res y', 'Å'), ('Res x', 'Å'), ('Jn', '@bessel'), ('Amp', '@image')]
                fig = create_image_figure(pwr_work, cutoff_res_x, cutoff_res_y, helical_radius, tilt, phase=phase_work if show_phase_work else None, pseudocolor=show_pseudocolor, title=title_pwr_work, yaxis_visible=False, tooltips=tooltips)
                figs.append(fig)

            if show_yprofile_work:
                ny, nx = pwr_work.shape
                dsy = 1/(ny//2*cutoff_res_y)
                y=np.arange(-ny//2, ny//2)*dsy
                yprofile = np.mean(pwr_work, axis=1)
                yprofile /= yprofile.max()
                source_data = dict(yprofile=yprofile, y=y, resy=np.abs(1./y))
                tools = 'box_zoom,hover,pan,reset,save,wheel_zoom'
                tooltips = [('Res y', '@resyÅ'), ('Amp', '$x')]
                fig = figure(frame_width=nx//2, frame_height=figs[-1].frame_height, y_range=figs[-1].y_range, y_axis_location = "right", 
                    title=None, tools=tools, tooltips=tooltips)
                fig.line(source=source_data, x='yprofile', y='y', line_width=2, color='blue')
                fig.yaxis.visible = False
                fig.hover[0].attachment="vertical"
                figs.append(fig)
                figs_skip_yprofile.append(fig)

            if show_phase_diff_work:
                ny, nx = phase_work.shape
                if nx%2:
                    phase_diff = phase_work - phase_work[:, ::-1]
                else:
                    phase_diff = phase_work * 1.0
                    phase_diff[:, 0] = np.pi/2
                    phase_diff[:, 1:] -= phase_diff[:, 1:][:, ::-1]
                phase_diff = np.rad2deg(np.arccos(np.cos(phase_diff)))   # set the range to [0, 180]. 0 -> even order, 180 - odd order
                
                tooltips = [("Res r", "Å"), ('Res y', 'Å'), ('Res x', 'Å'), ('Jn', '@bessel'), ('Phase Diff', '@image °')]
                fig = create_image_figure(phase_diff, cutoff_res_x, cutoff_res_y, helical_radius, tilt, phase=phase_work if show_phase_work else None, pseudocolor=show_pseudocolor, title=title_phase_work, yaxis_visible=False, tooltips=tooltips)
                figs.append(fig)
                
        if figs and show_LL:
            if max(m_groups[0]["LL"][0])>0:
                from bokeh.palettes import viridis, gray
                if show_pseudocolor:
                    ll_colors = gray(ng*2)[::-1]
                else:
                    ll_colors = viridis(ng*2)[::-1]
                ll_colors = np.array(ll_colors)[distinct_sampling(ng)]                
                x, y = m_groups[0]["LL"]
                tmp_x = np.sort(np.unique(x))
                width = np.mean(tmp_x[1:]-tmp_x[:-1])
                tmp_y = np.sort(np.unique(y))
                height = np.mean(tmp_y[1:]-tmp_y[:-1])/3
                for mi, m in enumerate(m_groups.keys()):
                    if not show_choices[m]: continue
                    x, y = m_groups[m]["LL"]
                    color = ll_colors[mi]
                    for f in figs:
                        if f in figs_skip_yprofile: continue
                        f.ellipse(x, y, width=width, height=height, line_width=4, line_color=color, fill_alpha=0)
            else:
                st.warning(f"No off-equator layer lines to draw for Pitch={pitch:.2f} Csym={csym} combinations. Consider increasing Pitch or reducing Csym")

        figs[-1].yaxis.fixed_location = figs[-1].x_range.end
        figs[-1].yaxis.visible = True
        add_linked_crosshair_tool(figs)

        all_figs = gridplot(children=[figs], toolbar_location='right')
        
        st.bokeh_chart(all_figs, use_container_width=False)

        if show_simu and movie_frames>0:
            with st.spinner(text="Generating movie of tilted power spectra/phases ..."):
                if use_plot_size:
                    ny, nx = pny, pnx
                    apix_simu = min(cutoff_res_y, cutoff_res_x)/2
                else:
                    ny, nx = data.shape
                    apix_simu = apix
                movie_filename = create_simulation_movie(movie_frames, tilt, twist, rise, csym, helical_radius, ball_radius, az, noise, ny, nx, pny, pnx, apix_simu, mask_radius, cutoff_res_x, cutoff_res_y, show_pseudocolor, log_xform, lp_fraction, hp_fraction)
                st.video(movie_filename) # it always show the video using the entire column width
       
    if input_mode in [2, 1]:
        params=dict(input_mode=input_mode)
        if input_mode == 2:
            params["emdid"] = emdid
        elif input_mode == 1:
            params["url"] = url
        params["show_pitch"] = int(show_pitch)
        if show_pitch:
            params["pitch"] = round(pitch,3)
        else:
            params["twist"] = round(twist,3)
        params.update(dict(rise=round(rise,3), csym=csym, radius=round(helical_radius,1), resx=round(cutoff_res_x,2), resy=round(cutoff_res_y,2)))
        if is_pwr: params["is_pwr"] = int(is_pwr)
    else:
        params = dict()
    st.experimental_set_query_params(**params)

@st.cache(persist=True, show_spinner=False, suppress_st_warning=True)
def create_simulation_movie(movie_frames, tilt_max, twist, rise, csym, helical_radius, ball_radius, az, noise, ny, nx, pny, pnx, apix, mask_radius, cutoff_res_x, cutoff_res_y, show_pseudocolor, log_xform, low_pass_fraction, high_pass_fraction):
    tilt_step = tilt_max/movie_frames
    fraction_x = mask_radius/(nx//2*apix)
    tapering_image = generate_tapering_filter(image_size=(ny, nx), fraction_start=[0.9, fraction_x], fraction_slope=0.1)
    image_filenames = []
    from bokeh.io import export_png
    progress_bar = st.empty()
    progress_bar.progress(0.0)
    for i in range(movie_frames+1):
        tilt = tilt_step * i
        proj = simulate_helix(twist, rise, csym, helical_radius=helical_radius, ball_radius=ball_radius, 
                ny=ny, nx=nx, apix=apix, tilt=tilt, az0=az)
        if noise>0:
            sigma = np.std(proj[np.nonzero(proj)])
            proj = proj + np.random.normal(loc=0.0, scale=noise*sigma, size=proj.shape)
        proj = proj * tapering_image
        proj_pwr, proj_phase = compute_power_spectra(proj, apix=apix, cutoff_res=(cutoff_res_y, cutoff_res_x), 
            output_size=(pny, pnx), log=log_xform, low_pass_fraction=low_pass_fraction, high_pass_fraction=high_pass_fraction)
        tooltips = [("Res r", "Å"), ('Res y', 'Å'), ('Res x', 'Å'), ('Jn', '@bessel'), ('Amp', '@image')]
        title = f"Simulated Power Spectra"
        fig_pwr = create_image_figure(proj_pwr, cutoff_res_x, cutoff_res_y, helical_radius, tilt, phase=None, pseudocolor=show_pseudocolor, title=title, yaxis_visible=False, tooltips=tooltips)
        if nx%2:
            phase_diff = proj_phase - proj_phase[:, ::-1]
        else:
            phase_diff = proj_phase * 1.0
            phase_diff[:, 0] = np.pi/2
            phase_diff[:, 1:] -= phase_diff[:, 1:][:, ::-1]
        phase_diff = np.rad2deg(np.arccos(np.cos(phase_diff)))   # set the range to [0, 180]. 0 -> even order, 180 - odd order
        tooltips = [("Res r", "Å"), ('Res y', 'Å'), ('Res x', 'Å'), ('Jn', '@bessel'), ('Phase Diff', '@image °')]
        title = f"Phase Diff Across Meridian (tilt={tilt:.2f}°)"
        fig_phase = create_image_figure(phase_diff, cutoff_res_x, cutoff_res_y, helical_radius, tilt, phase=None, pseudocolor=show_pseudocolor, title=title, yaxis_visible=False, tooltips=tooltips)
        figs = gridplot(children=[[fig_pwr, fig_phase]], toolbar_location=None)
        filename = f"image_{i:05d}.png"
        image_filenames.append(filename)
        export_png(figs, filename=filename)
        progress_bar.progress((i+1)/(movie_frames+1)) 
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    from moviepy.video.fx.all import resize
    movie = ImageSequenceClip(image_filenames, fps=min(20, movie_frames/5))
    movie_filename = "movie.mp4"
    movie.write_videofile(movie_filename)
    import os
    for f in image_filenames: os.remove(f)
    progress_bar.empty()
    return movie_filename

def create_image_figure(data, cutoff_res_x, cutoff_res_y, helical_radius, tilt, phase=None, pseudocolor=True, title="", yaxis_visible=True, tooltips=None):
    ny, nx = data.shape
    dsy = 1/(ny//2*cutoff_res_y)
    dsx = 1/(nx//2*cutoff_res_x)
    x_range = (-nx//2*dsx, nx//2*dsx)
    y_range = (-ny//2*dsy, ny//2*dsy)

    bessel = bessel_n_image(ny, nx, cutoff_res_x, cutoff_res_y, helical_radius, tilt).astype(np.int16)

    tools = 'box_zoom,pan,reset,save,wheel_zoom'
    fig = figure(title_location="below", frame_width=nx, frame_height=ny, 
        x_axis_label=None, y_axis_label=None, x_range=x_range, y_range=y_range, tools=tools)
    fig.grid.visible = False
    fig.title.text = title
    fig.title.align = "center"
    fig.title.text_font_size = "20px"
    fig.yaxis.visible = yaxis_visible   # leaving yaxis on will make the crosshair x-position out of sync with other figures

    source_data = dict(image=[data.astype(np.float16)], x=[-nx//2*dsx], y=[-ny//2*dsy], dw=[nx*dsx], dh=[ny*dsy], bessel=[bessel])
    if phase is not None: source_data["phase"] = [np.fmod(np.rad2deg(phase)+360, 360).astype(np.float16)]
    palette = 'Viridis256' if pseudocolor else 'Greys256'
    color_mapper = LinearColorMapper(palette=palette)    # Greys256, Viridis256
    image = fig.image(source=source_data, image='image', color_mapper=color_mapper, x='x', y='y', dw='dw', dh='dh')
    if tooltips is None:
        tooltips = [("Res r", "Å"), ('Res y', 'Å'), ('Res x', 'Å'), ('Jn', '@bessel'), ('Val', '@image')]
    if phase is not None: tooltips.append(("Phase", "@phase °"))
    image_hover = HoverTool(renderers=[image], tooltips=tooltips, attachment="vertical")
    fig.add_tools(image_hover)

    # avoid the need for embedding resr/resy/resx image -> smaller fig object and less data to transfer
    mousemove_callback_code = """
    var x = cb_obj.x
    var y = cb_obj.y
    var resr = Math.round((1./Math.sqrt(x*x + y*y) + Number.EPSILON) * 100) / 100
    var resy = Math.abs(Math.round((1./y + Number.EPSILON) * 100) / 100)
    var resx = Math.abs(Math.round((1./x + Number.EPSILON) * 100) / 100)
    hover.tooltips[0][1] = resr.toString() + " Å"
    hover.tooltips[1][1] = resy.toString() + " Å"
    hover.tooltips[2][1] = resx.toString() + " Å"
    """
    mousemove_callback = CustomJS(args={"hover":fig.hover[0]}, code=mousemove_callback_code)
    fig.js_on_event(MouseMove, mousemove_callback)
    
    return fig

def add_linked_crosshair_tool(figures):
    # create a linked crosshair tool among the figures
    crosshair = CrosshairTool(dimensions="both")
    crosshair.line_color = 'red'
    for fig in figures:
        fig.add_tools(crosshair)

def obtain_input_image(column, args, query_params, counter):
    def next_key():
        return str(next(counter))
    with column:
        input_modes = {0:"upload a mrc/mrcs file", 1:"url", 2:"emd-xxxx"}
        value = int(query_params["input_mode"][0]) if "input_mode" in query_params else args.input_mode
        input_mode = st.radio(label="How to obtain the input image/map:", options=list(input_modes.keys()), format_func=lambda i:input_modes[i], index=value, key=next_key())
        is_3d = False
        if input_mode == 2:  # "emd-xxxx":
            label = "Input an EMDB ID (emd-xxxx):"
            value = query_params["emdid"][0] if "emdid" in query_params else "emd-2699"
            emdid = st.text_input(label=label, value=value, key=next_key())
            data_all, apix = get_emdb_map(emdid.strip())
            is_3d = True
        else:
            if input_mode == 0:  # "upload a mrc/mrcs file":
                fileobj = st.file_uploader("Upload a mrc or mrcs file ", type=['mrc', 'mrcs', 'map', 'map.gz', 'tnf'], key=next_key())
                if fileobj is not None:
                    data_all, apix = get_2d_image_from_uploaded_file(fileobj)
                else:
                    st.stop()
            elif input_mode == 1:   # "url":
                    label = "Input a url of 2D projection image(s) or a 3D map:"
                    value = query_params["url"][0] if "url" in query_params else data_example.url
                    image_url = st.text_input(label=label, value=value, key=next_key())
                    data_all, apix = get_2d_image_from_url(image_url.strip())
            nz, ny, nx = data_all.shape
            if nx==ny and nz>nx//4:
                is_3d_auto = True
                is_3d = st.checkbox(label=f"The input ({nx}x{ny}x{nz}) is a 3D map", value=is_3d_auto, key=next_key())
        if is_3d:
            if not np.any(data_all):
                st.warning("All voxels of the input 3D map have zero value")
                st.stop()
            
            with st.beta_expander(label="Generate 2-D projection from the 3-D map", expanded=False):
                az = st.number_input(label=f"Rotation around the helical axis (°):", min_value=0.0, max_value=360., value=0.0, step=1.0, key=next_key())
                tilt = st.number_input(label=f"Tilt (°):", min_value=-180.0, max_value=180., value=0.0, step=1.0, key=next_key())
                data = generate_projection(data_all, az=az, tilt=tilt)
        else:
            nonzeros = nonzero_images(data_all)
            if nonzeros is None:
                st.warning("All pixels of the input 2D images have zero value")
                st.stop()
            
            nz, ny, nx = data_all.shape
            if nz>1:
                if len(nonzeros)==nz:
                    image_index = st.slider(label=f"Choose an image (out of {nz}):", min_value=1, max_value=nz, value=1, step=1, key=next_key())
                else:
                    image_index = st.select_slider(label=f"Choose an image ({len(nonzeros)} non-zero images out of {nz}):", options=list(nonzeros+1), value=nonzeros[0]+1, key=next_key())
                image_index -= 1
            else:
                image_index = 0
            data = data_all[image_index]

        if not np.any(data):
            st.warning("All pixels of the 2D image have zero value")
            st.stop()

        ny, nx = data.shape
        original_image = st.empty()
        with original_image:
            st.image(data, use_column_width=True, caption=f"Orignal image ({nx}x{ny})", clamp=True, key=next_key())

        with st.beta_expander(label="Transpose/Rotate/Shift the image", expanded=False):
            is_pwr_auto = guess_if_is_power_spectra(data)
            is_pwr = st.checkbox(label="Input image is power spectra ", value=is_pwr_auto, key=next_key())
            transpose_auto = input_mode not in [2] and nx > ny
            transpose = st.checkbox(label='Transpose the image', value=transpose_auto, key=next_key())
            apix = st.number_input('Pixel size (Å/pixel)', value=apix, min_value=0.1, max_value=10., step=0.01, format="%.4f", key=next_key())
            if is_pwr or is_3d:
                angle_auto, dx_auto = 0., 0.
            else:
                angle_auto, dx_auto = auto_vertical_center(data)
            angle = st.number_input('Rotate (°) ', value=-angle_auto, min_value=-180., max_value=180., step=1.0, key=next_key())
            dx = st.number_input('Shift along X-dim (Å) ', value=dx_auto*apix, min_value=-nx*apix, max_value=nx*apix, step=1.0, key=next_key())
        
        transformed_image = st.empty()
        transformed = transpose or angle or dx
        if transpose:
            data = data.T
        if angle or dx:
            data = rotate_shift_image(data, angle=-angle, post_shift=(0, dx/apix), order=1)

        mask_radius = 0
        radius_auto = 0
        if not is_pwr:
            radius_auto, mask_radius_auto = estimate_radial_range(data, thresh_ratio=0.1)
            mask_radius = st.number_input('Mask radius (Å) ', value=mask_radius_auto*apix, min_value=1.0, max_value=nx/2*apix, step=1.0, format="%.1f", key=next_key())

            fraction_x = mask_radius/(nx//2*apix)
            tapering_image = generate_tapering_filter(image_size=data.shape, fraction_start=[0.9, fraction_x], fraction_slope=0.1)
            data = data * tapering_image
            transformed = 1

            x = np.arange(-nx//2, nx//2)*apix
            ymax = np.max(data, axis=0)
            ymean = np.mean(data, axis=0)

            tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
            tooltips = [("X", "@x{0.0}Å")]
            p = figure(x_axis_label="x (Å)", y_axis_label="pixel value", frame_height=ny, tools=tools, tooltips=tooltips)
            p.line(x, ymax, line_width=2, color='red', legend_label="max")
            p.line(-x, ymax, line_width=2, color='red', line_dash="dashed", legend_label="max flipped")
            p.line(x, ymean, line_width=2, color='blue', legend_label="mean")
            p.line(-x, ymean, line_width=2, color='blue', line_dash="dashed", legend_label="mean flipped")
            rmin_span = Span(location=-mask_radius, dimension='height', line_color='green', line_dash='dashed', line_width=3)
            rmax_span = Span(location=mask_radius, dimension='height', line_color='green', line_dash='dashed', line_width=3)
            p.add_layout(rmin_span)
            p.add_layout(rmax_span)
            p.legend.visible = False
            p.legend.location = "top_right"
            p.legend.click_policy="hide"
            toggle_legend_js_x = CustomJS(args=dict(leg=p.legend[0]), code="""
                if (leg.visible) {
                    leg.visible = false
                    }
                else {
                    leg.visible = true
                }
            """)
            p.js_on_event(DoubleTap, toggle_legend_js_x)
            st.bokeh_chart(p, use_container_width=True)
        
        if transformed:
            with transformed_image:
                st.image(data, use_column_width=True, caption="Transformed image", clamp=True, key=next_key())

        #if not is_pwr:
        if 0:
            y = np.arange(-ny//2, ny//2)*apix
            xmax = np.max(data, axis=1)
            xmean = np.mean(data, axis=1)

            tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
            tooltips = [("Y", "@y{0.0}Å")]
            p = figure(x_axis_label="pixel value", y_axis_label="y (Å)", frame_height=ny, tools=tools, tooltips=tooltips)
            p.line(xmax, y, line_width=2, color='red', legend_label="max")
            p.line(xmean, y, line_width=2, color='blue', legend_label="mean")
            p.legend.visible = False
            p.legend.location = "top_right"
            p.legend.click_policy="hide"
            toggle_legend_js_y = CustomJS(args=dict(leg=p.legend[0]), code="""
                if (leg.visible) {
                    leg.visible = false
                    }
                else {
                    leg.visible = true
                }
            """)
            p.js_on_event(DoubleTap, toggle_legend_js_y)
            st.bokeh_chart(p, use_container_width=True)

    if transformed:
        image_container = transformed_image
        image_label = "Transformed image"
    else:
        image_container = original_image
        image_label = "Original image"

    if input_mode==2:
        input_params = (input_mode, (None, None, emdid))
    elif input_mode==1:
        input_params = (input_mode, (None, image_url, None))
    else:
        input_params = (input_mode, (fileobj, None, None))
    return data, apix, radius_auto, mask_radius, is_pwr, input_params, (image_container, image_label)

@st.cache(persist=True, show_spinner=False)
def bessel_n_image(ny, nx, nyquist_res_x, nyquist_res_y, radius, tilt):
    def build_bessel_order_table(xmax):
        from scipy.special import jnp_zeros
        table = [0]
        n = 1
        while 1:
            peak_x = jnp_zeros(n, 1)[0] # first peak
            if peak_x<xmax: table.append(peak_x)
            else: break
            n += 1
        return np.asarray(table)

    import numpy as np
    if tilt:
        dsx = 1./(nyquist_res_x*nx//2)
        dsy = 1./(nyquist_res_x*ny//2)
        Y, X = np.meshgrid(np.arange(ny, dtype=np.float32)-ny//2, np.arange(nx, dtype=np.float32)-nx//2, indexing='ij')
        Y = 2*np.pi * np.abs(Y)*dsy * radius
        X = 2*np.pi * np.abs(X)*dsx * radius
        Y /= np.cos(np.deg2rad(tilt))
        X = np.hypot(X, Y*np.sin(np.deg2rad(tilt)))
        table = build_bessel_order_table(X.max())        
        X = np.expand_dims(X.flatten(), axis=-1)
        indices = np.abs(table - X).argmin(axis=-1)
        return np.reshape(indices, (ny, nx)).astype(np.int16)
    else:
        ds = 1./(nyquist_res_x*nx//2)
        xs = 2*np.pi * np.abs(np.arange(nx)-nx//2)*ds * radius
        table = build_bessel_order_table(xs.max())        
        xs = np.expand_dims(xs, axis=-1) 
        indices = np.abs(table - xs).argmin(axis=-1)
        return np.tile(indices, (ny, 1)).astype(np.int16)

@st.cache(persist=True, show_spinner=False)
def simulate_helix(twist, rise, csym, helical_radius, ball_radius, ny, nx, apix, tilt=0, az0=None):
    def simulate_projection(centers, sigma, ny, nx, apix):
        sigma2 = sigma*sigma
        d = np.zeros((ny, nx))
        Y, X = np.meshgrid(np.arange(0, ny, dtype=np.float32)-ny//2, np.arange(0, nx, dtype=np.float32)-nx//2, indexing='ij')
        X *= apix
        Y *= apix
        for ci in range(len(centers)):
            yc, xc = centers[ci]
            x = X-xc
            y = Y-yc
            d += np.exp(-(x*x+y*y)/sigma2)
        return d
    def helical_unit_positions(twist, rise, csym, radius, height, tilt=0, az0=0):
        imax = int(height/rise)
        i0 = -imax
        i1 = imax
        
        centers = np.zeros(((2*imax+1)*csym, 3), dtype=np.float32)
        for i in range(i0, i1+1):
            z = rise*i
            for si in range(csym):
                angle = np.deg2rad(twist*i + si*360./csym + az0 + 90)   # start from +y axis
                x = np.cos(angle) * radius
                y = np.sin(angle) * radius
                centers[i*csym+si, 0] = x
                centers[i*csym+si, 1] = y
                centers[i*csym+si, 2] = z
        if tilt:
            from scipy.spatial.transform import Rotation as R
            rot = R.from_euler('x', tilt, degrees=True)
            centers = rot.apply(centers)
        centers = centers[:, [2, 0]]    # project along y
        return centers
    if az0 is None: az0 = np.random.uniform(0, 360)
    centers = helical_unit_positions(twist, rise, csym, helical_radius, height=ny*apix, tilt=tilt, az0=az0)
    projection = simulate_projection(centers, ball_radius, ny, nx, apix)
    return projection

@st.cache(persist=True, show_spinner=False)
def distinct_sampling(n):
    assert(n>0)
    if n==1: return [0]
    all = list(range(1, n))
    ret = [0]
    while all:
        ref = ret[-2:][0]
        err_max = 0
        val_max = None
        for v in all:
            err = abs(v-ref)
            if err>err_max:
                err_max = err
                val_max = v
        ret.append(val_max)
        all.remove(val_max)
    return ret

@st.cache(persist=True, show_spinner=False)
def compute_layer_line_positions(twist, rise, csym, radius, tilt, cutoff_res, m=[]):
    def pitch(twist, rise):
        p = np.abs(rise * 360./twist)   # pitch in Å
        return p
    
    def peak_sx(bessel_order, radius):
        from scipy.special import jnp_zeros
        sx = np.zeros(len(bessel_order))
        for bi in range(len(bessel_order)):
            if bessel_order[bi]==0:
                sx[bi]=0
                continue
            peak1 = jnp_zeros(bessel_order[bi], 1)[0] # first peak of first visible layerline (n=csym)
            sx[bi] = peak1/(2*np.pi*radius)
        return sx

    if not m:
        m_max = int(np.floor(np.abs(rise/cutoff_res)))+2
        m = list(range(-m_max, m_max+1))
        m.sort(key=lambda x: (abs(x), x))   # 0, -1, 1, -2, 2, ...
    
    smax = 1./cutoff_res

    tf = 1./np.cos(np.deg2rad(tilt))
    tf2 = np.sin(np.deg2rad(tilt))
    m_groups = {} # one group per m order
    for mi in range(len(m)):
        d = {}
        sy0 = m[mi] / rise

        # first peak positions of each layer line
        p = pitch(twist, rise)
        ds_p = 1/p
        ll_i_top = int((smax - sy0)/ds_p)
        ll_i_bottom = -int(np.abs(-smax - sy0)/ds_p)
        ll_i = np.array([i for i in range(ll_i_bottom, ll_i_top+1) if not i%csym], dtype=np.float32)
        sy = sy0 + ll_i * ds_p
        sx = peak_sx(bessel_order=ll_i, radius=radius)
        if tilt:
            sy = np.array(sy, dtype=np.float32) * tf
            sx = np.sqrt(np.power(np.array(sx, dtype=np.float32), 2) - np.power(sy*tf2, 2))
            sx[np.isnan(sx)] = 1e-6
        px  = list(sx) + list(-sx)
        py  = list(sy) + list(sy)
        d["LL"] = (px, py)

        m_groups[m[mi]] = d
    return m_groups

@st.cache(persist=True, show_spinner=False)
def resize_rescale_power_spectra(data, nyquist_res, cutoff_res=None, output_size=None, log=True, low_pass_fraction=0, high_pass_fraction=0):
    from scipy.ndimage.interpolation import map_coordinates
    ny, nx = data.shape
    ony, onx = output_size
    res_y, res_x = cutoff_res
    Y, X = np.meshgrid(np.arange(ony, dtype=np.float32)-ony//2, np.arange(onx, dtype=np.float32)-onx//2, indexing='ij')
    Y = Y/(ony//2) * nyquist_res/res_y * ny//2 + ny//2
    X = X/(onx//2) * nyquist_res/res_x * nx//2 + nx//2
    pwr = map_coordinates(data, (Y.flatten(), X.flatten()), order=3, mode='constant').reshape(Y.shape)
    if log: pwr = np.log(np.abs(pwr))
    if 0<low_pass_fraction<1 or 0<high_pass_fraction<1:
        pwr = low_high_pass_filter(pwr, low_pass_fraction=low_pass_fraction, high_pass_fraction=high_pass_fraction)
    pwr = normalize(pwr, percentile=(0, 100))
    return pwr

@st.cache(persist=True, show_spinner=False)
def compute_power_spectra(data, apix, cutoff_res=None, output_size=None, log=True, low_pass_fraction=0, high_pass_fraction=0):
    fft = fft_rescale(data, apix=apix, cutoff_res=cutoff_res, output_size=output_size)
    fft = np.fft.fftshift(fft)  # shift fourier origin from corner to center

    if log: pwr = np.log(np.abs(fft))
    if 0<low_pass_fraction<1 or 0<high_pass_fraction<1:
        pwr = low_high_pass_filter(pwr, low_pass_fraction=low_pass_fraction, high_pass_fraction=high_pass_fraction)
    pwr = normalize(pwr, percentile=(0, 100))

    phase = np.angle(fft, deg=False)
    return pwr, phase

@st.cache(persist=True, show_spinner=False)
def fft_rescale(image, apix=1.0, cutoff_res=None, output_size=None):
    if cutoff_res:
        cutoff_res_y, cutoff_res_x = cutoff_res
    else:
        cutoff_res_y, cutoff_res_x = 2*apix, 2*apix
    if output_size:
        ony, onx = output_size
    else:
        ony, onx = image.shape
    freq_y = np.fft.fftfreq(ony) * 2*apix/cutoff_res_y
    freq_x = np.fft.fftfreq(onx) * 2*apix/cutoff_res_x
    Y, X = np.meshgrid(freq_y, freq_x, indexing='ij')
    Y = (2*np.pi * Y).flatten(order='C')
    X = (2*np.pi * X).flatten(order='C')

    from finufft import nufft2d2
    fft = nufft2d2(x=Y, y=X, f=image.astype(np.complex), eps=1e-6)
    fft = fft.reshape((ony, onx))

    # phase shifts for real-space shifts by half of the image box in both directions
    phase_shift = np.ones(fft.shape)
    phase_shift[1::2, :] *= -1
    phase_shift[:, 1::2] *= -1
    fft *= phase_shift
    # now fft has the same layout and phase origin (i.e. np.fft.ifft2(fft) would obtain original image)
    return fft

@st.cache(persist=True, show_spinner=False)
def low_high_pass_filter(data, low_pass_fraction=0, high_pass_fraction=0):
    fft = np.fft.fft2(data)
    ny, nx = fft.shape
    Y, X = np.meshgrid(np.arange(ny, dtype=np.float32)-ny//2, np.arange(nx, dtype=np.float32)-nx//2, indexing='ij')
    Y /= ny//2
    X /= nx//2
    if 0<low_pass_fraction<1:
        f2 = np.log(2)/(low_pass_fraction**2)
        filter_lp = np.exp(- f2 * (X**2+Y**2))
        fft *= np.fft.fftshift(filter_lp)
    if 0<high_pass_fraction<1:
        f2 = np.log(2)/(high_pass_fraction**2)
        filter_hp = 1.0 - np.exp(- f2 * (X**2+Y**2))
        fft *= np.fft.fftshift(filter_hp)
    ret = np.abs(np.fft.ifft2(fft))
    return ret

@st.cache(persist=True, show_spinner=False)
def generate_tapering_filter(image_size, fraction_start=[0, 0], fraction_slope=0.1):
    ny, nx = image_size
    fy, fx = fraction_start
    if not (0<fy<1 or 0<fx<1): return np.ones((ny, nx))
    Y, X = np.meshgrid(np.arange(0, ny, dtype=np.float32)-ny//2, np.arange(0, nx, dtype=np.float32)-nx//2, indexing='ij')
    filter = np.ones_like(Y)
    if 0<fy<1:
        Y = np.abs(Y / (ny//2))
        inner = Y<fy
        outer = Y>fy+fraction_slope
        Y = (Y-fy)/fraction_slope
        Y = (1. + np.cos(Y*np.pi))/2.0
        Y[inner]=1
        Y[outer]=0
        filter *= Y
    if 0<fx<1:
        X = np.abs(X / (nx//2))
        inner = X<fx
        outer = X>fx+fraction_slope
        X = (X-fx)/fraction_slope
        X = (1. + np.cos(X*np.pi))/2.0
        X[inner]=1
        X[outer]=0
        filter *= X
    return filter

@st.cache(persist=True, show_spinner=False)
def estimate_radial_range(data, thresh_ratio=0.1):
    proj_y = np.sum(data, axis=0)
    n = len(proj_y)
    radius = np.mean([n//2-np.argmax(proj_y[:n//2+1]), np.argmax(proj_y[n//2:])])
    background = np.mean(proj_y[[0,1,2,-3,-3,-1]])
    thresh = (proj_y.max() - background) * thresh_ratio + background
    indices = np.nonzero(proj_y>thresh)
    xmin = np.min(indices)
    xmax = np.max(indices)
    mask_radius = max(abs(n//2-xmin), abs(xmax-n//2))
    return float(radius), float(mask_radius)    # pixel

@st.cache(persist=True, show_spinner=False)
def auto_vertical_center(image):
    image_work = image * 1.0
    background = np.mean(image_work[[0,1,2,-3,-2,-1],[0,1,2,-3,-2,-1]])
    thresh = (image_work.max()-background) * 0.2 + background
    image_work = (image_work-thresh)/(image_work.max()-thresh)
    image_work[image_work<0] = 0

    # rough estimate of rotation
    def score_rotation(angle):
        tmp = rotate_shift_image(data=image_work, angle=angle)
        y_proj = tmp.sum(axis=0)
        percentiles = (100, 95, 90, 85, 80) # more robust than max alone
        y_values = np.percentile(y_proj, percentiles)
        err = -np.sum(y_values)
        return err
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(score_rotation, bounds=(-90, 90), method='bounded')
    angle = res.x

    # further refine rotation
    def score_rotation_shift(x):
        angle, dy, dx = x
        tmp1 = rotate_shift_image(data=image_work, angle=angle, pre_shift=(dy, dx))
        tmp2 = rotate_shift_image(data=image_work, angle=angle+180, pre_shift=(dy, dx))
        tmps = [tmp1, tmp2, tmp1[::-1,:], tmp2[::-1,:], tmp1[:,::-1], tmp2[:,::-1]]
        tmp_mean = np.zeros_like(image_work)
        for tmp in tmps: tmp_mean += tmp
        tmp_mean /= len(tmps)
        err = 0
        for tmp in tmps:
            err += np.sum(np.abs(tmp - tmp_mean))
        err /= len(tmps) * image_work.size
        return err
    from scipy.optimize import fmin
    res = fmin(score_rotation_shift, x0=(angle, 0, 0), xtol=1e-2)
    angle = res[0]  # dy, dx are not robust enough

    # refine dx 
    image_work = rotate_shift_image(data=image_work, angle=angle)
    y = np.sum(image_work, axis=0)
    y /= y.max()
    y[y<0.2] = 0
    n = len(y)
    mask = np.where(y>0.5)
    max_shift = abs((np.max(mask)-n//2) - (n//2-np.min(mask)))*1.5

    import scipy.interpolate as interpolate
    x = np.arange(3*n)
    f = interpolate.interp1d(x, np.tile(y, 3), kind='linear')    # avoid out-of-bound errors
    def score_shift(dx):
        x_tmp = x[n:2*n]-dx
        tmp = f(x_tmp)
        err = np.sum(np.abs(tmp-tmp[::-1]))
        return err
    res = minimize_scalar(score_shift, bounds=(-max_shift, max_shift), method='bounded')
    dx = res.x + (0.0 if n%2 else 0.5)
    return angle, dx

@st.cache(persist=True, show_spinner=False)
def rotate_shift_image(data, angle=0, pre_shift=(0, 0), post_shift=(0, 0), rotation_center=None, order=1):
    # pre_shift/rotation_center/post_shift: [y, x]
    if angle==0 and pre_shift==[0,0] and post_shift==[0,0]: return data*1.0
    ny, nx = data.shape
    if rotation_center is None:
        rotation_center = np.array((ny//2, nx//2), dtype=np.float32)
    ang = np.deg2rad(angle)
    m = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
    pre_dy, pre_dx = pre_shift    
    post_dy, post_dx = post_shift

    offset = -np.dot(m, np.array([post_dy, post_dx], dtype=np.float32).T) # post_rotation shift
    offset += np.array(rotation_center, dtype=np.float32).T - np.dot(m, np.array(rotation_center, dtype=np.float32).T)  # rotation around the specified center
    offset += -np.array([pre_dy, pre_dx], dtype=np.float32).T     # pre-rotation shift

    from scipy.ndimage import affine_transform
    ret = affine_transform(data, matrix=m, offset=offset, order=order, mode='constant')
    return ret

@st.cache(persist=True, show_spinner=False)
def generate_projection(data, az=0, tilt=0):
    from scipy.spatial.transform import Rotation as R
    from scipy.ndimage import affine_transform
    # note the convention change
    # xyz in scipy is zyx in cryoEM maps
    rot = R.from_euler('zx', [tilt, az], degrees=True)  # order: right to left
    m = rot.as_matrix()
    nx, ny, nz = data.shape
    bcenter = np.array((nx//2, ny//2, nz//2), dtype=np.float)
    offset = bcenter.T - np.dot(m, bcenter.T)
    tmp = affine_transform(data, matrix=m, offset=offset, mode='nearest')
    ret = tmp.sum(axis=1)   # integrate along y-axis
    ret = normalize(ret)
    return ret

@st.cache(persist=True, show_spinner=False)
def normalize(data, percentile=(0, 100)):
    p0, p1 = percentile
    vmin, vmax = sorted(np.percentile(data, (p0, p1)))
    data2 = (data-vmin)/(vmax-vmin)
    return data2

@st.cache(persist=True, show_spinner=False)
def nonzero_images(data):
    assert(len(data.shape) == 3)
    nonzeros = np.any(data, axis=(1,2))
    n = np.count_nonzero(nonzeros)
    if n>0: 
        return np.nonzero(nonzeros)[0]
    else:
        None

@st.cache(persist=True, show_spinner=False)
def guess_if_is_power_spectra(data, thresh=15):
    median = np.median(data)
    max = np.max(data)
    sigma = np.std(data)
    if (max-median)>thresh*sigma: return True
    else: return False

@st.cache(persist=True, show_spinner=False)
def get_2d_image_from_uploaded_file(fileobj):
    import os, tempfile
    orignal_filename = fileobj.name
    suffix = os.path.splitext(orignal_filename)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(fileobj.read())
        data, apix = get_2d_image_from_file(temp.name)
    return data, apix

@st.cache(persist=True, show_spinner=False)
def get_emdb_map(emdid):
    emdid_number = emdid.lower().split("emd-")[-1]
    url = f"ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdid_number}/map/emd_{emdid_number}.map.gz"
    ds = np.DataSource(None)
    fp = ds.open(url)
    import mrcfile
    with mrcfile.open(fp.name) as mrc:
        vmin, vmax = np.min(mrc.data), np.max(mrc.data)
        data = ((mrc.data - vmin) / (vmax - vmin)).astype(np.float32)
        apix = mrc.voxel_size.x.item()
    return data, apix

@st.cache(persist=True, show_spinner=False)
def get_2d_image_from_url(url):
    ds = np.DataSource(None)
    fp=ds.open(url)
    return get_2d_image_from_file(fp.name)

@st.cache(persist=True, show_spinner=False)
def get_2d_image_from_file(filename):
    import mrcfile
    with mrcfile.open(filename) as mrc:
        data = mrc.data * 1.0
        apix = mrc.voxel_size.x.item()
    if data.dtype==np.dtype('complex64'):
        data_complex = data
        ny, nx = data_complex[0].shape
        data = np.zeros((len(data_complex), ny, (nx-1)*2), dtype=np.float)
        for i in range(len(data)):
            tmp = np.abs(np.fft.fftshift(np.fft.fft(np.fft.irfft(data_complex[i])), axes=1))
            data[i] = normalize(tmp, percentile=(0.1, 99.9))
    return data, apix

class Data(object):
    def __init__(self, twist, rise, csym, diameter, apix, url=None):
        self.twist = twist
        self.rise = rise
        self.pitch = (360./twist)*rise
        self.csym = csym
        self.diameter = diameter
        self.apix = apix
        self.url = url

data_examples = [
    Data(twist=29.40, rise=21.92, csym=6, diameter=138, apix=2.3438, url="https://tinyurl.com/y5tq9fqa")
]
data_example = data_examples[0]

@st.cache(suppress_st_warning=True)
def show_initial_message(message=""):
    if message: st.warning(body=message)

@st.cache(persist=True, show_spinner=False)
def setup_anonymous_usage_tracking():
    try:
        import os, stat
        index_file = os.path.dirname(st.__file__) + "/static/index.html"
        os.chmod(index_file, stat.S_IRUSR|stat.S_IWUSR|stat.S_IROTH)
        with open(index_file, "r+") as fp:
            txt = fp.read()
            if txt.find("gtag/js?")==-1:
                txt2 = txt.replace("<head>", '''<head><script async src="https://www.googletagmanager.com/gtag/js?id=G-8Z99BDVHTC"></script><script>window.dataLayer = window.dataLayer || [];function gtag(){dataLayer.push(arguments);}gtag('js', new Date());gtag('config', 'G-8Z99BDVHTC');</script>''')
                fp.seek(0)
                fp.write(txt2)
                fp.truncate()
    except:
        pass

if __name__ == "__main__":
    setup_anonymous_usage_tracking()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mode", metavar="<n>", choices=(0,1,2), type=int, help="input mode (0, 1, 2). default: %(default)s", default=1)
    parser.add_argument("--input_is_power", metavar="<0|1>", choices=(0,1), type=int, help="if the input is power spectra. default: %(default)s", default=0)
    parser.add_argument("--message", metavar="<str>", type=str, help="initial message. default: %(default)s", default="")
    args = parser.parse_args()
    main(args)
