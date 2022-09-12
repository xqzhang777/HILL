""" 
MIT License

Copyright (c) 2020-2022 Wen Jiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def import_with_auto_install(packages, scope=locals()):
    if isinstance(packages, str): packages=[packages]
    for package in packages:
        if package.find(":")!=-1:
            package_import_name, package_pip_name = package.split(":")
        else:
            package_import_name, package_pip_name = package, package
        try:
            scope[package_import_name] = __import__(package_import_name)
        except ImportError:
            import subprocess
            subprocess.call(f'pip install {package_pip_name}', shell=True)
            scope[package_import_name] =  __import__(package_import_name)
required_packages = "streamlit numpy scipy bokeh skimage:scikit_image mrcfile finufft xmltodict".split()
import_with_auto_install(required_packages)

import streamlit as st
import numpy as np
from bokeh.plotting import figure
from bokeh.models import CustomJS, Span, LinearColorMapper
from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool
from bokeh.events import MouseMove, DoubleTap
from bokeh.layouts import gridplot
import math, random, gc
gc.enable()

#from memory_profiler import profile
#@profile(precision=4)
def main(args):
    title = "HILL: Helical Indexing using Layer Lines"
    st.set_page_config(page_title=title, layout="wide")

    hosted, host = is_hosted(return_host=True)
    if hosted and host in ['heroku']:
        st.error(f"This app hosted on Heroku will be unavailable starting November 28, 2022 [when Heroku discontinues free hosting service](https://blog.heroku.com/next-chapter). Please switch to [the same app hosted elsewhere](https://jianglab-hill-streamlit-app-gk7bkc.streamlitapp.com)")

    st.title(title)

    st.web.server.server_util.MESSAGE_SIZE_LIMIT = 2e8  # default is 5e7 (50MB)
    st.elements.utils._shown_default_value_warning = True

    if len(st.session_state)<1:  # only run once at the start of the session
        set_initial_query_params(query_string=args.query_string) # only excuted on the first run

    if len(st.session_state)<1:  # only run once at the start of the session
        set_session_state_from_query_params()

    if "input_mode_0" not in st.session_state:
        set_session_state_from_data_example()
    
    col1, col2, col3, col4 = st.columns((1., 0.6, 0.4, 4.0))

    with col1:
        with st.expander(label="README", expanded=False):
            st.write("This Web app considers a biological helical structure as the product of a continous helix and a set of parallel planes, and based on the covolution theory, the Fourier Transform (FT) of a helical structure would be the convolution of the FT of the continous helix and the FT of the planes.  \nThe FT of a continous helix consists of equally spaced layer planes (3D) or layerlines (2D projection) that can be described by Bessel functions of increasing orders (0, +/-1, +/-2, ...) from the Fourier origin (i.e. equator). The spacing between the layer planes/lines is determined by the helical pitch (i.e. the shift along the helical axis for a 360 ° turn of the helix). If the structure has additional cyclic symmetry (for example, C6) around the helical axis, only the layer plane/line orders of integer multiplier of the symmetry (e.g. 0, +/-6, +/-12, ...) are visible. The primary peaks of the layer lines in the power spectra form a pattern similar to a X symbol.  \nThe FT of the parallel planes consists of equally spaced points along the helical axis (i.e. meridian) with the spacing being determined by the helical rise.  \nThe convolution of these two components (X-shaped pattern of layer lines and points along the meridian) generates the layer line patterns seen in the power spectra of the projection images of helical structures. The helical indexing task is thus to identify the helical rise, pitch (or twist), and cyclic symmetry that would predict a layer line pattern to explain the observed the layer lines in the power spectra. This Web app allows you to interactively change the helical parameters and superimpose the predicted layer liines on the power spectra to complete the helical indexing task.  \n  \nPS: power spectra; PD: phase difference between the two sides of meridian; YP: Y-axis power spectra profile; LL: layer lines; m: indices of the X-patterns along the meridian; Jn: Bessel order")

        data_all, image_index, data, apix, radius_auto, mask_radius, input_type, is_3d, input_params, (image_container, image_label) = obtain_input_image(col1, param_i=0)
        input_mode, (uploaded_filename, url, emd_id) = input_params

        if input_type in ["image"]:
            label = f"Replace amplitudes or phases with another image"
        elif input_type in ["PS"]:
            label = f"Load phases from another image"
        elif input_type in ["PD"]:
            label = f"Load amplitudes from another image"
        input_image2 = st.checkbox(label=label, value=False)        
        if input_image2:
            _, image_index2, data2, apix2, radius_auto2, mask_radius2, input_type2, is_3d2, input_params2, _ = obtain_input_image(col1, param_i=1, image_index_sync=image_index+1)
            input_mode2, (uploaded_filename2, url2, emd_id2) = input_params2
        else:
            image_index2, data2, apix2, radius_auto2, mask_radius2, input_type2, is_3d2 = [None] * 7
            input_mode2, (uploaded_filename2, url2, emd_id2) = None, (None, None, None)

        with st.expander(label="Server info", expanded=False):
            server_info_empty = st.empty()
            #server_info = f"Host: {get_hostname()}  \n"
            #server_info+= f"Account: {get_username()}"
            server_info = f"Uptime: {uptime():.1f} s  \n"
            server_info+= f"Mem (total): {mem_info()[0]:.1f} MB  \n"
            server_info+= f"Mem (quota): {mem_quota():.1f} MB  \n"
            server_info+= "Mem (used): {mem_used:.1f} MB"
            server_info_empty.markdown(server_info.format(mem_used=mem_used()))

        hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    with col2:
        query_params = st.experimental_get_query_params()
        if "pitch" in query_params and "rise" in query_params:
            st.session_state.rise = float(query_params["rise"][0])
            st.session_state.twist = pitch2twist(float(query_params["pitch"][0]), st.session_state.rise)
            query_params.pop('rise', None)
            query_params.pop('pitch', None)
            st.experimental_set_query_params(**query_params)
        
        reload = st.button("Copy pitch/rise↶")

        pitch_or_twist_choices = ["pitch", "twist"]
        pitch_or_twist = st.radio(label="", options=pitch_or_twist_choices, index=0, horizontal=True)
        use_pitch = 1 if pitch_or_twist=="pitch" else 0

        pitch_or_twist_number_input = st.empty()
        pitch_or_twist_text = st.empty()
        rise_empty = st.empty()

        ny, nx = data.shape
        max_rise = round(max(2000., max(ny, nx)*apix * 2.0), 2)
        min_rise = round(apix/2.0, 2)
        rise = rise_empty.number_input('Rise (Å)', min_value=min_rise, max_value=max_rise, step=1.0, format="%.3f", key="rise")

        if use_pitch:
            min_pitch = abs(rise)
            value = max(min_pitch, twist2pitch(st.session_state.twist, rise))
            pitch = pitch_or_twist_number_input.number_input('Pitch (Å)', value=value, min_value=min_pitch, step=1.0, format="%.2f", help="twist = 360 / (pitch/rise)")
            st.session_state.twist = pitch2twist(pitch, rise)
            pitch_or_twist_text.markdown(f"*(twist = {st.session_state.twist:.2f} °)*")
            twist = pitch2twist(pitch, rise)
        else:
            twist = pitch_or_twist_number_input.number_input('Twist (°)', min_value=0.0, max_value=180.0, step=1.0, format="%.2f", help="pitch = 360/twist * rise", key="twist")
            pitch = abs(round(twist2pitch(twist, rise), 2))
            pitch_or_twist_text.markdown(f"*(pitch = {pitch:.2f} Å)*")

        csym = st.number_input('Csym', min_value=1, step=1, help="Cyclic symmetry around the helical axis", key="csym")

        if input_image2: value = max(radius_auto*apix, radius_auto2*apix2)
        else: value = radius_auto*apix
        value = max(min(500.0, value), 1.0)
        helical_radius = 0.5*st.number_input('Filament/tube diameter (Å)', value=value*2, min_value=1.0, max_value=1000.0, step=10., format="%.1f", help="Mean radius of the tube/filament density from the helical axis", key="diameter")
        
        tilt = st.number_input('Out-of-plane tilt (°)', value=0.0, min_value=-90.0, max_value=90.0, step=1.0, help="Only used to compute the layerline positions and to simulate the helix. Will not change the power spectra and phase differences across meridian of the input image(s)", key="tilt")
        cutoff_res_x = st.number_input('Resolution limit - X (Å)', value=3*apix, min_value=2*apix, step=1.0, help="Set the highest resolution to be displayed in the X-direction", key="cutoff_res_x")
        cutoff_res_y = st.number_input('Resolution limit - Y (Å)', value=2*apix, min_value=2.*apix, step=1.0, help="Set the highest resolution to be displayed in the Y-direction", key="cutoff_res_y")
        with st.expander(label="Filters", expanded=False):
            log_xform = st.checkbox(label="Log(amplitude)", value=True, help="Perform log transform of the power spectra to allow clear display of layerlines at low and high resolutions")
            hp_fraction = st.number_input('Fourier high-pass (%)', value=0.4, min_value=0.0, max_value=100.0, step=0.1, format="%.2f", help="Perform high-pass Fourier filtering of the power spectra with filter=0.5 at this percentage of the Nyquist resolution") / 100.0
            lp_fraction = st.number_input('Fourier low-pass (%)', value=0.0, min_value=0.0, max_value=100.0, step=10.0, format="%.2f", help="Perform low-pass Fourier filtering of the power spectra with filter=0.5 at this percentage of the Nyquist resolution") / 100.0
            pnx = int(st.number_input('FFT X-dim size (pixels)', value=512, min_value=min(nx, 128), step=2, help="Set the size of FFT in X-dimension to this number of pixels", key="pnx"))
            pny = int(st.number_input('FFT Y-dim size (pixels)', value=1024, min_value=min(ny, 512), step=2, help="Set the size of FFT in Y-dimension to this number of pixels", key="pny"))
        with st.expander(label="Simulation", expanded=False):
            ball_radius = st.number_input('Gaussian radius (Å)', value=0.0, min_value=0.0, max_value=helical_radius, step=5.0, format="%.1f", help="A 3-D Gaussian function will be used to reprsent each subunit in the simulated helix. The Gaussian function will fall off from 1 to 0.5 at this radius. A value <=0 will disable the simulation", key="ball_radius")
            show_simu = True if ball_radius > 0 else False
            noise=0.0
            use_plot_size=False
            if show_simu:
                az = st.number_input('Azimuthal angle (°)', value=0.0, min_value=0.0, max_value=360.0, step=1.0, format="%.2f", help="Position the Gaussian in the central Z-section at this azimuthal angle", key="simuaz")
                noise = st.number_input('Noise (sigma)', value=0.001, min_value=0., step=1., format="%.2f", help="Add random noise to the simulated helix image", key="simunoise")
                use_plot_size = st.checkbox('Use plot size', value=False, help="If checked, the simulated helix image will use the image size of the displayed power spectra instead of the size of the input image", key="useplotsize")
        
        movie_frames = 0
        if is_3d or show_simu:
            with st.expander(label="Tilt movie", expanded=False):
                movie_frames = st.number_input('Movie frame #', value=0, min_value=0, max_value=1000, step=1, help="Set the number of movies frames that will be generated to show the morphing of the power spectra and phase differences across meridian images of the simulated helix at different tilt angles in the range from 0 to the value specified in the *Out-of-plane tilt (°)* input box. A value <=0 will not generate a movie")
                if movie_frames>0:
                    if is_3d and show_simu:
                        movie_modes = {0:"3D map", 1:"simulation"}
                        movie_mode = st.radio(label="Tilt:", options=list(movie_modes.keys()), format_func=lambda i:movie_modes[i], index=0, help="Tilt the input 3D map to different angles instead of simulating a helix at different tilt angles", horizontal=True)
                    elif is_3d:
                        movie_mode = 0
                    else:
                        movie_mode = 1
                    if movie_mode == 0:
                        movie_noise = st.number_input('Noise (sigma)', value=0., min_value=0., step=1., format="%.2f", help="Add random noise to the projection images")
        
        share_url = st.checkbox('Show sharable URL', value=False, help="Include relevant parameters in the browser URL to allow you to share the URL and reproduce the plots", key="share_url")
        if share_url:
            show_qr = st.checkbox('Show QR code of the URL', value=False, help="Display the QR code of the sharable URL", key="show_qr")
        else:
            show_qr = False

    if input_type in ["PS"]:
        pwr = resize_rescale_power_spectra(data, nyquist_res=2*apix, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                output_size=(pny, pnx), log=log_xform, low_pass_fraction=lp_fraction, high_pass_fraction=hp_fraction, norm=1)
        phase = None
        phase_diff = None
    elif input_type in ["PD"]:
        pwr = None
        phase = None
        phase_diff = resize_rescale_power_spectra(data, nyquist_res=2*apix, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                output_size=(pny, pnx), log=0, low_pass_fraction=0, high_pass_fraction=0, norm=0)
    else:
        pwr, phase = compute_power_spectra(data, apix=apix, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                output_size=(pny, pnx), log=log_xform, low_pass_fraction=lp_fraction, high_pass_fraction=hp_fraction)
        phase_diff = compute_phase_difference_across_meridian(phase)                
    
    if input_image2:
        if input_type2 in ["PS"]:
            pwr2 = resize_rescale_power_spectra(data2, nyquist_res=2*apix2, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                    output_size=(pny, pnx), log=log_xform, low_pass_fraction=lp_fraction, high_pass_fraction=hp_fraction, norm=1)
            phase2 = None
            phase_diff2 = None
        elif input_type2 in ["PD"]:
            pwr2 = None
            phase2 = None
            phase_diff2 = resize_rescale_power_spectra(data2, nyquist_res=2*apix2, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                output_size=(pny, pnx), log=0, low_pass_fraction=0, high_pass_fraction=0, norm=0)
        else:
            pwr2, phase2 = compute_power_spectra(data2, apix=apix2, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                    output_size=(pny, pnx), log=log_xform, low_pass_fraction=lp_fraction, high_pass_fraction=hp_fraction)
            phase_diff2 = compute_phase_difference_across_meridian(phase2)                
    else:
        pwr2 = None
        phase2 = None
        phase_diff2 = None

    with col3:
        st.subheader("Display:")
        show_pwr = False
        show_phase = False
        show_phase_diff = False
        show_yprofile = False
        if input_type in ["image", "PS"]:
            show_pwr = st.checkbox(label="PS", value=True, help="Show the power spectra", key="show_pwr")
        if show_pwr:
            show_yprofile = st.checkbox(label="YP", value=True, help="Show the Y-profile of the power spectra (i.e. horizontal projection of the power spectra", key="show_yprofile")
        if input_type in ["image"]:
            show_phase = st.checkbox(label="Phase", value=False, help="Show the phase values in the hover tooltips of the displayed power spectra and phase differences across meridian")
        if input_type in ["image", "PD"]:
            show_phase_diff = st.checkbox(label="PD", value=True, help="Show the phase differences across meridian", key="show_phase_diff")
        
        show_pwr2 = False
        show_phase2 = False
        show_phase_diff2 = False
        show_yprofile2 = False
        if input_image2:
            if input_type2 in ["image", "PS"]:
                if input_type2 in ["PS"]: value = True
                else: value = not show_pwr
                show_pwr2 = st.checkbox(label="PS2", value=value, help="Show the power spectra of the 2nd input image")
            if show_pwr2:
                show_yprofile2 = st.checkbox(label="YP2", value=show_yprofile, help="Show the Y-profile of the power spectra of the 2nd input image (i.e. horizontal projection of the power spectra")
            if input_type2 in ["image"]:
                if show_pwr2: show_phase2 = st.checkbox(label="Phase2", value=False)
            if input_type2 in ["image", "PD"]:
                if input_type2 in ["PD"]: value = True
                else: value = not show_phase_diff
                show_phase_diff2 = st.checkbox(label="PD2", value=value, help="Show the phase differences across meridian of the 2nd input image")

        show_pwr_simu = False
        show_phase_simu = False
        show_phase_diff_simu = False
        show_yprofile_simu = False
        if show_simu:
            show_pwr_simu = st.checkbox(label="PSSimu", value=show_pwr or show_pwr2)
            if show_pwr_simu:
                show_yprofile_simu = st.checkbox(label="YPSimu", value=show_yprofile or show_yprofile2)
                show_phase_simu = st.checkbox(label="PhaseSimu", value=show_phase or show_phase2)
            show_phase_diff_simu = st.checkbox(label="PDSimu", value=show_phase_diff or show_phase_diff2)

        show_LL_text = False
        if show_pwr or show_phase_diff or show_pwr2 or show_phase_diff2 or show_pwr_simu or show_phase_diff_simu:
            show_pseudocolor = st.checkbox(label="Color", value=True, help="Show the power spectra in pseudo color instead of grey scale")
            show_LL = st.checkbox(label="LL", value=True, help="Show the layer lines at positions computed from the current values of pitch/twist, rise, csym, radius, and tilt", key="show_LL")
            if show_LL:
                show_LL_text = st.checkbox(label="LLText", value=True, help="Show the layer lines using integer numbers for the Bessel orders instead of ellipses", key="show_LL_text")

                st.subheader("m:")
                m_max_auto = int(np.floor(np.abs(rise/cutoff_res_y)))+3
                m_max = int(st.number_input(label=f"Max=", min_value=1, value=m_max_auto, step=1, help="Maximal number of layer line groups to show"))
                m_groups = compute_layer_line_positions(twist=twist, rise=rise, csym=csym, radius=helical_radius, tilt=tilt, cutoff_res=cutoff_res_y, m_max=m_max)
                ng = len(m_groups)
                show_choices = {}
                lgs = sorted(m_groups.keys())[::-1]
                for lgi, lg in enumerate(lgs):
                    value = True if lg in [0, 1] else False
                    show_choices[lg] = st.checkbox(label=str(lg), value=value, help=f"Show the layer lines in group m={lg}")

    if show_simu:
        proj = simulate_helix(twist, rise, csym, helical_radius=helical_radius, ball_radius=ball_radius, 
                ny=data.shape[0], nx=data.shape[1], apix=apix, tilt=tilt, az0=az)
        if noise>0:
            sigma = np.std(proj[np.nonzero(proj)])
            proj = proj + np.random.normal(loc=0.0, scale=noise*sigma, size=proj.shape)
        fraction_x = mask_radius/(proj.shape[1]//2*apix)
        tapering_image = generate_tapering_filter(image_size=proj.shape, fraction_start=[0.8, fraction_x], fraction_slope=0.1)
        proj = proj * tapering_image
        with image_container:
            st.image([normalize(data), normalize(proj)], use_column_width=True, caption=[image_label, "Simulated"])

    with col4:
        if not (show_pwr or show_phase_diff or show_pwr2 or show_phase_diff2 or show_pwr_simu or show_phase_diff_simu): return

        items = [ (show_pwr, pwr, "Power Spectra", show_phase, phase, show_phase_diff, phase_diff, "Phase Diff Across Meridian", show_yprofile), 
                  (show_pwr2, pwr2, "Power Spectra - 2", show_phase2, phase2, show_phase_diff2, phase_diff2, "Phase Diff Across Meridian - 2", show_yprofile2)
                ]
        if show_pwr_simu or show_phase_diff_simu:
            if use_plot_size:
                apix_simu = min(cutoff_res_y, cutoff_res_x)/2
                proj = simulate_helix(twist, rise, csym, helical_radius=helical_radius, ball_radius=ball_radius, 
                        ny=pny, nx=pnx, apix=apix_simu, tilt=tilt, az0=az)
                if noise>0:
                    sigma = np.std(proj[np.nonzero(proj)])
                    proj = proj + np.random.normal(loc=0.0, scale=noise*sigma, size=proj.shape)
                fraction_x = mask_radius/(proj.shape[1]//2*apix_simu)
                tapering_image = generate_tapering_filter(image_size=proj.shape, fraction_start=[0.8, fraction_x], fraction_slope=0.1)
                proj = proj * tapering_image
            else:
                apix_simu = apix
            proj_pwr, proj_phase = compute_power_spectra(proj, apix=apix_simu, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                    output_size=(pny, pnx), log=log_xform, low_pass_fraction=lp_fraction, high_pass_fraction=hp_fraction)
            proj_phase_diff = compute_phase_difference_across_meridian(proj_phase)                
            items += [(show_pwr_simu, proj_pwr, "Simulated Power Spectra", show_phase_simu, proj_phase, show_phase_diff_simu, proj_phase_diff, "Simulated Phase Diff Across Meridian", show_yprofile_simu)]

        figs = []
        figs_image = []
        for item in items:
            show_pwr_work, pwr_work, title_pwr_work, show_phase_work, phase_work, show_phase_diff_work, phase_diff_work, title_phase_work, show_yprofile_work = item
            if show_pwr_work:
                tooltips = [("Res r", "Å"), ('Res y', 'Å'), ('Res x', 'Å'), ('Jn', '@bessel'), ('Amp', '@image')]
                fig = create_layerline_image_figure(pwr_work, cutoff_res_x, cutoff_res_y, helical_radius, tilt, phase=phase_work if show_phase_work else None, pseudocolor=show_pseudocolor, title=title_pwr_work, yaxis_visible=False, tooltips=tooltips)
                figs.append(fig)
                figs_image.append(fig)

            if show_yprofile_work:
                ny, nx = pwr_work.shape
                dsy = 1/(ny//2*cutoff_res_y)
                y=np.arange(-ny//2, ny//2)*dsy
                yinv = y*1.0
                yinv[yinv==0] = 1e-10
                yinv = 1/np.abs(yinv)
                yprofile = np.mean(pwr_work, axis=1)
                yprofile /= yprofile.max()
                source_data = ColumnDataSource(data=dict(yprofile=yprofile, y=y, resy=yinv))
                tools = 'box_zoom,hover,pan,reset,save,wheel_zoom'
                tooltips = [('Res y', '@resyÅ'), ('Amp', '$x')]
                fig = figure(frame_width=nx//2, frame_height=figs[-1].frame_height, y_range=figs[-1].y_range, y_axis_location = "right", 
                    title=None, tools=tools, tooltips=tooltips)
                fig.line(source=source_data, x='yprofile', y='y', line_width=2, color='blue')
                fig.yaxis.visible = False
                fig.hover[0].attachment="vertical"
                figs.append(fig)

            if show_phase_diff_work:
                tooltips = [("Res r", "Å"), ('Res y', 'Å'), ('Res x', 'Å'), ('Jn', '@bessel'), ('Phase Diff', '@image °')]
                fig = create_layerline_image_figure(phase_diff_work, cutoff_res_x, cutoff_res_y, helical_radius, tilt, phase=phase_work if show_phase_work else None, pseudocolor=show_pseudocolor, title=title_phase_work, yaxis_visible=False, tooltips=tooltips)
                figs.append(fig)
                figs_image.append(fig)
                
        figs[-1].yaxis.fixed_location = figs[-1].x_range.end
        figs[-1].yaxis.visible = True
        add_linked_crosshair_tool(figs, dimensions="width")
        add_linked_crosshair_tool(figs_image, dimensions="both")

        fig_ellipses = []
        if figs_image and show_LL:
            if max(m_groups[0]["LL"][0])>0:
                color = 'white' if show_pseudocolor else 'red'
                ll_line_dashes = 'solid dashed dotted dotdash dashdot'.split()             
                x, y, n = m_groups[0]["LL"]
                tmp_x = np.sort(np.unique(x))
                width = np.mean(tmp_x[1:]-tmp_x[:-1])
                tmp_y = np.sort(np.unique(y))
                height = np.mean(tmp_y[1:]-tmp_y[:-1])/3
                for mi, m in enumerate(m_groups.keys()):
                    if not show_choices[m]: continue
                    x, y, bessel_order = m_groups[m]["LL"]
                    if show_LL_text:
                        texts = [str(int(n)) for n in bessel_order]
                    tags = [m, bessel_order]
                    line_dash = ll_line_dashes[abs(m)%len(ll_line_dashes)]
                    for f in figs_image:
                        if show_LL_text: 
                            text_labels = f.text(x, y, y_offset=2, text=texts, text_color="white", text_baseline="middle", text_align="center")
                            text_labels.tags = tags
                            fig_ellipses.append(text_labels)
                        else:
                            ellipses = f.ellipse(x, y, width=width, height=height, line_width=2, line_color=color, line_dash=line_dash, fill_alpha=0)
                            ellipses.tags = tags
                            fig_ellipses.append(ellipses)
            else:
                st.warning(f"No off-equator layer lines to draw for Pitch={pitch:.2f} Csym={csym} combinations. Consider increasing Pitch or reducing Csym")

        from bokeh.models import CustomJS
        from bokeh.events import MouseEnter
        title_js = CustomJS(args=dict(title=title), code="""
            document.title=title
        """)
        figs[0].js_on_event(MouseEnter, title_js)

        if fig_ellipses:
            from bokeh.models import Slider, CustomJS
            slider_pitch = Slider(start=pitch/2, end=pitch*2.0, value=pitch, step=pitch*0.002, title="Pitch (Å)", width=pnx)
            slider_rise = Slider(start=rise/2, end=min(max_rise, rise*2.0), value=rise, step=min(max_rise, rise*2.0)*0.001, title="Rise (Å)", width=pnx)
            callback_code = """
                var pitch_inv = 1./slider_pitch.value
                var rise_inv = 1./slider_rise.value
                for (var fi = 0; fi < fig_ellipses.length; fi++) {
                    var ellipses = fig_ellipses[fi]
                    const m = ellipses.tags[0]
                    const ns = ellipses.tags[1]
                    var y = ellipses.data_source.data.y
                    for (var i = 0; i < ns.length; i++) {
                        const n = ns[i]
                        y[i] = m * rise_inv + n * pitch_inv
                    }
                    ellipses.data_source.change.emit()
                }
            """
            callback = CustomJS(args=dict(fig_ellipses=fig_ellipses, slider_pitch=slider_pitch, slider_rise=slider_rise), code=callback_code)
            slider_pitch.js_on_change('value', callback)
            slider_rise.js_on_change('value', callback)

            callback_code = """
                let url = new URL(document.location)
                let params = url.searchParams
                params.set("pitch", Math.round(slider_pitch.value*100.)/100.)
                params.set("rise", Math.round(slider_rise.value*100.)/100.)
                //document.location = url.href
                history.replaceState({}, document.title, url.href)
                if (reload) {
                    var class_names = ["css-1cpxqw2 edgvbvh9"]
                    console.log(class_names)
                    var i
                    for (i=0; i<class_names.length; i++) {
                        console.log(i, class_names[i])
                        let reload_buttons = document.getElementsByClassName(class_names[i])
                        console.log(reload_buttons)
                        if (reload_buttons.length>0) {
                            reload_buttons[reload_buttons.length-1].click()
                            break
                        }
                    }
                }
            """
            reload = input_mode in [1, 2, 3] and input_mode2 in [None, 1, 2, 3]
            callback = CustomJS(args=dict(slider_pitch=slider_pitch, slider_rise=slider_rise, reload=reload), code=callback_code)
            slider_pitch.js_on_change('value_throttled', callback)
            slider_rise.js_on_change('value_throttled', callback)

            if len(figs)==1:
                from bokeh.layouts import column
                figs[0].toolbar_location="right"
                figs_grid = column(children=[slider_pitch, slider_rise, figs[0]])
                override_height = pny+180
            else:
                from bokeh.layouts import layout
                figs_row = gridplot(children=[figs], toolbar_location='right')
                figs_grid = layout(children=[[slider_pitch, slider_rise], figs_row])
                override_height = pny+120
        else:
            figs_grid = gridplot(children=[figs], toolbar_location='right')
            override_height = pny+120

        st.bokeh_chart(figs_grid, use_container_width=False)

        if movie_frames>0:
            with st.spinner(text="Generating movie of tilted power spectra/phases ..."):
                if movie_mode==0:
                    params = (movie_mode, data_all, movie_noise, apix)
                if movie_mode==1:
                    if use_plot_size:
                        ny, nx = pny, pnx
                        apix_simu = min(cutoff_res_y, cutoff_res_x)/2
                    else:
                        ny, nx = data.shape
                        apix_simu = apix
                    params = (movie_mode, twist, rise, csym, noise, helical_radius, ball_radius, az, ny, nx, apix_simu)
                movie_filename = create_movie(movie_frames, tilt, params, pny, pnx, mask_radius, cutoff_res_x, cutoff_res_y, show_pseudocolor, log_xform, lp_fraction, hp_fraction)
                st.video(movie_filename) # it always show the video using the entire column width

        del data_all, data, figs_grid

        if share_url:
            set_query_params_from_session_state()
            if show_qr:
                qr_image = qr_code()
                st.image(qr_image)
        else:
            st.experimental_set_query_params()

        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu). Report problems to Wen Jiang (jiang12 at purdue.edu)*")


@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False, suppress_st_warning=True)
def create_movie(movie_frames, tilt_max, movie_mode_params, pny, pnx, mask_radius, cutoff_res_x, cutoff_res_y, show_pseudocolor, log_xform, lp_fraction, hp_fraction):
    if movie_mode_params[0] == 0:
        movie_mode, data_all, noise, apix = movie_mode_params
        nz, ny, nx = data_all.shape
        helical_radius = 0
    else:
        movie_mode, twist, rise, csym, noise, helical_radius, ball_radius, az, ny, nx, apix =movie_mode_params
    tilt_step = tilt_max/movie_frames
    fraction_x = mask_radius/(nx//2*apix)
    tapering_image = generate_tapering_filter(image_size=(ny, nx), fraction_start=[0.8, fraction_x], fraction_slope=0.1)
    image_filenames = []
    from bokeh.io import export_png
    progress_bar = st.empty()
    progress_bar.progress(0.0)
    for i in range(movie_frames+1):
        tilt = tilt_step * i
        if movie_mode==0:
            proj = generate_projection(data_all, az=0, tilt=tilt, output_size=(ny, nx))
        else:
            proj = simulate_helix(twist, rise, csym, helical_radius=helical_radius, ball_radius=ball_radius, 
                ny=ny, nx=nx, apix=apix, tilt=tilt, az0=az)
        if noise>0:
            sigma = np.std(proj[np.nonzero(proj)])
            proj = proj + np.random.normal(loc=0.0, scale=noise*sigma, size=proj.shape)
        proj = proj * tapering_image

        figs = []
        title = f"Projection"
        fig_proj = create_layerline_image_figure(proj, cutoff_res_x, cutoff_res_y, helical_radius, tilt, phase=None, pseudocolor=show_pseudocolor, title=title, yaxis_visible=False, tooltips=None)
        figs.append(fig_proj)

        proj_pwr, proj_phase = compute_power_spectra(proj, apix=apix, cutoff_res=(cutoff_res_y, cutoff_res_x), 
            output_size=(pny, pnx), log=log_xform, low_pass_fraction=lp_fraction, high_pass_fraction=hp_fraction)
        title = f"Power Spectra"
        fig_pwr = create_layerline_image_figure(proj_pwr, cutoff_res_x, cutoff_res_y, helical_radius, tilt, phase=None, pseudocolor=show_pseudocolor, title=title, yaxis_visible=False, tooltips=None)
        from bokeh.models import Label
        label = Label(x=0., y=0.9/cutoff_res_y, text=f"tilt = {tilt:.2f}°", text_align='center', text_color='white', text_font_size='30px', visible=True)
        fig_pwr.add_layout(label)
        figs.append(fig_pwr)

        phase_diff = compute_phase_difference_across_meridian(proj_phase)
        title = f"Phase Diff Across Meridian"
        fig_phase = create_layerline_image_figure(phase_diff, cutoff_res_x, cutoff_res_y, helical_radius, tilt, phase=None, pseudocolor=show_pseudocolor, title=title, yaxis_visible=False, tooltips=None)
        figs.append(fig_phase)

        fig_all = gridplot(children=[figs], toolbar_location=None)
        filename = f"image_{i:05d}.png"
        image_filenames.append(filename)
        export_png(fig_all, filename=filename)
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

def create_image_figure(image, dx, dy, title="", title_location="below", plot_width=None, plot_height=None, x_axis_label='x', y_axis_label='y', tooltips=None, show_axis=True, show_toolbar=True, crosshair_color="white", aspect_ratio=None):
    from bokeh.plotting import figure
    h, w = image.shape
    if aspect_ratio is None and (plot_width and plot_height):
        aspect_ratio = plot_width/plot_height
    tools = 'box_zoom,crosshair,pan,reset,save,wheel_zoom'
    fig = figure(title_location=title_location, 
        frame_width=plot_width, frame_height=plot_height, 
        x_axis_label=x_axis_label, y_axis_label=y_axis_label,
        x_range=(-w//2*dx, (w//2-1)*dx), y_range=(-h//2*dy, (h//2-1)*dy), 
        tools=tools, aspect_ratio=aspect_ratio)
    fig.grid.visible = False
    if title:
        fig.title.text=title
        fig.title.align = "center"
        fig.title.text_font_size = "18px"
        fig.title.text_font_style = "normal"
    if not show_axis: fig.axis.visible = False
    if not show_toolbar: fig.toolbar_location = None

    source_data = ColumnDataSource(data=dict(image=[image], x=[-w//2*dx], y=[-h//2*dy], dw=[w*dx], dh=[h*dy]))
    from bokeh.models import LinearColorMapper
    color_mapper = LinearColorMapper(palette='Greys256')    # Greys256, Viridis256
    image = fig.image(source=source_data, image='image', color_mapper=color_mapper,
                x='x', y='y', dw='dw', dh='dh'
            )

    # add hover tool only for the image
    from bokeh.models.tools import HoverTool, CrosshairTool
    if not tooltips:
        tooltips = [("x", "$xÅ"), ('y', '$yÅ'), ('val', '@image')]
    image_hover = HoverTool(renderers=[image], tooltips=tooltips)
    fig.add_tools(image_hover)
    fig.hover[0].attachment="vertical"
    crosshair = [t for t in fig.tools if isinstance(t, CrosshairTool)]
    if crosshair: 
        for ch in crosshair: ch.line_color = crosshair_color
    return fig

def create_layerline_image_figure(data, cutoff_res_x, cutoff_res_y, helical_radius, tilt, phase=None, pseudocolor=True, title="", yaxis_visible=True, tooltips=None):
    ny, nx = data.shape
    dsy = 1/(ny//2*cutoff_res_y)
    dsx = 1/(nx//2*cutoff_res_x)
    x_range = (-(nx//2+0.5)*dsx, (nx//2-0.5)*dsx)
    y_range = (-(ny//2+0.5)*dsy, (ny//2-0.5)*dsy)

    bessel = bessel_n_image(ny, nx, cutoff_res_x, cutoff_res_y, helical_radius, tilt).astype(np.int16)

    tools = 'box_zoom,pan,reset,save,wheel_zoom'
    fig = figure(title_location="below", frame_width=nx, frame_height=ny, 
        x_axis_label=None, y_axis_label=None, x_range=x_range, y_range=y_range, tools=tools)
    fig.grid.visible = False
    fig.title.text = title
    fig.title.align = "center"
    fig.title.text_font_size = "20px"
    fig.yaxis.visible = yaxis_visible   # leaving yaxis on will make the crosshair x-position out of sync with other figures

    source_data = ColumnDataSource(data=dict(image=[data.astype(np.float16)], x=[-nx//2*dsx], y=[-ny//2*dsy], dw=[nx*dsx], dh=[ny*dsy], bessel=[bessel]))
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

def add_linked_crosshair_tool(figures, dimensions="both"):
    # create a linked crosshair tool among the figures
    crosshair = CrosshairTool(dimensions=dimensions)
    crosshair.line_color = 'red'
    for fig in figures:
        fig.add_tools(crosshair)

def obtain_input_image(column, param_i=0, image_index_sync=0):
    if is_hosted():
        max_map_size  = mem_quota()/2    # MB
        max_map_dim   = int(pow(max_map_size*pow(2, 20)/4, 1./3.)//10*10)    # pixels in any dimension
        stop_map_size = mem_quota()*0.75 # MB
    else:
        max_map_size = -1   # no limit
        max_map_dim  = -1
    if max_map_size>0:
        warning_map_size = f"Due to the resource limit, the maximal map size should be {max_map_dim}x{max_map_dim}x{max_map_dim} voxels or less to avoid crashing the server process"

    with column:
        input_modes = {0:"upload", 1:"url", 2:"emd-xxxxx"}
        help = "Only maps in MRC (.mrc) or CCP4 (.map) format are supported. Compressed maps (.gz) will be automatically decompressed"
        if max_map_size>0: help += f". {warning_map_size}"
        input_mode = st.radio(label="How to obtain the input image/map:", options=list(input_modes.keys()), format_func=lambda i:input_modes[i], index=1,help=help, horizontal=True, key=f'input_mode_{param_i}')
        is_3d = False
        is_pwr_auto = None
        is_pd_auto = None
        if input_mode == 2:            
            emdb_ids, resolutions = get_emdb_ids()
            if not emdb_ids:
                st.warning("failed to obtained a list of helical structures in EMDB")
                return
            key_emd_id = f"emd_id_{param_i}"
            url = "https://www.ebi.ac.uk/emdb/search/*%20AND%20structure_determination_method:%22helical%22?rows=10&sort=release_date%20desc"
            st.markdown(f'[All {len(emdb_ids)} helical structures in EMDB]({url})')
            help = "Randomly select another helical structure in EMDB"
            if max_map_size>0: help += f". {warning_map_size}"
            button_clicked = st.button(label="Select a random EMDB ID", help=help)
            if button_clicked:
                import random
                st.session_state[key_emd_id] = 'emd-' + random.choice(emdb_ids)
            help = None
            if max_map_size>0: help = warning_map_size
            label = "Input an EMDB ID (emd-xxxxx):"
            st.text_input(label=label, value="emd-10499", help=help, key=key_emd_id)
            emd_id = st.session_state[key_emd_id].lower().split("emd-")[-1]
            if emd_id not in emdb_ids:
                emd_id_bad = emd_id
                import random
                emd_id = random.choice(emdb_ids)
                st.warning(f"EMD-{emd_id_bad} is not a helical structure. Please input a valid id (for example, a randomly selected valid id 'emd-{emd_id}')")
                st.stop()
            resolution = resolutions[emdb_ids.index(emd_id)]
            msg = f'[EMD-{emd_id}](https://www.ebi.ac.uk/emdb/entry/EMD-{emd_id}) | resolution={resolution}Å'
            params = get_emdb_helical_parameters(emd_id)
            if max_map_size>0 and params and "nz" in params and "ny" in params and "nx" in params:
                nz = params["nz"]
                ny = params["ny"]
                nx = params["nx"]
                map_size = nz*ny*nx*4 / pow(2, 20)
                if map_size>stop_map_size:
                    msg_map_too_large = f"As the map size ({map_size:.1f} MB, {nx}x{ny}x{nz} voxels) is too large for the resource limit ({mem_quota():.1f} MB memory cap) of the hosting service, HILL will stop analyzing it to avoid crashing the server. Please bin/crop your map so that it is {max_map_size} MB ({max_map_dim}x{max_map_dim}x{max_map_dim} voxels) or less, and then try again. Please check the [HILL web site](https://jiang.bio.purdue.edu/hill) to learn how to run HILL on your local computer with larger memory to support large maps"
                    st.warning(msg_map_too_large)
                    st.stop()
            if params:
                st.session_state[f"input_type_{param_i}"] = "image"
                st.session_state.twist = abs(params['twist'])
                st.session_state.rise = params['rise']
                st.session_state.csym = params['csym']
                st.session_state.resolution = params['resolution']
                msg += f"  \ntwist={params['twist']}° | rise={params['rise']}Å | c{params['csym']}"
            else:
                msg +=  "  \n*helical params not available*"
            st.markdown(msg)
            with st.spinner(f'Downloading EMD-{emd_id} from {get_emdb_map_url(emd_id)}'):
                data_all, apix_auto = get_emdb_map(emd_id)
                st.session_state.apix_0 = apix_auto
            if data_all is None:
                st.warning(f"Failed to download [EMD-{emd_id}](https://www.ebi.ac.uk/emdb/entry/EMD-{emd_id})")
                return

            image_index = 0
            is_3d = True
            is_pwr_auto = False
            is_pd_auto = False
        else:
            is_3d_auto = False
            if input_mode == 0:  # "upload a mrc/mrcs file":
                help = None
                if max_map_size>0: help = warning_map_size
                fileobj = st.file_uploader("Upload a mrc or mrcs file ", type=['mrc', 'mrcs', 'map', 'map.gz', 'tnf'], help=help, key=f'upload_{param_i}')
                if fileobj is not None:
                    is_pwr_auto = fileobj.name.find("ps.mrcs")!=-1
                    is_pd_auto = fileobj.name.find("pd.mrcs")!=-1
                    data_all, apix_auto = get_2d_image_from_uploaded_file(fileobj)
                    is_3d_auto = guess_if_3d(filename=fileobj.name, data=data_all)
                else:
                    st.stop()
            elif input_mode == 1:   # "url":
                    label = "Input a url of 2D image(s) or a 3D map:"
                    key_image_url = f'url_{param_i}'
                    if key_image_url not in st.session_state:
                        st.session_state[key_image_url] = data_examples[0].url
                    help = "An online url (http:// or ftp://) or a local file path (/path/to/your/structure.mrc)"
                    if max_map_size>0: help += f". {warning_map_size}"
                    image_url = st.text_input(label=label, help=help, key=key_image_url).strip()
                    is_pwr_auto = image_url.find("ps.mrcs")!=-1
                    is_pd_auto = image_url.find("pd.mrcs")!=-1
                    with st.spinner(f'Downloading {image_url.strip()}'):
                        data_all, apix_auto = get_2d_image_from_url(image_url)
                    is_3d_auto = guess_if_3d(filename=image_url, data=data_all)
            nz, ny, nx = data_all.shape
            if nz > 1:
                is_3d = st.checkbox(label=f"The input ({nx}x{ny}x{nz}) is a 3D map", value=is_3d_auto, key=f'is_3d_{param_i}', help="The app thinks the input image contains a stack of 2D images. Check this box to inform the app that the input is a 3D map")
            else:
                is_3d = False
        if is_3d:
            if not np.any(data_all):
                st.warning("All voxels of the input 3D map have zero value")
                st.stop()
            
            with st.expander(label="Generate 2-D projection from the 3-D map", expanded=False):
                az = st.number_input(label=f"Rotation around the helical axis (°):", min_value=0.0, max_value=360., value=0.0, step=1.0, key=f'az_{param_i}')
                tilt = st.number_input(label=f"Tilt (°):", min_value=-180.0, max_value=180., value=0.0, step=1.0, key=f'tilt_{param_i}')
                data = generate_projection(data_all, az=az, tilt=tilt)
                image_index = 0
                data_to_show = np.array([0])
        else:
            nonzeros = nonzero_images(data_all)
            if nonzeros is None:
                st.warning("All pixels of the input 2D images have zero value")
                st.stop()
            
            if "skipped" in st.session_state:
                data_to_show = np.array([i for i in nonzeros if i not in st.session_state.skipped])
            else:
                data_to_show = nonzeros

            if len(data_to_show) < 1:
                st.warning(f"All {len(data_all)} images have been skipped")
                st.stop()

            nz, ny, nx = data_all.shape
            if len(data_to_show)>1:
                st.markdown('<div style="text-align: center;">Click to choose an image:</div>', unsafe_allow_html=True)
                from st_clickable_images import clickable_images
                images = [encode_numpy(data_all[i], hflip=True) for i in data_to_show]
                image_index = clickable_images(
                    images,
                    titles=[f"{i+1}" for i in data_to_show],
                    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                    img_style={"margin": "2px", "height": "128px"},
                    key=f"image_index_{param_i}"
                )
                if image_index<0: image_index = 0
                image_index = data_to_show[image_index]
            else:
                image_index = data_to_show[0]
            data = data_all[image_index]

        if not np.any(data):
            st.warning("All pixels of the 2D image have zero value")
            st.stop()

        ny, nx = data.shape
        original_image = st.empty()

        image_parameters_expander = st.expander(label="Image parameters", expanded=False)
        with image_parameters_expander:
            if not is_3d and nz>1:
                rerun = False
                skip = st.button(label="Skip this image", help="Click the button to mark this image as a bad image that should be skipped")
                if skip:
                    rerun = True
                    if "skipped" not in st.session_state: st.session_state.skipped = set()
                    st.session_state.skipped.add(image_index)
                if "skipped" in st.session_state and len(st.session_state.skipped):
                    skipped_str = ' '.join(map(str, np.array(sorted(list(st.session_state.skipped)))+1))
                    skipped_str_new = st.text_input(label="Skipped images:", value=skipped_str, help=f"Specify the list of image indices (1 for first image, separated by spaces) that should be skipped")
                    if skipped_str_new != skipped_str:
                        rerun = True
                        tmp = []
                        for s in skipped_str_new.split():
                            try:
                                tmp.append(int(s))
                            except:
                                pass
                        skipped_new = np.array(tmp, dtype=int)-1
                        st.session_state.skipped = set(skipped_new)
                if rerun:
                    st.experimental_rerun()

            input_type_auto = None
            if input_type_auto is None:
                if is_pwr_auto is None: is_pwr_auto = guess_if_is_power_spectra(data)
                if is_pd_auto is None: is_pd_auto = guess_if_is_phase_differences_across_meridian(data)
                if is_pwr_auto: input_type_auto = "PS"
                elif is_pd_auto: input_type_auto = "PD"
                else: input_type_auto = "image"
            mapping = {"image":0, "PS":1, "PD":2}
            input_type = st.radio(label="Input is:", options="image PS PD".split(), index=mapping[input_type_auto], help="image: real space image; PS: power spectra; PD: phage differences across meridian", horizontal=True, key=f'input_type_{param_i}')
            if input_type in ["PS", "PD"]:
                apix = 0.5 * st.number_input('Nyquist res (Å)', value=2*apix_auto, min_value=0.1, max_value=30., step=0.01, format="%.5g", key=f'apix_nyquist_{param_i}')
            else:
                apix = st.number_input('Pixel size (Å/pixel)', value=apix_auto, min_value=0.1, max_value=30., step=0.01, format="%.5g", key=f'apix_{param_i}')
            transpose_auto = input_mode not in [2, 3] and nx > ny
            transpose = st.checkbox(label='Transpose the image', value=transpose_auto, key=f'transpose_{param_i}')
            negate_auto = not guess_if_is_positive_contrast(data)
            negate = st.checkbox(label='Invert the image contrast', value=negate_auto, key=f'negate_{param_i}')
            if input_type in ["PS", "PD"] or is_3d:
                angle_auto, dx_auto = 0., 0.
            else:
                angle_auto, dx_auto = auto_vertical_center(data)
            angle = st.number_input('Rotate (°) ', value=-angle_auto, min_value=-180., max_value=180., step=1.0, format="%.4g", key=f'angle_{param_i}')
            dx = st.number_input('Shift along X-dim (Å) ', value=dx_auto*apix, min_value=-nx*apix, max_value=nx*apix, step=1.0, format="%.3g", key=f'dx_{param_i}')
            dy = st.number_input('Shift along Y-dim (Å) ', value=0.0, min_value=-ny*apix, max_value=ny*apix, step=1.0, format="%.3g", key=f'dy_{param_i}')

            mask_empty = st.container()        

        with original_image:
            if is_3d:
                image_label = f"Orignal image ({nx}x{ny})"
            else:
                image_label = f"Orignal image {image_index+1}/{nz} ({nx}x{ny})"
            #st.image(normalize(data), use_column_width=True, caption=image_label)
            fig = create_image_figure(data, apix, apix, title=image_label, title_location="below", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=None, show_axis=False, show_toolbar=False, crosshair_color="white", aspect_ratio=1)
            st.bokeh_chart(fig, use_container_width=True)

        transformed_image = st.empty()
        transformed = transpose or negate or angle or dx
        if transpose:
            data = data.T
        if negate:
            data = -data
        if angle or dx or dy:
            data = rotate_shift_image(data, angle=-angle, post_shift=(dy/apix, dx/apix), order=1)

        radius_auto = 0
        mask_radius = 0
        if input_type in ["image"]:
            radius_auto, mask_radius_auto = estimate_radial_range(data, thresh_ratio=0.1)
            mask_radius = mask_empty.number_input('Mask radius (Å) ', value=min(mask_radius_auto*apix, nx/2*apix), min_value=1.0, max_value=nx/2*apix, step=1.0, format="%.1f", key=f'mask_radius_{param_i}')
            mask_len_percent_auto = 90.0
            mask_len_fraction = mask_empty.number_input('Mask length (%) ', value=mask_len_percent_auto, min_value=10.0, max_value=100.0, step=1.0, format="%.1f", key=f'mask_len_{param_i}') / 100.0

            x = np.arange(-nx//2, nx//2)*apix
            ymax = np.max(data, axis=0)
            ymean = np.mean(data, axis=0)

            tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
            tooltips = [("X", "@x{0.0}Å")]
            p = figure(x_axis_label="x (Å)", y_axis_label="pixel value", frame_height=200, tools=tools, tooltips=tooltips)
            p.line(x, ymax, line_width=2, color='red', legend_label="max")
            p.line(-x, ymax, line_width=2, color='red', line_dash="dashed", legend_label="max flipped")
            p.line(x, ymean, line_width=2, color='blue', legend_label="mean")
            p.line(-x, ymean, line_width=2, color='blue', line_dash="dashed", legend_label="mean flipped")
            rmin_span = Span(location=-mask_radius, dimension='height', line_color='green', line_dash='dashed', line_width=3)
            rmax_span = Span(location=mask_radius, dimension='height', line_color='green', line_dash='dashed', line_width=3)
            p.add_layout(rmin_span)
            p.add_layout(rmax_span)
            p.yaxis.visible = False
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
        
            fraction_x = mask_radius/(nx//2*apix)
            tapering_image = generate_tapering_filter(image_size=data.shape, fraction_start=[mask_len_fraction, fraction_x], fraction_slope=(1.0-mask_len_fraction)/2.)
            data = data * tapering_image
            transformed = 1

        if transformed:
            with transformed_image:
                if is_3d:
                    image_label = f"Transformed image ({nx}x{ny})"
                else:
                    image_label = f"Transformed image {image_index+1}/{nz} ({nx}x{ny})"
                #st.image(normalize(data), use_column_width=True, caption=image_label)
                fig = create_image_figure(data, apix, apix, title=image_label, title_location="below", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=None, show_axis=False, show_toolbar=False, crosshair_color="white", aspect_ratio=1)
                st.bokeh_chart(fig, use_container_width=True)

        if input_type in ["image"]:
            acf = auto_correlation(data, sqrt=True, high_pass_fraction=0.1)
            y = np.arange(-ny//2, ny//2)*apix
            xmax = np.max(acf, axis=1)

            tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
            tooltips = [("Axial Shift", "@y{0.0}Å")]
            p = figure(x_axis_label="Auto-correlation", y_axis_label="Axial Shift (Å)", frame_height=ny, tools=tools, tooltips=tooltips)
            p.line(xmax, y, line_width=2, color='red', legend_label="ACF")
            if 0:
                xmean = np.mean(acf, axis=1)
                p.line(xmean, y, line_width=2, color='blue', legend_label="mean")
            p.hover[0].attachment = "above"
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
        if is_3d:
            image_label = f"Transformed image"
        else:
            image_label = f"Transformed image {image_index+1}"
    else:
        image_container = original_image
        if is_3d:
            image_label = f"Original image"
        else:
            image_label = f"Original image {image_index+1}"

    if input_mode in [2, 3]:
        input_params = (input_mode, (None, None, emd_id))
    elif input_mode==1:
        input_params = (input_mode, (None, image_url, None))
    else:
        input_params = (input_mode, (fileobj, None, None))
    return data_all, image_index, data, apix, radius_auto, mask_radius, input_type, is_3d, input_params, (image_container, image_label)

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
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

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
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

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def compute_layer_line_positions(twist, rise, csym, radius, tilt, cutoff_res, m_max=-1):
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

    if m_max<1:
        m_max = int(np.floor(np.abs(rise/cutoff_res)))+3
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
        p = twist2pitch(twist, rise)
        ds_p = 1/p
        ll_i_top = int(np.abs(smax - sy0)/ds_p) * 2
        ll_i_bottom = -int(np.abs(-smax - sy0)/ds_p) * 2
        ll_i = np.array([i for i in range(ll_i_bottom, ll_i_top+1) if not i%csym], dtype=np.float32)
        sy = sy0 + ll_i * ds_p
        sx = peak_sx(bessel_order=ll_i, radius=radius)
        if tilt:
            sy = np.array(sy, dtype=np.float32) * tf
            sx = np.sqrt(np.power(np.array(sx, dtype=np.float32), 2) - np.power(sy*tf2, 2))
            sx[np.isnan(sx)] = 1e-6
        px  = list(sx) + list(-sx)
        py  = list(sy) + list(sy)
        n = list(ll_i) + list(ll_i)
        d["LL"] = (px, py, n)
        d["m"] = m

        m_groups[m[mi]] = d
    return m_groups

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def compute_phase_difference_across_meridian(phase):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
    phase_diff = phase * 0
    phase_diff[..., 1:] = phase[..., 1:] - phase[..., 1:][..., ::-1]
    phase_diff = np.rad2deg(np.arccos(np.cos(phase_diff)))   # set the range to [0, 180]. 0 -> even order, 180 - odd order
    return phase_diff

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def resize_rescale_power_spectra(data, nyquist_res, cutoff_res=None, output_size=None, log=True, low_pass_fraction=0, high_pass_fraction=0, norm=1):
    from scipy.ndimage import map_coordinates
    ny, nx = data.shape
    ony, onx = output_size
    res_y, res_x = cutoff_res
    Y, X = np.meshgrid(np.arange(ony, dtype=np.float32)-(ony//2+0.5), np.arange(onx, dtype=np.float32)-(onx//2+0.5), indexing='ij')
    Y = Y/(ony//2+0.5) * nyquist_res/res_y * ny//2 + ny//2+0.5
    X = X/(onx//2+0.5) * nyquist_res/res_x * nx//2 + nx//2+0.5
    pwr = map_coordinates(data, (Y.flatten(), X.flatten()), order=3, mode='constant').reshape(Y.shape)
    if log: pwr = np.log1p(np.abs(pwr))
    if 0<low_pass_fraction<1 or 0<high_pass_fraction<1:
        pwr = low_high_pass_filter(pwr, low_pass_fraction=low_pass_fraction, high_pass_fraction=high_pass_fraction)
    if norm: pwr = normalize(pwr, percentile=(0, 100))
    return pwr

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def compute_power_spectra(data, apix, cutoff_res=None, output_size=None, log=True, low_pass_fraction=0, high_pass_fraction=0):
    fft = fft_rescale(data, apix=apix, cutoff_res=cutoff_res, output_size=output_size)
    fft = np.fft.fftshift(fft)  # shift fourier origin from corner to center

    if log: pwr = np.log1p(np.abs(fft))
    if 0<low_pass_fraction<1 or 0<high_pass_fraction<1:
        pwr = low_high_pass_filter(pwr, low_pass_fraction=low_pass_fraction, high_pass_fraction=high_pass_fraction)
    pwr = normalize(pwr, percentile=(0, 100))

    phase = np.angle(fft, deg=False)
    return pwr, phase

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
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
    fft = nufft2d2(x=Y, y=X, f=image.astype(np.complex128), eps=1e-6)
    fft = fft.reshape((ony, onx))

    # phase shifts for real-space shifts by half of the image box in both directions
    phase_shift = np.ones(fft.shape)
    phase_shift[1::2, :] *= -1
    phase_shift[:, 1::2] *= -1
    fft *= phase_shift
    # now fft has the same layout and phase origin (i.e. np.fft.ifft2(fft) would obtain original image)
    return fft

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def auto_correlation(data, sqrt=True, high_pass_fraction=0):
    from scipy.signal import correlate2d
    fft = np.fft.rfft2(data)
    product = fft*np.conj(fft)
    if sqrt: product = np.sqrt(product)
    if 0<high_pass_fraction<=1:
        ny, nx = product.shape
        Y, X = np.meshgrid(np.arange(-ny//2, ny//2, dtype=float), np.arange(-nx//2, nx//2, dtype=float), indexing='ij')
        Y /= ny//2
        X /= nx//2
        f2 = np.log(2)/(high_pass_fraction**2)
        filter = 1.0 - np.exp(- f2 * Y**2) # Y-direction only
        product *= np.fft.fftshift(filter)
    corr = np.fft.fftshift(np.fft.irfft2(product))
    corr /= np.max(corr)
    return corr

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
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

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
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

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def estimate_radial_range(data, thresh_ratio=0.1):
    proj_y = np.sum(data, axis=0)
    n = len(proj_y)
    background = np.mean(proj_y[[0,1,2,-3,-3,-1]])
    thresh = (proj_y.max() - background) * thresh_ratio + background
    indices = np.nonzero(proj_y>thresh)
    xmin = np.min(indices)
    xmax = np.max(indices)
    mask_radius = max(abs(n//2-xmin), abs(xmax-n//2))
    proj_y -= thresh
    proj_y[proj_y<0] = 0
    def fitRadialProfile(x, radProfile):
        a, b, w, rcore, rmax= x  # y = a*(sqrt(rmax^2-x^2)+(w-1)*sqrt(rcore^2-x^2))+b
        try:
            n = len(radProfile)
            x = np.abs(np.arange(n, dtype=float)-n/2)
            yshell = radProfile * 0
            mask = x<=abs(rmax)
            yshell[mask] = np.sqrt(rmax*rmax - x[mask]*x[mask])
            ycore = radProfile * 0
            mask = x<=abs(rcore)
            ycore[mask] = np.sqrt(rcore*rcore - x[mask]*x[mask])
            y = a*(yshell+(w-1)*ycore)+b
            score = np.linalg.norm(y-radProfile)
        except:
            score = 1e10
        return score
    from scipy.optimize import minimize
    from itertools import product
    bounds = ((0, None), (None, None), (0, None), (0, mask_radius), (0, mask_radius))
    vals_a = (1, 2, 4, 8)
    vals_w = (0, 0.5)
    vals_rcore = (0, mask_radius/2)
    results = []
    for val_a, val_w, val_rcore in product(vals_a, vals_w, vals_rcore):
        x0 = (val_a, 0, val_w, val_rcore, mask_radius)
        res = minimize(fitRadialProfile, x0, args=(proj_y,), method='Nelder-Mead', bounds=bounds, tol=1e-6)
        a, b, w, rcore, rmax = res.x
        results.append((res.fun, w, rcore, rmax, val_a, val_w, val_rcore))
    result = sorted(results)[0]
    w, rcore, rmax = result[1:4]
    rmean = 0.5 * (rmax*rmax+(w-1)*rcore*rcore) / (rmax+(w-1)*rcore)
    return float(rmean), float(mask_radius)    # pixel

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def auto_vertical_center(image):
    image_work = image * 1.0
    background = np.mean(image_work[[0,1,2,-3,-2,-1],[0,1,2,-3,-2,-1]])
    max_val = image_work.max()
    thresh = (max_val-background) * 0.2 + background
    if background < thresh < max_val:
        image_work = (image_work-thresh)/(max_val-thresh)
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
    res = minimize_scalar(score_rotation, bounds=(-90, 90), method='bounded', options={'disp':0})
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
    res = fmin(score_rotation_shift, x0=(angle, 0, 0), xtol=1e-2, disp=0)
    angle = res[0]  # dy, dx are not robust enough

    # refine dx 
    image_work = rotate_shift_image(data=image_work, angle=angle)
    y = np.sum(image_work, axis=0)
    y /= y.max()
    y[y<0.2] = 0
    n = len(y)
    try:
        mask = np.where(y>0.5)
        max_shift = abs((np.max(mask)-n//2) - (n//2-np.min(mask)))*1.5
    except:
        max_shift = n/4

    import scipy.interpolate as interpolate
    x = np.arange(3*n)
    f = interpolate.interp1d(x, np.tile(y, 3), kind='linear')    # avoid out-of-bound errors
    def score_shift(dx):
        if dx<-n or dx>n: return np.finfo(np.float32).max
        x_tmp = x[n:2*n]-dx
        tmp = f(x_tmp)
        err = np.sum(np.abs(tmp-tmp[::-1]))
        return err
    res = minimize_scalar(score_shift, bounds=(-max_shift, max_shift), method='bounded', options={'disp':0})
    dx = res.x + (0.0 if n%2 else 0.5)
    return set_to_periodic_range(angle), dx

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def rotate_shift_image(data, angle=0, pre_shift=(0, 0), post_shift=(0, 0), rotation_center=None, order=1):
    # pre_shift/rotation_center/post_shift: [y, x]
    if angle==0 and pre_shift==[0,0] and post_shift==[0,0]: return data*1.0
    ny, nx = data.shape
    if rotation_center is None:
        rotation_center = np.array((ny//2, nx//2), dtype=np.float32)
    ang = np.deg2rad(angle)
    m = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]], dtype=np.float32)
    pre_dy, pre_dx = pre_shift    
    post_dy, post_dx = post_shift

    offset = -np.dot(m, np.array([post_dy, post_dx], dtype=np.float32).T) # post_rotation shift
    offset += np.array(rotation_center, dtype=np.float32).T - np.dot(m, np.array(rotation_center, dtype=np.float32).T)  # rotation around the specified center
    offset += -np.array([pre_dy, pre_dx], dtype=np.float32).T     # pre-rotation shift

    from scipy.ndimage import affine_transform
    ret = affine_transform(data, matrix=m, offset=offset, order=order, mode='constant')
    return ret

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def generate_projection(data, az=0, tilt=0, output_size=None):
    from scipy.spatial.transform import Rotation as R
    from scipy.ndimage import affine_transform
    # note the convention change
    # xyz in scipy is zyx in cryoEM maps
    rot = R.from_euler('zx', [tilt, az], degrees=True)  # order: right to left
    m = rot.as_matrix()
    nx, ny, nz = data.shape
    bcenter = np.array((nx//2, ny//2, nz//2), dtype=np.float32)
    offset = bcenter.T - np.dot(m, bcenter.T)
    tmp = affine_transform(data, matrix=m, offset=offset, mode='nearest')
    ret = tmp.sum(axis=1)   # integrate along y-axis
    if output_size is not None:
        ony, onx = output_size
        ny, nx = ret.shape
        if ony!=ny or onx!=nx:
            top_bottom_mean = np.mean(ret[(0,-1),:])
            ret2 = np.zeros((ony, onx), dtype=np.float32) + top_bottom_mean
            y0 = ony//2 - ny//2
            x0 = onx//2 - nx//2
            ret2[y0:y0+ny, x0:x0+nx] = ret
            ret = ret2 
    ret = normalize(ret)
    return ret

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def normalize(data, percentile=(0, 100)):
    p0, p1 = percentile
    vmin, vmax = sorted(np.percentile(data, (p0, p1)))
    data2 = (data-vmin)/(vmax-vmin)
    return data2

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def nonzero_images(data, thresh_ratio=1e-3):
    assert(len(data.shape) == 3)
    sigmas = np.std(data, axis=(1,2))
    thresh = sigmas.max() * thresh_ratio
    nonzeros = np.where(sigmas>thresh)[0]
    if len(nonzeros)>0: 
        return nonzeros
    else:
        None

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def guess_if_is_phase_differences_across_meridian(data, err=30):
    if np.any(data[:, 0]):
        return False
    if not (data.min()==0 and (0<=180-data.max()<err)):
        return False
    sym_diff = data[:, 1:] - data[:, 1:][:, ::-1]
    if np.any(sym_diff):
        return False
    return True

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def guess_if_is_power_spectra(data, thresh=15):
    median = np.median(data)
    max = np.max(data)
    sigma = np.std(data)
    if (max-median)>thresh*sigma: return True
    else: return False

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def guess_if_is_positive_contrast(data):
    edge_mean = np.mean([data[0, :].mean(), data[-1, :].mean(), data[:, 0].mean(), data[:, -1].mean()])
    return edge_mean < np.mean(data)

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def guess_if_3d(filename, data=None):
    if filename.endswith(".mrcs"): return False
    if filename.startswith("cryosparc") and filename.endswith("_class_averages.mrc"): return False    # cryosparc_P*_J*_*_class_averages.mrc
    if data is None: return None
    if len(data.shape)<3: return False
    if len(data.shape)>3: return None
    nz, ny, nx = data.shape
    if nz==1: return False
    if nz==ny and nz==nx: return True
    if ny==nx and nz in [50, 100, 200]: return False
    return None

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def get_2d_image_from_uploaded_file(fileobj):
    import os, tempfile
    orignal_filename = fileobj.name
    suffix = os.path.splitext(orignal_filename)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(fileobj.read())
        data, apix = get_2d_image_from_file(temp.name)
    return data.astype(np.float32), apix

@st.experimental_memo(persist='disk', show_spinner=False, ttl=24*60*60.) # refresh every day
def get_emdb_ids():
    try:
        import pandas as pd
        entries = pd.read_csv("https://www.ebi.ac.uk/emdb/api/search/current_status:%22REL%22%20AND%20structure_determination_method:%22helical%22?wt=csv&download=true&fl=emdb_id,resolution")
        emdb_ids = list(entries.iloc[:,0].str.split('-', expand=True).iloc[:, 1].values)
        resolutions = entries.iloc[:,1].values
    except:
        emdb_ids = []
        resolutions = []
    return emdb_ids, resolutions

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def get_emdb_helical_parameters(emd_id):
    try:
        emd_id2 = ''.join([s for s in str(emd_id) if s.isdigit()])
        url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emd_id2}/header/emd-{emd_id2}.xml"
        from urllib.request import urlopen
        with urlopen(url) as response:
            xml_data = response.read()
        import_with_auto_install(["xmltodict"])
        import xmltodict
        data = xmltodict.parse(xml_data)
        ret = {}
        ret['sample'] = data['emd']['sample']['name']
        ret["method"] = data['emd']['structure_determination_list']['structure_determination']['method']
        dimensions = data['emd']['map']['dimensions']
        ret["nz"] = int(dimensions["sec"])
        ret["ny"] = int(dimensions["row"])
        ret["nx"] = int(dimensions["col"])
        res_dict = dict_recursive_search(data, 'resolution')
        if res_dict:
            ret["resolution"] = float(res_dict['#text'])
        if ret["method"] == 'helical':
            #ret["resolution"] = float(data['emd']['structure_determination_list']['structure_determination']['helical_processing']['final_reconstruction']['resolution']['#text'])
            helical_parameters = data['emd']['structure_determination_list']['structure_determination']['helical_processing']['final_reconstruction']['applied_symmetry']['helical_parameters']
            #assert(helical_parameters['delta_phi']['@units'] == 'deg')
            #assert(helical_parameters['delta_z']['@units'] == 'Å')
            ret["twist"] = float(helical_parameters['delta_phi']['#text'])
            ret["rise"] = float(helical_parameters['delta_z']['#text'])
            ret["csym"] = int(helical_parameters['axial_symmetry'][1:])
            return ret
        else:
            return None
    except:
        return None

def get_emdb_map_url(emd_id: str):
    server = "https://ftp.wwpdb.org/pub"    # Rutgers University, USA
    #server = "https://ftp.ebi.ac.uk/pub/databases" # European Bioinformatics Institute, England
    #server = "http://ftp.pdbj.org/pub" # Osaka University, Japan
    url = f"{server}/emdb/structures/EMD-{emd_id}/map/emd_{emd_id}.map.gz"
    return url

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def get_emdb_map(emd_id: str):
    url = get_emdb_map_url(emd_id)
    ds = np.DataSource(None)
    fp = ds.open(url)
    import mrcfile
    with mrcfile.open(fp.name) as mrc:
        vmin, vmax = np.min(mrc.data), np.max(mrc.data)
        data = ((mrc.data - vmin) / (vmax - vmin))
        apix = mrc.voxel_size.x.item()
    return data.astype(np.float32), apix

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False, suppress_st_warning=True)
def get_2d_image_from_url(url):
    url_final = get_direct_url(url)    # convert cloud drive indirect url to direct url
    ds = np.DataSource(None)
    if not ds.exists(url_final):
        st.error(f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview")
        st.stop()
    with ds.open(url) as fp:
        data = get_2d_image_from_file(fp.name)
    return data

#@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False)
def get_2d_image_from_file(filename):
    try:
        import mrcfile
        with mrcfile.open(filename) as mrc:
            data = mrc.data.astype(np.float32)
            apix = mrc.voxel_size.x.item()
    except:
        from skimage.io import imread
        data = imread(filename, as_gray=1) * 1.0    # return: numpy array
        data = data[::-1, :]
        apix = 1.0

    if data.dtype==np.dtype('complex64'):
        data_complex = data
        ny, nx = data_complex[0].shape
        data = np.zeros((len(data_complex), ny, (nx-1)*2), dtype=np.float32)
        for i in range(len(data)):
            tmp = np.abs(np.fft.fftshift(np.fft.fft(np.fft.irfft(data_complex[i])), axes=1))
            data[i] = normalize(tmp, percentile=(0.1, 99.9))
    if len(data.shape)==2:
        data = np.expand_dims(data, axis=0)
    return data.astype(np.float32), apix

def twist2pitch(twist, rise):
    if twist:
        return 360. * rise/abs(twist)
    else:
        return rise

def pitch2twist(pitch, rise):
    if pitch>rise:
        return 360. * rise/pitch
    else:
        return 0.

def encode_numpy(img, hflip=True):
    if img.dtype != np.dtype('uint8'):
        vmin, vmax = img.min(), img.max()
        tmp = (255*(img-vmin)/(vmax-vmin)).astype(np.uint8)
    else:
        tmp = img
    if hflip:
        tmp = tmp[::-1, :]
    import io, base64
    from PIL import Image
    pil_img = Image.fromarray(tmp)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64, {encoded}"

class Data:
    def __init__(self, twist, rise, csym, diameter, dx=0, apix_or_nqyuist=None, url=None, input_type="image"):
        self.input_type = input_type
        self.twist = twist
        self.rise = rise
        self.csym = csym
        self.diameter = diameter
        self.dx = dx
        if self.input_type in ["PS", "PD"]:
            self.nyquist = apix_or_nqyuist
        else:
            self.apix = apix_or_nqyuist
        self.url = url

data_examples = [
    Data(twist=29.40, rise=21.92, csym=6, diameter=138, url="https://tinyurl.com/y5tq9fqa"),
    Data(twist=0.92, rise=4.83, csym=1, diameter=60, url="https://tinyurl.com/2p9yxe7x"),
    Data(twist=36.0, rise=3.4, csym=1, diameter=20, dx=5, input_type="PS", apix_or_nqyuist=2.5, url="https://upload.wikimedia.org/wikipedia/en/b/b2/Photo_51_x-ray_diffraction_image.jpg")
]

def set_session_state_from_data_example():
    data = np.random.choice(data_examples)
    st.session_state.input_mode_0 = 1
    st.session_state.input_type_0 = data.input_type
    st.session_state.url_0 = data.url
    st.session_state.dx_0 = float(data.dx)
    if data.input_type in ["PS", "PD"]:
        if data.nyquist is not None:
            st.session_state.apix_nyquist_0 = data.nyquist
    else:
        if data.apix is not None:
            st.session_state.apix_0 = data.apix
    st.session_state.rise = float(data.rise)
    st.session_state.twist = float(abs(data.twist))
    st.session_state.csym = int(data.csym)
    st.session_state.diameter = float(data.diameter)

@st.experimental_memo(persist=None, show_spinner=False)
def set_initial_query_params(query_string):
    if len(query_string)<1: return
    from urllib.parse import parse_qs
    d = parse_qs(query_string)
    if len(d)<1: return
    st.session_state.update(d)

int_types = ['csym', 'do_random_embid_0', 'do_random_embid_1', 'image_index_0', 'image_index_1', 'input_mode_0', 'input_mode_1', 'is_3d_0', 'is_3d_1', 'negate_0', 'negate_1', 'pnx', 'pny', 'show_LL', 'show_LL_text', 'show_phase_diff', 'show_pwr', 'show_yprofile', 'simunoise', 'transpose_0', 'transpose_1', 'share_url', 'show_qr', 'useplotsize']
float_types = ['angle_0', 'angle_1', 'apix_0', 'apix_1', 'apix_nyquist_0', 'apix_nyquist_1', 'az_0', 'az_1', 'ball_radius', 'cutoff_res_x', 'cutoff_res_y', 'diameter', 'dx_0', 'dx_1', 'dy_0', 'dy_1', 'mask_radius_0', 'mask_radius_1', 'mask_len_0', 'mask_len_1', 'resolution', 'rise', 'simuaz', 'simunoise', 'tilt', 'tilt_0', 'tilt_1', 'twist']
default_values = {'angle_0':0, 'angle_1':0, 'az_0':0, 'ball_radius':0, 'csym':1, 'do_random_embid_0':0, 'dx_0':0, 'dx_1':0, 'dy_0':0, 'dy_1':0, 'image_index_0':0, 'input_type_0':'image', 'is_3d_0':0, 'is_3d_1':0, 'mask_len_0':90, 'mask_len_1':90, 'negate_0':0, 'negate_1':0, 'pnx':512, 'pny':1024, 'show_LL':1, 'show_LL_text':1, 'show_phase_diff':1, 'show_pwr':1, 'show_yprofile':1, 'simuaz':0, 'simunoise':0, 'tilt':0, 'tilt_0':0, 'tilt_1':0, 'transpose_0':0, 'transpose_1':0, 'share_url':0, 'show_qr':0, 'useplotsize':0}
def set_query_params_from_session_state():
    d = {}
    attrs = sorted(st.session_state.keys())
    for attr in attrs:
        v = st.session_state[attr]
        if attr in default_values and v==default_values[attr]: continue
        if attr in int_types or isinstance(v, bool):
            d[attr] = int(v)
        elif attr in float_types:
            d[attr] = f'{float(v):g}'
        else:
            d[attr] = v
    st.experimental_set_query_params(**d)

def set_session_state_from_query_params():
    query_params = st.experimental_get_query_params()
    for attr in sorted(query_params.keys()):
            if attr in int_types:
                st.session_state[attr] = int(query_params[attr][0])
            elif attr in float_types:
                st.session_state[attr] = float(query_params[attr][0])
            else:
                st.session_state[attr] = query_params[attr][0]

def get_direct_url(url):
    import re
    if url.startswith("https://drive.google.com/file/d/"):
        hash = url.split("/")[5]
        return f"https://drive.google.com/uc?export=download&id={hash}"
    elif url.startswith("https://app.box.com/s/"):
        hash = url.split("/")[-1]
        return f"https://app.box.com/shared/static/{hash}"
    elif url.startswith("https://www.dropbox.com"):
        if url.find("dl=1")!=-1: return url
        elif url.find("dl=0")!=-1: return url.replace("dl=0", "dl=1")
        else: return url+"?dl=1"
    elif url.find("sharepoint.com")!=-1 and url.find("guestaccess.aspx")!=-1:
        return url.replace("guestaccess.aspx", "download.aspx")
    elif url.startswith("https://1drv.ms"):
        import base64
        data_bytes64 = base64.b64encode(bytes(url, 'utf-8'))
        data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
        return f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    else:
        return url

def set_to_periodic_range(v, min=-180, max=180):
    from math import fmod
    tmp = fmod(v-min, max-min)
    if tmp>=0: tmp+=min
    else: tmp+=max
    return tmp

def dict_recursive_search(d, key, default=None):
    stack = [iter(d.items())]
    while stack:
        for k, v in stack[-1]:
            if k == key:          
                return v
            elif isinstance(v, dict):
                stack.append(iter(v.items()))
                break
        else:
            stack.pop()
    return default

@st.experimental_memo(persist='disk', show_spinner=False)
def setup_anonymous_usage_tracking():
    try:
        import pathlib, stat
        index_file = pathlib.Path(st.__file__).parent / "static/index.html"
        index_file.chmod(stat.S_IRUSR|stat.S_IWUSR|stat.S_IRGRP|stat.S_IROTH)
        txt = index_file.read_text()
        if txt.find("gtag/js?")==-1:
            txt = txt.replace("<head>", '''<head><script async src="https://www.googletagmanager.com/gtag/js?id=G-7SGET974KQ"></script><script>window.dataLayer = window.dataLayer || [];function gtag(){dataLayer.push(arguments);}gtag('js', new Date());gtag('config', 'G-7SGET974KQ');</script>''')
            index_file.write_text(txt)
    except:
        pass

def mem_info():
    import_with_auto_install(["psutil"])
    from psutil import virtual_memory
    mem = virtual_memory()
    mb = pow(2, 20)
    return (mem.total/mb, mem.available/mb, mem.used/mb, mem.percent)

def mem_quota():
    fqdn = get_hostname()
    if fqdn.find("heroku")!=-1:
        return 512  # MB
    username = get_username()
    if username.find("appuser")!=-1:    # streamlit share
        return 1024  # MB
    available_mem = mem_info()[1]
    return available_mem

def mem_used():
    import_with_auto_install(["psutil"])
    from psutil import Process
    from os import getpid
    mem = Process(getpid()).memory_info().rss / 1024**2   # MB
    return mem

def uptime():
    import_with_auto_install(["uptime"])
    from uptime import uptime
    return uptime()

def get_username():
    from getpass import getuser
    return getuser()

def get_hostname():
    import socket
    fqdn = socket.getfqdn()
    return fqdn

def is_hosted(return_host=False):
    hosted = False
    host = ""
    fqdn = get_hostname()
    if fqdn.find("heroku")!=-1:
        hosted = True
        host = "heroku"
    username = get_username()
    if username.find("appuser")!=-1:
        hosted = True
        host = "streamlit"
    if not host:
        host = "localhost"
    if return_host:
        return hosted, host
    else:
        return hosted

def qr_code(url=None, size = 8):
    import_with_auto_install(["qrcode"])
    import qrcode
    if url is None: # ad hoc way before streamlit can return the url
        _, host = is_hosted(return_host=True)
        if len(host)<1: return None
        if host == "streamlit":
            url = "https://share.streamlit.io/wjiang/HILL/main/"
        elif host == "heroku":
            url = "https://helical-indexing-HILL.herokuapp.com/"
        else:
            url = f"http://{host}:8501/"
        import urllib
        params = st.experimental_get_query_params()
        d = {k:params[k][0] for k in params}
        url += "?" + urllib.parse.urlencode(d)
    if not url: return None
    img = qrcode.make(url)  # qrcode.image.pil.PilImage
    data = np.array(img.convert("RGBA"))
    return data

if __name__ == "__main__":
    setup_anonymous_usage_tracking()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_string", metavar="<str>", type=str, help="set initial url query params from this string. default: %(default)s", default="")
    args = parser.parse_args()

    main(args)
    gc.collect(2)
