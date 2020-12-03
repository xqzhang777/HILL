from bokeh.plotting import figure
import streamlit as st
import numpy as np

def main():
    st.set_page_config(page_title="Helical Indexing", layout="wide")
    st.server.server_util.MESSAGE_SIZE_LIMIT = 2e8  # default is 5e7 (50MB)

    title = "Helical indexing using layer lines"
    st.title(title)

    col1, col2, col3, col4 = st.beta_columns((1.5, 0.75, 0.25, 3.5))

    with col1:
        with st.beta_expander(label="README", expanded=False):
            st.write("This Web app considers a biological helical structure as the product of a continous helix and a set of parallel planes, and based on the covolution theory, the Fourier Transform (FT) of a helical structure would be the convolution of the FT of the continous helix and the FT of the planes.  \nThe FT of a continous helix consists of equally spaced layer planes (3D) or layerlines (2D projection) that can be described by Bessel functions of increasing orders (0, +/-1, +/-2, ...) from the Fourier origin (i.e. equator). The spacing between the layer planes/lines is determined by the helical pitch (i.e. the shift along the helical axis for a 360 ° turn of the helix). If the structure has additional cyclic symmetry (for example, C6) around the helical axis, only the layer plane/line orders of integer multiplier of the symmetry (e.g. 0, +/-6, +/-12, ...) are visible. The primary peaks of the layer lines in the power spectra form a pattern similar to a X symbol.  \nThe FT of the parallel planes consists of equally spaced points along the helical axis (i.e. meridian) with the spacing being determined by the helical rise.  \nThe convolution of these two components (X-shaped pattern of layer lines and points along the meridian) generates the layer line patterns seen in the power spectra of the projection images of helical structures. The helical indexing task is thus to identify the helical rise, pitch (or twist), and cyclic symmetry that would predict a layer line pattern to explain the observed the layer lines in the power spectra. This Web app allows you to interactively change the helical parameters and superimpose the predicted layer liines on the power spectra to complete the helical indexing task.  \n  \nPS: power spectra; PD: phase difference between the two sides of meridian; YP: Y-axis power spectra profile; LL: layer lines; m: indices of the X-patterns along the meridian; Jn: Bessel order")
        
        # make radio display horizontal
        st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        input_mode = st.radio(label="How to obtain the input image/map:", options=["upload a mrc/mrcs file", "url", "emd-xxxx"], index=1)
        if input_mode == "upload a mrc/mrcs file":
            fileobj = st.file_uploader("Upload a mrc or mrcs file", type=['mrc', 'mrcs', 'map', 'map.gz'])
            if fileobj is not None:
                data_all, apix = get_2d_image_from_uploaded_file(fileobj)
            else:
                return
        else:
            if input_mode == "url":
                label = "Input a url of 2D projection image(s) or a 3D map:"
                value = data_example.url
                image_url = st.text_input(label=label, value=value)
                data_all, apix = get_2d_image_from_url(image_url.strip())
            elif input_mode == "emd-xxxx":
                label = "Input an EMDB ID (emd-xxxx):"
                value = "emd-2699"
                emdid = st.text_input(label=label, value=value)
                data_all, apix = get_emdb_map_projections(emdid.strip())

        nz, ny, nx = data_all.shape
        if nz>1:
            image_index = st.slider(label=f"Choose an image (out of {nz}):", min_value=1, max_value=nz, value=1, step=1)
            image_index -= 1
        else:
            image_index = 0
        data = data_all[image_index]
        ny, nx = data.shape
        original_image = st.empty()
        with original_image:
            st.image(data, width=min(600, nx), caption=f"Orignal image ({nx}x{ny})", clamp=True)

        with st.beta_expander(label="Transpose/Rotate/Shift the image", expanded=False):
            is_pwr = st.checkbox(label="Input image is power spectra", value=False)
            transpose = st.checkbox(label='Transpose the image', value=False)
            if transpose:
                data = data.T
                ny, nx = data.shape
            angle_auto, dx_auto = auto_vertical_center(data)
            angle = st.number_input('Rotate (°)', value=-angle_auto, min_value=-180., max_value=180., step=1.0)
            dx = st.number_input('Shift along X-dim (Å)', value=dx_auto*apix, min_value=-nx*apix, max_value=nx*apix, step=1.0)
        
        rotated_image = st.empty()
        if angle or dx:
            data = rotate_shift_image(data, angle=-angle, post_shift=(0, dx/apix), order=3)
            with rotated_image:
                st.image(data, width=min(600, nx), caption="Transformed image", clamp=True)

        radius_auto, mask_radius_auto = estimate_radial_range(data, thresh_ratio=0.1)

        plotx = st.empty()
        mask_radius = st.number_input('Mask radius (Å)', value=mask_radius_auto*apix, min_value=1.0, max_value=nx/2*apix, step=1.0, format="%.1f")

        fraction_x = mask_radius/(nx//2*apix)
        data = tapering(data, fraction_start=[0.9, fraction_x], fraction_slope=0.1)
        with rotated_image:
            st.image(data, width=min(600, nx), caption="Transformed image", clamp=True)

        ploty = st.empty()
        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu). Report problems to Wen Jiang (jiang12 at purdue.edu)*")

    with col2:
        apix = st.number_input('Pixel size (Å/pixel)', value=apix, min_value=0.1, max_value=10., step=0.01, format="%.4f")
        pitch_or_twist = st.beta_container()
        rise = st.number_input('Rise (Å)', value=data_example.rise, min_value=-180.0, max_value=180.0, step=1.0, format="%.3f")
        csym = st.number_input('Csym', value=data_example.csym, min_value=1, max_value=16, step=1)

        radius = st.number_input('Radius (Å)', value=radius_auto*apix, min_value=10.0, max_value=1000.0, step=10., format="%.1f")
        
        tilt = st.number_input('Out-of-plane tilt (°)', value=0.0, min_value=-90.0, max_value=90.0, step=1.0)
        cutoff_res_x = st.number_input('Limit FFT X-dim to resolution (Å)', value=round(3*apix, 0), min_value=2*apix, step=1.0)
        cutoff_res_y = st.number_input('Limit FFT Y-dim to resolution (Å)', value=round(3*apix, 0), min_value=2*apix, step=1.0)
        pnx = st.number_input('FFT X-dim size (pixels)', value=max(min(nx,ny), 512), min_value=min(nx, 128), step=2)
        pny = st.number_input('FFT Y-dim size (pixels)', value=max(max(nx,ny), 1024), min_value=min(ny, 512), step=2)
        st.subheader("Simulate the helix with Gaussians")
        ball_radius = st.number_input('Gaussian radius (Å)', value=0.0, min_value=0.0, max_value=radius, step=5.0, format="%.1f")
        show_simu = True if ball_radius > 0 else False
        if show_simu:
            az = st.number_input('Azimuthal angle (°)', value=0, min_value=0, max_value=360, step=1, format="%.1f")
            noise = st.number_input('Noise (sigma)', value=0., min_value=0., step=1., format="%.1f")

    if not is_pwr:
        with ploty:
            y = np.arange(-ny//2, ny//2)*apix
            xmax = np.max(data, axis=1)
            xmean = np.mean(data, axis=1)

            tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
            tooltips = [("Y", "@y{0.0}Å")]
            p = figure(x_axis_label="pixel value", y_axis_label="y (Å)", frame_height=ny, tools=tools, tooltips=tooltips)
            p.line(xmax, y, line_width=2, color='red', legend_label="max")
            p.line(xmean, y, line_width=2, color='blue', legend_label="mean")
            p.legend.location = "top_right"
            p.legend.click_policy="hide"
            from bokeh import events
            from bokeh.models import CustomJS
            toggle_legend_js_y = CustomJS(args=dict(leg=p.legend[0]), code="""
                if (leg.visible) {
                    leg.visible = false
                    }
                else {
                    leg.visible = true
                }
            """)
            p.js_on_event(events.DoubleTap, toggle_legend_js_y)
            st.bokeh_chart(p, use_container_width=True)

        with plotx:
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
            from bokeh.models import Span
            rmin_span = Span(location=-mask_radius, dimension='height', line_color='green', line_dash='dashed', line_width=3)
            rmax_span = Span(location=mask_radius, dimension='height', line_color='green', line_dash='dashed', line_width=3)
            p.add_layout(rmin_span)
            p.add_layout(rmax_span)
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
            p.js_on_event(events.DoubleTap, toggle_legend_js_x)
            st.bokeh_chart(p, use_container_width=True)

    with col3:
        st.subheader("Display:")
        show_pitch = st.checkbox(label="Pitch", value=True)
        with pitch_or_twist:
            if show_pitch:
                pitch = st.number_input('Pitch (Å)', value=data_example.pitch, min_value=1.0, max_value=max(pny,pnx)*apix, step=1.0, format="%.2f")
                if pitch < cutoff_res_y:
                    st.warning(f"pitch is too small. it should be > {cutoff_res_y} (Limit FFT X-dim to resolution (Å))")
                    return
                twist = 360./(pitch/rise)
                st.markdown(f"*(twist = {twist:.2f} Å)*")
            else:
                twist = st.number_input('Twist (°)', value=data_example.twist, min_value=-180.0, max_value=180.0, step=1.0, format="%.2f")
                pitch = (360./twist)*rise
                st.markdown(f"*(pitch = {pitch:.2f} Å)*")
        show_pwr = st.checkbox(label="PS", value=True)
        if show_pwr:
            if is_pwr:
                show_phase = False
                show_phase_diff = False
            else:
                show_phase = st.checkbox(label="Phase", value=False)
                show_phase_diff = st.checkbox(label="PD", value=True)
            show_yprofile = st.checkbox(label="YP", value=False)
            show_pseudocolor = st.checkbox(label="Color", value=True)
            show_LL = st.checkbox(label="LL", value=True)
            if show_LL:
                m_groups = compute_layer_line_positions(twist=twist, rise=rise, csym=csym, radius=radius, tilt=tilt, cutoff_res=cutoff_res_y)
                ng = len(m_groups)
                st.subheader("m=")
                show_choices = {}
                lgs = sorted(m_groups.keys())[::-1]
                for lgi, lg in enumerate(lgs):
                    value = True if lg in [0, 1] else False
                    show_choices[lg] = st.checkbox(label=str(lg), value=value)

    if show_simu:
        proj = simulate_helix(twist, rise, csym, helical_radius=radius, ball_radius=ball_radius, 
                ny=data.shape[0], nx=data.shape[1], apix=apix, tilt=tilt, az0=az)
        if noise>0:
            sigma = np.std(proj[np.nonzero(proj)])
            proj = proj + np.random.normal(loc=0.0, scale=noise*sigma, size=proj.shape)
        proj = tapering(proj, fraction_start=[0.7, 0], fraction_slope=0.1)
        if transpose or angle or dx:
            image_container = rotated_image
            image_label = "Rotated image"
        else:
            image_container = original_image
            image_label = "Original image"
        with image_container:
            st.image([data, proj], width=data.shape[1], caption=[image_label, "Simulated"], clamp=True)

    with col4:
        if not show_pwr: return

        fig = None
        fig_phase = None
        fig_y = None
        fig_proj = None
        fig_proj_phase = None

        if is_pwr:
            pwr = resize_rescale_power_spectra(data, nyquist_res=2*apix, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                    output_size=(pny, pnx), low_pass_fraction=0.2, high_pass_fraction=0.004)
            phase = None
        else:
            pwr, phase = compute_power_spectra(data, apix=apix, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                    output_size=(pny, pnx), low_pass_fraction=0.2, high_pass_fraction=0.004)

        ny, nx = pwr.shape
        dsy = 1/(ny//2*cutoff_res_y)
        dsx = 1/(nx//2*cutoff_res_x)
        x_range = (-nx//2*dsx, nx//2*dsx)
        y_range = (-ny//2*dsy, ny//2*dsy)

        sy, sx = np.meshgrid(np.arange(-ny//2, ny//2)*dsy, np.arange(-nx//2, nx//2)*dsx, indexing='ij', copy=False)
        sy = sy.astype(np.float16)
        sx = sx.astype(np.float16)
        resx = np.abs(1./sx).astype(np.float16)
        resy = np.abs(1./sy).astype(np.float16)
        res  = 1./np.hypot(sx, sy).astype(np.float16)
        bessel = bessel_n_image(ny, nx, cutoff_res_x, cutoff_res_y, radius, tilt).astype(np.float16)

        from bokeh.models import LinearColorMapper
        tools = 'box_zoom,pan,reset,save,wheel_zoom'
        fig = figure(title_location="below", frame_width=nx, frame_height=ny, 
            x_axis_label=None, y_axis_label=None, x_range=x_range, y_range=y_range, tools=tools)
        fig.grid.visible = False
        fig.title.text = f"Power Spectra"
        fig.title.align = "center"
        fig.title.text_font_size = "20px"
        fig.yaxis.visible = False   # leaving yaxis on will make the crosshair x-position out of sync with other figures

        source_data = dict(image=[pwr.astype(np.float16)], x=[-nx//2*dsx], y=[-ny//2*dsy], dw=[nx*dsx], dh=[ny*dsy], resx=[resx], resy=[resy], res=[res], bessel=[bessel])
        if show_phase: source_data["phase"] = [np.fmod(np.rad2deg(phase)+360, 360).astype(np.float16)]
        from bokeh.models import LinearColorMapper
        palette = 'Viridis256' if show_pseudocolor else 'Greys256'
        color_mapper = LinearColorMapper(palette=palette)    # Greys256, Viridis256
        image = fig.image(source=source_data, image='image', color_mapper=color_mapper, x='x', y='y', dw='dw', dh='dh')
        # add hover tool only for the image
        from bokeh.models.tools import HoverTool
        tooltips = [("Res", "@resÅ"), ('Res y', '@resyÅ'), ('Res x', '@resxÅ'), ('Jn', '@bessel'), ('Amp', '@image')]
        if show_phase: tooltips.append(("Phase", "@phase°"))
        image_hover = HoverTool(renderers=[image], tooltips=tooltips)
        fig.add_tools(image_hover)

        # create a linked crosshair tool among the figures
        from bokeh.models import CrosshairTool
        crosshair = CrosshairTool(dimensions="both")
        crosshair.line_color = 'red'
        fig.add_tools(crosshair)

        if show_phase_diff:
            if nx%2:
                phase_diff = phase - phase[:, ::-1]
            else:
                phase_diff = phase * 1.0
                phase_diff[:, 0] = np.pi/2
                phase_diff[:, 1:] -= phase_diff[:, 1:][:, ::-1]
            phase_diff = np.rad2deg(np.arccos(np.cos(phase_diff)))   # set the range to [0, 180]. 0 -> even order, 180 - odd order
            
            fig_phase = figure(title_location="below", frame_width=nx, frame_height=ny, 
                x_axis_label=None, y_axis_label=None, x_range=fig.x_range, y_range=fig.y_range, y_axis_location = "right",
                tools=tools)
            fig_phase.grid.visible = False
            fig_phase.title.text = f"Phase Diff Across Meridian"
            fig_phase.title.align = "center"
            fig_phase.title.text_font_size = "20px"
            if show_simu:
                fig_phase.yaxis.visible = False

            source_data["image"] = [phase_diff.astype(np.float16)]
            phase_image = fig_phase.image(source=source_data, image='image', color_mapper=color_mapper,
                        x='x', y='y', dw='dw', dh='dh'
                    )
            # add hover tool only for the image
            tooltips = [("Res", "@resÅ"), ('Res y', '@resyÅ'), ('Res x', '@resxÅ'), ('Jn', '@bessel'), ('Phase Diff', '@image°')]
            phase_hover = HoverTool(renderers=[phase_image], tooltips=tooltips)
            fig_phase.add_tools(phase_hover)
            fig_phase.add_tools(crosshair)

        if show_simu and show_pwr:
            proj_pwr, proj_phase = compute_power_spectra(proj, apix=apix, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                    output_size=(pny, pnx), low_pass_fraction=0.2, high_pass_fraction=0.004)

            fig_proj = figure(title_location="below", frame_width=nx, frame_height=ny, 
                x_axis_label=None, y_axis_label=None, 
                x_range=x_range, y_range=y_range, y_axis_location = "right",
                tools=tools)
            fig_proj.grid.visible = False
            fig_proj.title.text = f"Simulated Power Spectra"
            fig_proj.title.align = "center"
            fig_proj.title.text_font_size = "20px"
            if show_phase_diff:
                fig_proj.yaxis.visible = False

            source_data["image"] = [proj_pwr.astype(np.float16)]
            if show_phase: source_data["phase"] = [np.fmod(np.rad2deg(proj_phase)+360, 360).astype(np.float16)]
            proj_image = fig_proj.image(source=source_data, image='image', color_mapper=color_mapper,
                        x='x', y='y', dw='dw', dh='dh'
                    )
            # add hover tool only for the image
            tooltips = [("Res", "@resÅ"), ('Res y', '@resyÅ'), ('Res x', '@resxÅ'), ('Jn', '@bessel'), ('Amp', '@image')]
            if show_phase: tooltips.append(("Phase", "@phase"))
            image_hover = HoverTool(renderers=[proj_image], tooltips=tooltips)
            fig_proj.add_tools(image_hover)
            fig_proj.add_tools(crosshair)

            if show_phase_diff:
                if nx%2:
                    phase_diff = proj_phase - proj_phase[:, ::-1]
                else:
                    phase_diff = proj_phase * 1.0
                    phase_diff[:, 0] = np.pi/2
                    phase_diff[:, 1:] -= phase_diff[:, 1:][:, ::-1]
                phase_diff = np.rad2deg(np.arccos(np.cos(phase_diff)))   # set the range to [0, 180]. 0 -> even order, 180 - odd order
                
                fig_proj_phase = figure(title_location="below", frame_width=nx, frame_height=ny, 
                    x_axis_label=None, y_axis_label=None, 
                    x_range=x_range, y_range=y_range, y_axis_location = "right",
                    tools=tools)
                fig_proj_phase.grid.visible = False
                fig_proj_phase.title.text = f"Phase Diff Across Meridian"
                fig_proj_phase.title.align = "center"
                fig_proj_phase.title.text_font_size = "20px"

                source_data["image"] = [phase_diff.astype(np.float16)]
                phase_image = fig_proj_phase.image(source=source_data, image='image', color_mapper=color_mapper,
                            x='x', y='y', dw='dw', dh='dh'
                        )
                # add hover tool only for the image
                tooltips = [("Res", "@resÅ"), ('Res y', '@resyÅ'), ('Res x', '@resxÅ'), ('Jn', '@bessel'), ('Phase Diff', '@image°')]
                phase_hover = HoverTool(renderers=[phase_image], tooltips=tooltips)
                fig_proj_phase.add_tools(phase_hover)
                fig_proj_phase.add_tools(crosshair)

        if show_pwr and show_LL:
            if max(m_groups[0]["LL"][0])>0:
                from bokeh.palettes import viridis, gray
                if show_pseudocolor:
                    ll_colors = gray(ng*2)[::-1]
                else:
                    ll_colors = viridis(ng*2)[::-1]
                x, y = m_groups[0]["LL"]
                tmp_x = np.sort(np.unique(x))
                width = np.mean(tmp_x[1:]-tmp_x[:-1])
                tmp_y = np.sort(np.unique(y))
                height = np.mean(tmp_y[1:]-tmp_y[:-1])/3
                for mi, m in enumerate(m_groups.keys()):
                    if not show_choices[m]: continue
                    x, y = m_groups[m]["LL"]
                    color = ll_colors[mi]
                    if fig:
                        fig.ellipse(x, y, width=width, height=height, line_width=4, line_color=color, fill_alpha=0)
                    if fig_proj:
                        fig_proj.ellipse(x, y, width=width, height=height, line_width=4, line_color=color, fill_alpha=0)
            else:
                st.warning(f"No off-equator layer lines to draw for Pitch={pitch:.2f} Csym={csym} combinations. Consider increasing Pitch or reducing Csym")

        if show_yprofile:
            y=np.arange(-ny//2, ny//2)*dsy
            ll_profile = np.max(pwr, axis=1)
            ll_profile /= ll_profile.max()
            source_data = dict(ll=ll_profile, y=y, resy=np.abs(1./y))
            if fig_proj:
                ll_profile_proj = np.mean(proj_pwr, axis=1)
                ll_profile_proj /= ll_profile_proj.max()
                source_data["ll_proj"] = ll_profile_proj           

            tools = 'box_zoom,hover,pan,reset,save,wheel_zoom'
            tooltips = [('Res y', '@resyÅ'), ('Amp', '$x')]
            fig_y = figure(frame_width=nx//2, frame_height=ny, y_range=fig.y_range, y_axis_location = "right", 
                title=None, tools=tools, tooltips=tooltips)
            fig_y.line(source=source_data, x='ll', y='y', line_width=2, color='blue')
            if fig_proj:
                fig_y.line(source=source_data, x='ll_proj', y='y', line_width=2, color='red')
            fig_y.add_tools(crosshair)
            if show_phase_diff or show_simu:
                fig_y.yaxis.visible = False

        if fig_phase or fig_proj or fig_proj_phase or fig_y:
            figs = [f for f in [fig, fig_y, fig_phase, fig_proj, fig_proj_phase] if f]
            from bokeh.layouts import gridplot
            fig = gridplot(children=[figs], toolbar_location='right')
        
        st.bokeh_chart(fig, use_container_width=False)

        return

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
def resize_rescale_power_spectra(data, nyquist_res, cutoff_res=None, output_size=None, low_pass_fraction=0, high_pass_fraction=0):
    from scipy.ndimage.interpolation import map_coordinates
    ny, nx = data.shape
    ony, onx = output_size
    res_y, res_x = cutoff_res
    Y, X = np.meshgrid(np.arange(ony, dtype=np.float32)-ony//2, np.arange(onx, dtype=np.float32)-onx//2, indexing='ij')
    Y = Y/(ony//2) * nyquist_res/res_y * ny//2 + ny//2
    X = X/(onx//2) * nyquist_res/res_x * nx//2 + nx//2
    pwr = map_coordinates(data, (Y.flatten(), X.flatten()), order=3, mode='constant').reshape(Y.shape)
    pwr = np.log(np.abs(pwr))
    if 0<low_pass_fraction<1 or 0<high_pass_fraction<1:
        pwr = low_high_pass_filter(pwr, low_pass_fraction=low_pass_fraction, high_pass_fraction=high_pass_fraction)
    pwr = normalize(pwr, percentile=(0, 100))
    return pwr

@st.cache(persist=True, show_spinner=False)
def compute_power_spectra(data, apix, cutoff_res=None, output_size=None, low_pass_fraction=0, high_pass_fraction=0):
    fft = fft_rescale(data, apix=apix, cutoff_res=cutoff_res, output_size=output_size)
    fft = np.fft.fftshift(fft)  # shift fourier origin from corner to center

    pwr = np.log(np.abs(fft))
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
def tapering(data, fraction_start=[0, 0], fraction_slope=0.1):
    fy, fx = fraction_start
    if not (0<fy<1 and 0<fx<1): return data
    ny, nx = data.shape
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
    data2 = data * filter
    return data2

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
    background = np.mean(image[[0,1,2,-3,-2,-1],[0,1,2,-3,-2,-1]])
    thresh = (image.max()-background) * 0.2 + background
    image_work = 1.0 * image
    image_work[image<thresh] = 0

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
    n = len(y)
    from scipy.ndimage.measurements import center_of_mass
    cx = int(round(center_of_mass(y)[0]))
    max_shift = abs((cx-n//2)*2)+3

    import scipy.interpolate as interpolate
    x = np.arange(3*n)
    f = interpolate.interp1d(x, np.tile(y, 3), kind='cubic')    # avoid out-of-bound errors
    def score_shift(dx):
        x_tmp = x[n:2*n]-dx
        tmp = f(x_tmp)
        err = np.sum(np.abs(tmp-tmp[::-1]))
        return err
    res = minimize_scalar(score_shift, bounds=(-max_shift, max_shift), method='bounded')
    dx = res.x + (0.0 if n%2 else 0.5)
    return angle, dx

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
def normalize(data, percentile=(0, 100)):
    p0, p1 = percentile
    vmin, vmax = sorted(np.percentile(data, (p0, p1)))
    data2 = (data-vmin)/(vmax-vmin)
    return data2

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
def get_emdb_map_projections(emdid):
    emdid_number = emdid.lower().split("emd-")[-1]
    url = f"ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdid_number}/map/emd_{emdid_number}.map.gz"
    return get_2d_image_from_url(url, as_3d=True)

@st.cache(persist=True, show_spinner=False)
def get_2d_image_from_url(url, as_3d=False):
    ds = np.DataSource(None)
    fp=ds.open(url)
    return get_2d_image_from_file(fp.name, as_3d=as_3d)

@st.cache(persist=True, show_spinner=False)
def get_2d_image_from_file(filename, as_3d=False):
    import mrcfile
    with mrcfile.open(filename) as mrc:
        data = mrc.data * 1.0
        apix = mrc.voxel_size.x.item()
    nz, ny, nx = data.shape
    if nz>1 and as_3d:
        projs = np.zeros((2, nz, nx))
        for ai, axis in enumerate([1, 2]):
            proj = data.mean(axis=axis)
            proj = normalize(proj, percentile=(0, 100))
            projs[ai] = proj
        return projs, apix
    else:
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

@st.cache(persist=True, show_spinner=False)
def setup_anonymous_usage_tracking():
    try:
        import os, stat
        index_file = os.path.dirname(st.__file__) + "/static/index.html"
        os.chmod(index_file, stat.S_IRUSR|stat.S_IWUSR|stat.S_IROTH)
        with open(index_file, "r+") as fp:
            txt = fp.read()
            if txt.find("gtag.js")==-1:
                txt2 = txt.replace("<head>", '''<head><!-- Global site tag (gtag.js) - Google Analytics --><script async src="https://www.googletagmanager.com/gtag/js?id=G-8Z99BDVHTC"></script><script>window.dataLayer = window.dataLayer || [];function gtag(){dataLayer.push(arguments);}gtag('js', new Date());gtag('config', 'G-8Z99BDVHTC');</script>''')
                fp.seek(0)
                fp.write(txt2)
                fp.truncate()
    except:
        pass

if __name__ == "__main__":
    setup_anonymous_usage_tracking()
    main()
