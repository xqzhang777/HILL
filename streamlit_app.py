import streamlit as st
import numpy as np

def main():
    st.set_page_config(page_title="Helical Indexing", layout="wide")

    title = "Helical indexing using layer lines"
    st.title(title)

    col1, col2, col3, col4 = st.beta_columns((1.5, 0.75, 0.25, 3.5))

    with col1:
        with st.beta_expander(label="README", expanded=False):
            st.write("This Web app considers a biological helical structure as the product of a continous helix and a set of parallel planes, and based on the covolution theory, the Fourier Transform (FT) of a helical structure would be the convolution of the FT of the continous helix and the FT of the planes.  \nThe FT of a continous helix consists of equally spaced layer planes (3D) or layerlines (2D projection) that can be described by Bessel functions of increasing orders (0, +/-1, +/-2, ...) from the Fourier origin (i.e. equator). The spacing between the layer planes/lines is determined by the helical pitch (i.e. the shift along the helical axis for a 360 ° turn of the helix). If the structure has additional cyclic symmetry (for example, C6) around the helical axis, only the layer plane/line orders of integer multiplier of the symmetry (e.g. 0, +/-6, +/-12, ...) are visible. The primary peaks of the layer lines in the power spectra form a pattern similar to a X symbol.  \nThe FT of the parallel planes consists of equally spaced points along the helical axis (i.e. meridian) with the spacing being determined by the helical rise.  \nThe convolution of these two components (X-shaped pattern of layer lines and points along the meridian) generates the layer line patterns seen in the power spectra of the projection images of helical structures. The helical indexing task is thus to identify the helical rise, pitch (or twist), and cyclic symmetry that would predict a layer line pattern to explain the observed the layer lines in the power spectra. This Web app allows you to interactively change the helical parameters and superimpose the predicted layer liines on the power spectra to complete the helical indexing task.  \n  \nPS: power spectra; YP: Y-axis power spectra profile; LL: layer lines; m: indices of the X-patterns along the meridian")
        
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
            st.image(data, width=nx, caption=f"Orignal image ({nx}x{ny})", clamp=True)
        with st.beta_expander(label="Rotate/shift the image", expanded=True):
            angle = st.number_input('Rotate (°)', value=0.0, min_value=-180., max_value=180., step=1.0)
            dx = st.number_input('Shift along X-dim (Å)', value=0.0, min_value=-nx*apix, max_value=nx*apix, step=1.0)
        rotated_image = st.empty()
        if angle or dx:
            data = rotate_shift_image(data, angle, dx/apix)
            with rotated_image:
                st.image(data, width=nx, caption="Rotated image", clamp=True)
        plotx = st.empty()
        ploty = st.empty()
        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu). Report problems to Wen Jiang (jiang12 at purdue.edu)*")

    with col2:
        apix = st.number_input('Pixel size (Å/pixel)', value=apix, min_value=0.1, max_value=10., step=0.01, format="%.4f")
        pitch_or_twist = st.beta_container()
        rise = st.number_input('Rise (Å)', value=data_example.rise, min_value=-180.0, max_value=180.0, step=1.0, format="%.2f")
        csym = st.number_input('Csym', value=data_example.csym, min_value=1, max_value=16, step=1)
        radius = st.number_input('Radius (Å)', value=data_example.diameter/2., min_value=10.0, max_value=1000.0, step=10., format="%.1f")
        tilt = st.number_input('Out-of-plane tilt (°)', value=0.0, min_value=-90.0, max_value=90.0, step=1.0)
        cutoff_res_x = st.number_input('Limit FFT X-dim to resolution (Å)', value=round(3*apix, 0), min_value=2*apix, step=1.0)
        cutoff_res_y = st.number_input('Limit FFT Y-dim to resolution (Å)', value=round(3*apix, 0), min_value=2*apix, step=1.0)
        pnx = st.number_input('FFT X-dim size (pixels)', value=max(nx, 512), min_value=min(nx, 128), step=2)
        pny = st.number_input('FFT Y-dim size (pixels)', value=max(ny, 1024), min_value=min(ny, 512), step=2)
        st.subheader("Simulate the helix with Gaussians")
        ball_radius = st.number_input('Gaussian radius (Å)', value=0.0, min_value=0.0, max_value=radius, step=5.0, format="%.1f")
        az = st.number_input('Azimuthal angle (°)', value=0, min_value=0, max_value=360, step=1, format="%.1f")

    with ploty:
        y = np.arange(-ny//2, ny//2)*apix
        xmax = np.max(data, axis=1)
        xmean = np.mean(data, axis=1)

        from bokeh.plotting import figure
        tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
        p = figure(x_axis_label="pixel value", y_axis_label="y (Å)", frame_height=ny, tools=tools)
        p.line(xmax, y, line_width=2, color='red')
        p.line(xmean, y, line_width=2, color='blue')
        st.bokeh_chart(p, use_container_width=True)

    with plotx:
        x = np.arange(-nx//2, nx//2)*apix
        ymax = np.max(data, axis=0)
        ymean = np.mean(data, axis=0)

        from bokeh.plotting import figure
        tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
        p = figure(x_axis_label="x (Å)", y_axis_label="pixel value", frame_height=ny, tools=tools)
        p.line(x, ymax, line_width=2, color='red', legend_label="max")
        p.line(x, ymean, line_width=2, color='blue', legend_label="mean")
        p.legend.location = "top_right"
        st.bokeh_chart(p, use_container_width=True)

    with col3:
        st.subheader("Display:")
        show_pitch = st.checkbox(label="Pitch", value=True)
        with pitch_or_twist:
            if show_pitch:
                pitch = st.number_input('Pitch (Å)', value=data_example.pitch, min_value=1.0, max_value=pny//2*apix, step=1.0, format="%.2f")
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
        show_yprofile = st.checkbox(label="YP", value=True)
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
    
    if ball_radius>0:
        proj = simulate_helix(twist, rise, csym, helical_radius=radius, ball_radius=ball_radius, 
                ny=data.shape[0], nx=data.shape[1], apix=apix, tilt=tilt, az0=az)
        proj = tapering(proj, fraction=0.3)
        if angle or dx:
            image_container = rotated_image
            image_label = "Rotated image"
        else:
            image_container = original_image
            image_label = "Original image"
        with image_container:
            st.image([data, proj], width=data.shape[1], caption=[image_label, "Simulated"], clamp=True)

    with col4:
        if not show_pwr:
            return

        data = tapering(data, fraction=0.1)
        pwr = compute_power_spectra(data, apix=apix, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                output_size=(pny, pnx), low_pass_fraction=0.2, high_pass_fraction=0.004)

        ny, nx = pwr.shape
        dsy = 1/(ny//2*cutoff_res_y)
        dsx = 1/(nx//2*cutoff_res_x)

        from bokeh.models import LinearColorMapper
        tools = 'box_zoom,pan,reset,save,wheel_zoom'
        fig = figure(title_location="below", frame_width=nx, frame_height=ny, 
            x_axis_label=None, y_axis_label=None, 
            x_range=(-nx//2*dsx, nx//2*dsx), y_range=(-ny//2*dsy, ny//2*dsy), 
            tools=tools)
        fig.grid.visible = False
        fig.title.text = f"Power Spectra"
        fig.title.align = "center"
        fig.title.text_font_size = "20px"

        sy, sx = np.meshgrid(np.arange(-ny//2, ny//2)*dsy, np.arange(-nx//2, nx//2)*dsx, indexing='ij', copy=False)
        resx = np.abs(1./sx)
        resy = np.abs(1./sy)
        res  = 1./np.hypot(sx, sy)

        source_data = dict(image=[pwr], x=[-nx//2*dsx], y=[-ny//2*dsy], dw=[nx*dsx], dh=[ny*dsy], resx=[resx], resy=[resy], res=[res])
        from bokeh.models import LinearColorMapper
        palette = 'Viridis256' if show_pseudocolor else 'Greys256'
        color_mapper = LinearColorMapper(palette=palette)    # Greys256, Viridis256
        image = fig.image(source=source_data, image='image', color_mapper=color_mapper,
                    x='x', y='y', dw='dw', dh='dh'
                )
        # add hover tool only for the image
        from bokeh.models.tools import HoverTool
        tooltips = [("Res", "@resÅ"), ('Res x', '@resxÅ'), ('Res y', '@resyÅ'), ('PS', '@image')]
        image_hover = HoverTool(renderers=[image], tooltips=tooltips)
        fig.add_tools(image_hover)

        # create a linked crosshair tool among the figures
        from bokeh.models import CrosshairTool
        crosshair = CrosshairTool(dimensions="both")
        crosshair.line_color = 'red'
        fig.add_tools(crosshair)

        if ball_radius>0:
            proj_pwr = compute_power_spectra(proj, apix=apix, cutoff_res=(cutoff_res_y, cutoff_res_x), 
                    output_size=(pny, pnx), low_pass_fraction=0.2, high_pass_fraction=0.004)

            fig_proj = figure(title_location="below", frame_width=nx, frame_height=ny, 
                x_axis_label=None, y_axis_label=None, 
                x_range=fig.x_range, y_range=fig.y_range, y_axis_location = "right",
                tools=tools)
            fig_proj.grid.visible = False
            fig_proj.title.text = f"Simulated Power Spectra"
            fig_proj.title.align = "center"
            fig_proj.title.text_font_size = "20px"

            source_data["image"] = [proj_pwr]
            proj_image = fig_proj.image(source=source_data, image='image', color_mapper=color_mapper,
                        x='x', y='y', dw='dw', dh='dh'
                    )
            # add hover tool only for the image
            image_hover = HoverTool(renderers=[proj_image], tooltips=tooltips)
            fig_proj.add_tools(image_hover)
            fig_proj.add_tools(crosshair)
        else:
            fig_proj = None

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
            tooltips = [('Res y', '@resyÅ'), ('PS', '$x')]
            fig_y = figure(frame_width=nx//2, frame_height=ny, y_range=fig.y_range, y_axis_location = "right", 
                title=None, tools=tools, tooltips=tooltips)
            fig_y.line(source=source_data, x='ll', y='y', line_width=2, color='blue')
            if fig_proj:
                fig_y.line(source=source_data, x='ll_proj', y='y', line_width=2, color='red')
            fig_y.add_tools(crosshair)
            if ball_radius:
                fig_y.yaxis.visible = False
        else:
            fig_y = None

        if show_LL:
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
                fig.ellipse(x, y, width=width, height=height, line_width=4, line_color=color, fill_alpha=0)
                if fig_proj:
                    fig_proj.ellipse(x, y, width=width, height=height, line_width=4, line_color=color, fill_alpha=0)

        if fig_proj or fig_y:
            figs = [f for f in [fig, fig_y, fig_proj] if f]
            from bokeh.layouts import gridplot
            fig = gridplot([figs], toolbar_location='right')
        
        st.bokeh_chart(fig, use_container_width=False)

        return

@st.cache(persist=True, show_spinner=False)
def simulate_helix(twist, rise, csym, helical_radius, ball_radius, ny, nx, apix, tilt=0, az0=None):
    def simulate_projection(centers, sigma, ny, nx, apix):
        sigma2 = sigma*sigma
        d = np.zeros((ny, nx))
        Y, X = np.meshgrid(np.arange(0, ny, dtype=np.float)-ny//2, np.arange(0, nx, dtype=np.float)-nx//2, indexing='ij')
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
        
        centers = np.zeros(((2*imax+1)*csym, 3), dtype=np.float)
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
        m_max = int(np.floor(np.abs(rise/cutoff_res)))
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
        ll_i = np.array([i for i in range(ll_i_bottom, ll_i_top+1) if not i%csym])
        sy = sy0 + ll_i * ds_p
        sx = peak_sx(bessel_order=ll_i, radius=radius)
        if tilt:
            sy = np.array(sy) * tf
            sx = np.sqrt(np.power(np.array(sx), 2) - np.power(sy*tf2, 2))
            sx[np.isnan(sx)] = 1e-6
        px  = list(sx) + list(-sx)
        py  = list(sy) + list(sy)
        d["LL"] = (px, py)

        m_groups[m[mi]] = d
    return m_groups

@st.cache(persist=True, show_spinner=False)
def compute_power_spectra(data, apix, cutoff_res=None, output_size=None, low_pass_fraction=0, high_pass_fraction=0):
    fft = fft_rescale(data, apix=apix, cutoff_res=cutoff_res, output_size=output_size)
    pwr = np.log(np.abs(np.fft.fftshift(fft)))
    if 0<low_pass_fraction<1 or 0<high_pass_fraction<1:
        fft = np.fft.fft2(pwr)
        ny, nx = fft.shape
        Y, X = np.meshgrid(np.arange(ny, dtype=np.float)-ny//2, np.arange(nx, dtype=np.float)-nx//2, indexing='ij')
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
        pwr = np.abs(np.fft.ifft2(fft))
    pwr = normalize(pwr, percentile=(0, 100))
    return pwr

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

    return fft

@st.cache(persist=True, show_spinner=False)
def tapering(data, fraction=0):
    if not (0<fraction<1): return data
    ny, nx = data.shape
    Y, _ = np.meshgrid(np.arange(0, ny, dtype=np.float)-ny//2, np.arange(0, nx, dtype=np.float)-nx//2, indexing='ij')
    Y = np.abs(Y / (ny//2))
    Y = (Y-(1-fraction))/fraction
    inner = Y<0
    Y = (1. + np.cos(Y*np.pi))/2.0
    Y[inner]=1
    data2 = data * Y
    return data2

@st.cache(persist=True, show_spinner=False)
def rotate_shift_image(data, angle=0, dx=0, dy=0):
    if angle==0 and dx==0 and dy==0: return data
    ny, nx = data.shape
    center = np.array((ny//2, nx//2), dtype=np.float)
    ang = np.deg2rad(angle)
    m = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    offset = center.T - np.dot(m, center.T) - np.array([-dy, dx]).T
    from scipy.ndimage import affine_transform
    ret = affine_transform(data, matrix=m, offset=offset, mode='nearest')
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
def get_2d_image_from_url(url):
    known_urls = [ example.url for example in data_examples]
    if url in known_urls or url.endswith(".class_averages.blob") or url.endswith(".templates_selected.blob"): # cryosparc 2D class averages
        data, apix = get_2d_image_from_url(url)
    elif (url.find("emd_")!=-1 and url.endswith(".map.gz")) or url.endswith(".map") or url.endswith(".mrc"):
        data, apix = get_2d_image_from_url(url)
    else:
        from skimage.io import imread
        data = imread(url)
        data = np.expand_dims(data, axis=0)
        apix = 1.0
    return data, apix

@st.cache(persist=True, show_spinner=False)
def get_emdb_map_projections(emdid):
    emdid_number = emdid.lower().split("emd-")[-1]
    url = f"ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdid_number}/map/emd_{emdid_number}.map.gz"
    return get_2d_image_from_url(url)

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
        is2d = mrc.is_image_stack() or mrc.is_single_image()
    nz, ny, nx = data.shape
    assert( nx==ny )

    if is2d:
        return data, apix
    else:
        projs = np.zeros((2, nz, nx))
        for ai, axis in enumerate([1, 2]):
            proj = data.mean(axis=axis)
            proj = normalize(proj, percentile=(0, 100))
            projs[ai] = proj
        return projs, apix
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
