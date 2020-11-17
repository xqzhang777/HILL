import streamlit as st
import numpy as np

def main():
    st.set_page_config(page_title="Helical Indexing", layout="wide")

    title = "Helical indexing using layer lines"
    st.title(title)

    col1, col2, col3, col4 = st.beta_columns((1.5, 0.75, 0.25, 3.5))

    with col1:
        with st.beta_expander(label="README", expanded=False):
            st.write("This Web app considers a biological helical structure as the product of a continous helix and a set of parallel planes, and based on the covolution theory, the Fourier Transform (FT) of a helical structure would be the convolution of the FT of the continous helix and the FT of the planes.  \nThe FT of a continous helix consists of equally spaced layer planes (3D) or layer lines (2D projection) that can be described by Bessel functions of increasing orders (0, +/-1, +/-2, ...) from the Fourier origin (i.e. equator). The spacing between the layer planes/lines is determined by the helical pitch (i.e. the shift along the helical axis for a 360 degree turn of the helix). If the structure has additional cyclic symmetry (for example, C6) around the helical axis, only the layer plane/line orders of integer multiplier of the symmetry (e.g. 0, +/-6, +/-12, ...) are visible. The overall shape of the FT is similar to a X symbol.  \nThe FT of the parallel planes consists of equally spaced points along the helical axis (i.e. meridian) with the spacing being determined by the helical rise.  \nThe convolution of these two components (X-shaped layer lines and points along the meridian) generates the layer line patterns seen in the power spectra of the projection images of helical structures. The helical indexing task is thus to identify the helical rise, pitch (or twist), and cyclic symmetry that would predict a lay line pattern to explain the observed the layer lines in the power spectra. This Web app allows you to interactively change the helical parameters and superimpose the predicted layer liines on the power spectra to complete the helical indexing task.  \n  \nPS: power spectra; YP: Y-axis power spectra profile; X: the X pattern; LL: layer lines; m: indices of the X-patterns along the meridian")
        
        label = "Input a url of 2D projection image(s) or an EMDB 3D map:"
        image_url = st.text_input(label=label, value=data_example.url)
        image_url = image_url.strip()
        data_all, apix = get_2d_image(image_url)
        nz, ny, nx = data_all.shape
        if nz>1:
            image_index = st.slider(label=f"Choose an image (out of {nz}):", min_value=1, max_value=nz, value=1, step=1)
            image_index -= 1
        else:
            image_index = 0
        data = data_all[image_index]
        ny, nx = data.shape
        st.image(data, width=nx, caption=f"Orignal image ({nx}x{ny})", clamp=[data.min(), data.max()])
        angle = st.number_input('Rotate (Degree)', value=0.0, min_value=-180., max_value=180., step=1.0)
        if angle!=0:
            data = rotate_image(data, angle)
            st.image(data, width=nx, caption="Rotated image", clamp=[data.min(), data.max()])
        ploty = st.empty()
        plotx = st.empty()
        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu). Report problems to Wen Jiang (jiang12 at purdue.edu)*")

    with col2:
        apix = st.number_input('Pixel size (Angstrom/pixel)', value=apix, min_value=0.1, max_value=10., step=0.01, format="%.4f")
        twist = st.number_input('Twist (degree)', value=data_example.twist, min_value=-180.0, max_value=180.0, step=1.0, format="%.2f")
        rise = st.number_input('Rise (Angstrom)', value=data_example.rise, min_value=-180.0, max_value=180.0, step=1.0, format="%.2f")
        csym = st.number_input('Csym', value=data_example.csym, min_value=1, max_value=16, step=1)
        radius = st.number_input('Radius (Angstrom)', value=data_example.diameter/2., min_value=10.0, max_value=1000.0, step=10., format="%.1f")
        pnx = st.number_input('X-dim padding (pixels)', value=max(nx, 512), min_value=nx, step=2)
        pny = st.number_input('Y-dim padding (pixels)', value=max(ny, 1024), min_value=ny, step=2)
        cutoff_res = st.number_input('Limit FFT to resolution (Angstrom)', value=max(rise*0.4, 2*apix), min_value=2*apix, step=1.0)

    with ploty:
        y = np.arange(-ny//2, ny//2)*apix
        xmax = np.max(data, axis=1)
        xmean = np.mean(data, axis=1)

        from bokeh.plotting import figure
        tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
        p = figure(x_axis_label="pixel value", y_axis_label="y (Angstrom)", frame_height=ny, tools=tools)
        p.line(xmax, y, line_width=2, color='red')
        p.line(xmean, y, line_width=2, color='blue')
        st.bokeh_chart(p, use_container_width=True)

    with plotx:
        x = np.arange(-nx//2, nx//2)*apix
        ymax = np.max(data, axis=0)
        ymean = np.mean(data, axis=0)

        from bokeh.plotting import figure
        tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
        p = figure(x_axis_label="x (Angstrom)", y_axis_label="pixel value", frame_height=ny, tools=tools)
        p.line(x, ymax, line_width=2, color='red', legend_label="max")
        p.line(x, ymean, line_width=2, color='blue', legend_label="mean")
        p.legend.location = "top_right"
        st.bokeh_chart(p, use_container_width=True)

    data_padded = pad_2d_image(data, pnx, pny)
    pwr = compute_power_spectra(data_padded, apix=apix, cutoff_res=cutoff_res, high_pass_fraction=0.004)
    m_groups = compute_layer_line_positions(twist=twist, rise=rise, csym=csym, radius=radius, cutoff_res=cutoff_res)
    ng = len(m_groups)

    with col3:
        st.subheader("Display:")
        show_pwr = st.checkbox(label="PS", value=True)
        show_yprofile = st.checkbox(label="YP", value=True)
        show_pseudocolor = st.checkbox(label="Color", value=True)
        show_X = st.checkbox(label="X", value=True)
        show_LL = st.checkbox(label="LL", value=True)
        st.subheader("m=")
        show_choices = {}
        lgs = sorted(m_groups.keys())[::-1]
        for lgi, lg in enumerate(lgs):
            value = True if lg in [0, 1] else False
            show_choices[lg] = st.checkbox(label=str(lg), value=value)
    
    with col4:
        ny, nx = pwr.shape
        dsy = 1/(ny//2*cutoff_res)
        dsx = 1/(nx//2*cutoff_res)

        if show_pwr:
            from bokeh.models import LinearColorMapper
            tools = 'box_zoom,crosshair,pan,reset,save,wheel_zoom'
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

        from bokeh.palettes import viridis, gray
        if show_pseudocolor:
            ll_colors = gray(ng*2)[::-1]
        else:
            ll_colors = viridis(ng*2)[::-1]

        if show_LL:
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

        if show_X:
            for mi, m in enumerate(m_groups.keys()):
                if not show_choices[m]: continue
                x, y = m_groups[m]["X"]
                color = ll_colors[mi]
                fig.multi_line(x, y, line_width=4, line_color=color, line_dash="dashed")

        if show_yprofile:
            y=np.arange(-ny//2, ny//2)*dsy
            ll_profile = np.mean(pwr, axis=1)             # entire layerline

            from bokeh.layouts import gridplot
            tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
            tooltips = [('Res y', '@resyÅ'), ('PS', '$x')]
            fig_y = figure(frame_width=nx//2, frame_height=ny, y_range=fig.y_range, title=None, tools=tools, tooltips=tooltips)
            source_data = dict(ll=ll_profile, y=y, resy=np.abs(1./y))
            fig_y.line(source=source_data, x='ll', y='y', line_width=2, color='blue')

            # link the crosshair tool
            from bokeh.models import CrosshairTool
            crosshair = CrosshairTool(dimensions="both")
            fig.add_tools(crosshair)
            fig_y.add_tools(crosshair)

            fig = gridplot([[fig, fig_y]], toolbar_location='right')
        
        st.bokeh_chart(fig, use_container_width=False)

        return

@st.cache(persist=True, show_spinner=False)
def compute_layer_line_positions(twist, rise, csym, radius, cutoff_res, m=[]):
    def pitch(twist, rise):
        p = np.abs(rise * 360./twist)   # pitch in Angstrom
        return p

    def X_line_slope(twist, rise, csym, radius):
        from scipy.special import jnp_zeros
        peak1 = jnp_zeros(csym, 1)[0] # first peak of first visible layerline (n=csym)
        sx = peak1/(2*np.pi*radius)
        p = pitch(twist, rise)
        sy = 1/p * csym
        slope = sy/sx # slope for the line from bottom/left to top/right
        return slope
    
    def sx_at_sy(sy, slope, intercept):
        sx = (np.array(sy)-intercept) / slope
        return sx

    if not m:
        m_max = int(np.floor(np.abs(rise/cutoff_res)))
        m = list(range(-m_max, m_max+1))
        m.sort(key=lambda x: (abs(x), x))   # 0, -1, 1, -2, 2, ...
    
    slope = X_line_slope(twist, rise, csym, radius)
    smax = 1./cutoff_res

    m_groups = {} # one group per m order
    for mi in range(len(m)):
        d = {}
        sy0 = m[mi] / rise

        # X pattern
        sy = [-smax, smax]
        sx = sx_at_sy(sy, slope=slope, intercept=sy0)
        x  = [sx, -sx]
        y  = [sy, sy]
        d["X"] = (x, y)

        # first peak positions of each layer line
        p = pitch(twist, rise)
        ds_p = 1/p
        ll_i_top = int((smax - sy0)/ds_p)
        ll_i_bottom = -int(np.abs(-smax - sy0)/ds_p)
        ll_i = np.array([i for i in range(ll_i_bottom, ll_i_top+1) if not i%csym])
        sy = sy0 + ll_i * ds_p
        sx = sx_at_sy(sy, slope=slope, intercept=sy0)
        px  = list(sx) + list(-sx)
        py  = list(sy) + list(sy)
        d["LL"] = (px, py)

        m_groups[m[mi]] = d
    return m_groups

@st.cache(persist=True, show_spinner=False)
def compute_power_spectra(data, apix, cutoff_res=0, high_pass_fraction=0):
    pwr = np.log(np.abs(np.fft.fftshift(np.fft.fft2(data))))
    if 0<high_pass_fraction<=1:
        fft = np.fft.fft2(pwr)
        ny, nx = fft.shape
        Y, X = np.meshgrid(np.arange(-ny//2, ny//2, dtype=np.float), np.arange(-nx//2, nx//2, dtype=np.float), indexing='ij')
        Y /= ny//2
        X /= nx//2
        f2 = np.log(2)/(high_pass_fraction**2)
        filter = 1.0 - np.exp(- f2 * Y**2) # Y-direction only
        fft *= np.fft.fftshift(filter)
        pwr = np.abs(np.fft.ifft2(fft))
    if cutoff_res>=2*apix:
        ny0, nx0 = pwr.shape
        scale_factor = cutoff_res/(2*apix)
        from skimage.transform import rescale
        pwr = rescale(pwr, scale_factor, order=3)
        ny, nx = pwr.shape
        sfi = int(round(scale_factor))
        pwr = pwr[ny//2-ny0//2:ny//2+ny0//2+sfi, nx//2-nx0//2:nx//2+nx0//2+sfi] 
    pwr = normalize(pwr, percentile=(0, 100))
    return pwr

@st.cache(persist=True, show_spinner=False)
def pad_2d_image(data, nx, ny):
    ny0, nx0 = data.shape
    px = max(0, (nx-nx0)//2)
    py = max(0, (ny-ny0)//2)
    pvx = data[:, [0,-1]].mean()
    pvy = data[[0,-1], :].mean()
    constant_values = (pvx+pvy)/2
    data2 = np.pad(data, pad_width=((py,py), (px,px)), mode='constant', constant_values=constant_values)    
    return data2

@st.cache(persist=True, show_spinner=False)
def rotate_image(data, angle=0):
    if angle==0: return data
    from skimage.transform import rotate
    data = rotate(data, angle)  
    return data

@st.cache(persist=True, show_spinner=False)
def normalize(data, percentile=(0, 100)):
    p0, p1 = percentile
    vmin, vmax = sorted(np.percentile(data, (p0, p1)))
    data2 = (data-vmin)/(vmax-vmin)
    return data2

@st.cache(persist=True, show_spinner=False)
def get_3d_map_projections(url):
    ds = np.DataSource(None)
    fp=ds.open(url)
    import mrcfile
    with mrcfile.open(fp.name) as mrc:
        data = mrc.data
        apix = mrc.voxel_size.x.item()
    nz, ny, nx = data.shape
    assert( nx==ny )
    projs = np.zeros((2, nz, nx))
    for ai, axis in enumerate([1, 2]):
        proj = data.mean(axis=axis)
        proj = normalize(proj, percentile=(0, 100))
        projs[ai] = proj
    return projs, apix

@st.cache(persist=True, show_spinner=False)
def get_2d_stack(url):
    ds = np.DataSource(None)
    fp=ds.open(url)
    import mrcfile
    with mrcfile.open(fp.name) as mrc:
        data = mrc.data*1.0
        apix = mrc.voxel_size.x.item()
    return data, apix

@st.cache(persist=True, show_spinner=False)
def get_2d_image(url):
    known_urls = [ example.url for example in data_examples]
    if url in known_urls or url.endswith(".class_averages.blob") or url.endswith(".templates_selected.blob"):
        data, apix = get_2d_stack(url)
    elif (url.find("emd_")!=-1 and url.endswith(".map.gz")) or url.endswith(".map") or url.endswith(".mrc"):
        data, apix = get_3d_map_projections(url)
    elif url.lower().startswith("emd-"):
        emd_id = url.lower().split("emd-")[-1]
        url2 = f"ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emd_id}/map/emd_{emd_id}.map.gz"
        data, apix = get_3d_map_projections(url2)
    else:
        from skimage.io import imread
        data = imread(url)
        data = np.expand_dims(data, axis=0)
        apix = 1.0
    return data, apix

class Data(object):
    def __init__(self, twist, rise, csym, diameter, apix, url=None):
        self.twist = twist
        self.rise = rise
        self.csym = csym
        self.diameter = diameter
        self.apix = apix
        self.url = url

data_examples = [
    Data(twist=29.40, rise=21.92, csym=6, diameter=135, apix=2.3438, url="https://tinyurl.com/y5tq9fqa")
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
