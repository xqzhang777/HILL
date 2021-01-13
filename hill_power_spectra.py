#!/usr/bin/env python

""" 
MIT License

Copyright (c) 2021 Wen Jiang

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

import sys, pathlib, math

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
import_with_auto_install("numpy scipy pandas mrcfile finufft joblib".split())

import numpy as np

def main():
    args =  parse_command_line()

    data = image2dataframe(args.inputImage)
    
    if args.apix>0:
        data.loc[:, "apix"] = args.apix
    if "apix" not in data:
        with mrcfile.open(data["filename"].iloc[0], mode=u'r', header_only=True) as mrc:
            data.loc[:, "apix"] = mrc.voxel_size.x
    args.apix = data['apix'].iloc[0]
    if abs(args.verbose)>0:
        print(f"Sampling: {args.apix:.4f}")

    if args.cutoffRes < args.apix * 2.0:
        args.cutoffRes = args.apix * 2.0
    if abs(args.verbose)>0:
        print(f"Cutoff resolution: {args.cutoffRes:.4f}")

    if not args.groupby:
        if abs(args.verbose)>0:
            print(f"Availabe parameters for --groupby: {data.columns.values}")
        if "phi0" in data:
            if "class" in data: args.groupby = ["class"]
            elif "helicaltube" in data: args.groupby = ["filename", "helicaltube"]
        elif len(data["filename"].unique())==1:
            with mrcfile.open(data["filename"].iloc[0], mode=u'r', header_only=True) as mrc:
                nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
            if nx==ny and nz!=nx and nz<=500 and "helicaltube" not in data:
                args.groupby = ["pid"]
    
    if args.groupby == ["None"]: args.groupby = []

    required_attrs = "apix".split()
    if args.groupby:
        if "helicaltube" in args.groupby:
            if "filename" not in args.groupby:
                args.groupby = ["filename"] + args.groupby
            required_attrs += ["phi0"]
        required_attrs += args.groupby
    missing_attrs = [attr for attr in required_attrs if attr not in data]
    if missing_attrs:
        print(f"ERROR: parameters '{' '.join(missing_attrs)}'' are not available. available parameters are '{' '.join(data)}''")
        sys.exit(-1)

    if args.verbose:
        if "helicaltube" in data:
            helices = data.groupby(["filename", "helicaltube"])
            print(f'Read {len(data)} segments/particles in {len(helices)} helices in {len(data["filename"].unique())} micrographs from {args.inputImage}')
        else:
            print(f'Read {len(data)} particles in {len(data["filename"].unique())} micrographs from {args.inputImage}')

    if args.groupby:
        groups = data.groupby(args.groupby, sort=True)
        if args.verbose:
            print(f"{len(groups)} groups based on {args.groupby}")
        if args.minCount>0:
            groups, groups_all = [], groups
            for gi, g in enumerate(groups_all):
                n = len(g[1])
                if n<args.minCount:
                    if args.verbose>0:
                        print(f"\tGroup {gi+1}/{len(groups)} - {g[0]}: skipped as it has only {n} particles (<{args.minCount})")
                    continue
                groups.append(g)
            if args.verbose:
                print(f"{len(groups)} groups after removing small groups (<{args.minCount} particles)")
    else:
        groups = [("all_particles", data)]

    compute_phase_differences = args.forcePhaseDiff or "phi0" in data

    def particle_subsets(groups, max_batch_size=args.batchSize):
        for gi, g in enumerate(groups):
            group_name, group_particles = g
            if len(args.groupby)>1:
                group_name=tuple(["%s=%s" % (attr, group_name[ai]) for ai, attr in enumerate(args.groupby)])
            elif len(args.groupby)==1:
                group_name=tuple(["%s=%s" % (args.groupby[0], group_name)])
            else:
                group_name=None

            mgraphs = list(group_particles.groupby(["filename"], sort=True))
            nptcls = [len(m[1]) for m in mgraphs]
            batches = []
            i0 = 0
            while True:
                for i in range(i0, len(nptcls)):
                    if sum(nptcls[i0 : i+1]) >= max_batch_size or i==len(nptcls)-1:
                        batches.append(mgraphs[i0:i+1])
                        i0 = i+1
                if i0>=len(nptcls)-1: break
            for bi, batch in enumerate(batches):
                yield batch, (gi, bi, group_name, len(groups), len(batches))

    from joblib import Parallel, delayed
    fftavgs = Parallel(n_jobs=args.cpu, verbose=max(0, abs(args.verbose)-2), prefer="processes")(
        delayed(averageOneBatch)(batch, group_id, compute_phase_differences, args.diameterMask, args.cutoffRes, args.fftX, args.fftY, args.align, args.verbose) for batch, group_id in particle_subsets(groups))
    
    if args.verbose>0 and len(fftavgs)>1:
        print(f"Combining results of {len(fftavgs)} tasks")
    
    outputPrefix = args.outputPrefix or pathlib.Path(args.inputImage).stem
    if args.groupby: outputPrefix += f".groupby-{'-'.join(args.groupby)}"
    if args.align: outputPrefix += ".algined"
    outputLstFile = outputPrefix+ (".ps-pd.lst" if compute_phase_differences else ".ps.lst")
    psFile = outputPrefix+".ps.mrcs"    # power spectra

    if compute_phase_differences:
        pdFile = outputPrefix+".pd.mrcs"    # phase differences across meridian
    else:
        pdFile = None

    mrc_ps = mrcfile.new_mmap(psFile, shape=(len(groups), args.fftY, args.fftX), mrc_mode=2, overwrite=True)
    mrc_ps.voxel_size = args.cutoffRes/2
    if compute_phase_differences:
        mrc_pd = mrcfile.new_mmap(pdFile, shape=(len(groups), args.fftY, args.fftX), mrc_mode=2, overwrite=True)
        mrc_pd.voxel_size = args.cutoffRes/2
    else:
        mrc_pd = None

    import pandas as pd
    data_output = pd.DataFrame(index=list(range(len(groups))), columns="pid filename".split())
    data_output.loc[:, "nyquist"] = args.cutoffRes

    results = {}    
    for i in range(len(fftavgs)):
        ps_avg, pd_avg, count, group_id = fftavgs[i]
        if group_id not in results:
            d = {}
            d["ps_avg"] = np.zeros_like(ps_avg)
            if pd_avg is not None:
                d["pd_avg"] = np.zeros_like(pd_avg)
            d["count"] = 0
            results[group_id] = d
        results[group_id]["ps_avg"] += ps_avg
        if pd_avg is not None: results[group_id]["pd_avg"] += pd_avg
        results[group_id]["count"] += count

    for group_id in results:
        gi, _, group_name, _, _ = group_id
        data_output.loc[gi, "pid"] = gi
        data_output.loc[gi, "count"] = results[group_id]["count"]
        if group_name:
            data_output.loc[gi, "group"] = str(group_name)
        ps_avg = results[group_id]["ps_avg"] / results[group_id]["count"]
        ps_avg = np.fft.fftshift(ps_avg)
        mrc_ps.data[gi] = ps_avg
        if "pd_avg" in results[group_id]:
            pd_avg = results[group_id]["pd_avg"] / results[group_id]["count"]
            pd_avg = np.rad2deg(np.arccos(pd_avg))
            pd_avg = np.fft.fftshift(pd_avg)
            mrc_pd.data[gi] = pd_avg
    mrc_ps.close()
    if mrc_pd is not None: mrc_pd.close() 

    data_output.loc[:, "filename"] = psFile
    if pdFile: data_output.loc[:, "pdfile"] = pdFile

    cols = [c for c in "pid filename group count pdfile".split() if c in data_output]
    data_output = data_output.loc[:, cols]
    dataframe2lst(data_output, outputLstFile)
    
    if args.verbose:
        if pdFile:
            print(f"{len(data_output)} power spectra/phase differences across meridian images saved to {outputLstFile}")
        else:
            print(f"{len(data_output)} power spectra images saved to {outputLstFile}")

    if args.showPlot:
        params = {}
        if pdFile:
            params["input_mode"] = [1, 1]
            params["url"] = [pathlib.Path(pdFile).absolute(), pathlib.Path(psFile).absolute()]
            params["input_type"] = ["PD", "PS"]
        else:
            params["input_mode"] = [1]
            params["url"] = [pathlib.Path(psFile).absolute()]
            params["input_type"] = ["PS"]
        params["sync_i"] = 1
        if args.diameterMask>0:
            params["radius"] = round(args.diameterMask/2 * 0.9)
        if args.cutoffRes>args.apix*2:
            params["resx"] = args.cutoffRes * 1.5
            params["resy"] = args.cutoffRes

        query_string = get_query_string(params)
        code_path = pathlib.Path(__file__).parent / "hill.py"
        if code_path.exists():
            code_path = code_path.as_posix()
        else:
            code_path = "https://raw.githubusercontent.com/wjiang/HILL/main/hill.py"
        import_with_auto_install("streamlit")
        import subprocess
        cmd = f"streamlit run {code_path} -- --query_string '{query_string}'"
        print(cmd)
        subprocess.call(cmd, shell=True)

def averageOneBatch(mgraphs, group_id, compute_phase_differences, diameterMask, cutoff_res, pad_nx, pad_ny, align, verbose):
    nPtcls = sum([len(m[1]) for m in mgraphs])
    if verbose>0:
        gi, bi, _, ng, nb = group_id
        print(f"Group {gi+1}/{ng} - Batch {bi+1}/{nb}: {nPtcls} particles from {len(mgraphs)} micrographs")
    phi0Angles = np.array([90]*nPtcls)
    apix = mgraphs[0][1]["apix"].iloc[0]
    tapering_filter = None
    data_orig = None
    data_in = None
    i0 = 0
    for mgraph in mgraphs:
        _, particles = mgraph
        n = len(particles)
        pids = particles["pid"].astype(int).values
        filename = particles["filename"].iloc[0]
        with mrcfile.mmap(filename, mode='r') as mrc:
            _, ny, nx = mrc.data.shape
            if data_orig is None:
                data_orig = np.zeros((nPtcls, ny, nx), dtype=np.float32)
                if align:
                    data_in = np.zeros((nPtcls, ny, nx), dtype=np.float32)
                else:
                    data_in = data_orig
            if tapering_filter is None:
                if diameterMask > 0:
                    fraction_x = diameterMask/apix / nx
                else:
                    fraction_x = 0.9
                tapering_filter = generate_tapering_filter(image_size=(ny, nx), fraction_start=[0.9, fraction_x], fraction_slope=0.1)

            if "phi0" in particles:
                phi0Angles[i0:i0+n] = particles["phi0"].astype(float).values

            for i in range(n):
                data_orig[i0+i] = mrc.data[pids[i]]
        i0 += n

    if align:
        da = np.zeros(nPtcls)
        dxy = np.zeros(nPtcls)
        for pi in range(nPtcls):
            d = data_orig[pi]
            dphi = - phi0Angles[pi]
            d_aligned, da[pi], dxy[pi] = rotation_trans_align(image=d, angle0=dphi, dx0=0, dy0=0, mask=tapering_filter)
            data_in[pi] = d_aligned * tapering_filter
        if verbose>1:
            gi, bi, _, ng, nb = group_id
            print(f"Group {gi+1}/{ng} - Batch {bi+1}/{nb}: mean rotation = {np.mean(np.abs(da)):.2f}°\t shift = {np.mean(np.abs(dxy))*apix:.1f}Å")
    else:
        for pi in range(nPtcls):
            d = data_orig[pi]
            dphi = - phi0Angles[pi]
            if dphi != 0:
                d_rotated = rotate_shift_image(data=d, angle=dphi, post_shift=(0, 0))
            else:
                d_rotated = d
            data_in[pi] = d_rotated * tapering_filter

    ps_avg = np.zeros((pad_ny, pad_nx), dtype=np.float32)
    if compute_phase_differences:
        pd_avg = np.zeros((pad_ny, pad_nx), dtype=np.float32)
    
    data_fft = fft_rescale(images=data_in, apix=apix, cutoff_res=(cutoff_res, cutoff_res), output_size=(pad_ny, pad_nx))
    amp = np.abs(data_fft)
    amp *= amp
    ps_avg = np.sum(amp, axis=0)

    if compute_phase_differences:
        phase = np.angle(data_fft)
        cos = compute_phase_difference_across_meridian(phase, compute_cosine=True)
        pd_avg = np.sum(cos, axis=0)
    else:
        pd_avg = None

    return (ps_avg, pd_avg, nPtcls, group_id)

def compute_phase_difference_across_meridian(phase, compute_cosine=False):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
    phase_diff = phase * 0
    phase_diff[..., 1:] = phase[..., 1:] - phase[..., 1:][..., ::-1]
    if compute_cosine:
        phase_diff = np.cos(phase_diff)
    return phase_diff

def fft_rescale(images, apix=1.0, cutoff_res=None, output_size=None):
    assert(len(images.shape) in [2, 3])

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

    images_work = images
    if len(images.shape) == 3:
        n = images.shape[0]
        if n==1:
            images_work = images[0]
    else:
        n = 1

    from finufft import nufft2d2
    fft = nufft2d2(x=Y, y=X, f=images_work.astype(np.complex), eps=1e-6)
    if n>1:
        fft = fft.reshape((n, ony, onx))

        # phase shifts for real-space shifts by half of the image box in both directions
        phase_shift = np.ones(fft.shape)
        phase_shift[:, 1::2, :] *= -1
        phase_shift[:, :, 1::2] *= -1
        fft *= phase_shift
    else:
        fft = fft.reshape((ony, onx))

        # phase shifts for real-space shifts by half of the image box in both directions
        phase_shift = np.ones(fft.shape)
        phase_shift[1::2, :] *= -1
        phase_shift[:, 1::2] *= -1
        fft *= phase_shift
        if len(images.shape)==3 and images.shape[0]==1:
            fft = fft[np.newaxis, :, :]
    # now fft has the same layout and phase origin (i.e. np.fft.ifft2(fft) would obtain original image)
    return fft

def rotation_trans_align(image, angle0, dx0=0, dy0=0, mask=None):
    # further refine rotation/shift
    def score_rotation_shift(x):
        da, dy, dx = x
        angle = angle0 + da
        tmp = rotate_shift_image(data=image, angle=angle, pre_shift=(dy, dx))
        tmps = [tmp, tmp[::-1, :], tmp[:, ::-1], tmp[::-1, ::-1]]
        tmp = rotate_shift_image(data=image, angle=angle+180, pre_shift=(dy, dx))
        tmps += [tmp, tmp[::-1, :], tmp[:, ::-1], tmp[::-1, ::-1]]
        n = len(tmps)
        tmp_mean = np.zeros_like(image)
        for tmp in tmps: tmp_mean += tmp
        tmp_mean /= n
        err = 0
        for tmp in tmps:
            err += np.sum(np.abs(tmp - tmp_mean)*mask)
        err /= n * image.size
        return err
    from scipy.optimize import fmin
    res = fmin(score_rotation_shift, x0=(0, dy0, dx0), xtol=1e-4, disp=0)
    da, dy, dx = res
    ret = rotate_shift_image(data=image, angle=angle0+da, pre_shift=(dy, dx))
    return ret, da, np.hypot(dy, dx)

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

def star2dataframe(starFile):
    import pandas as pd
    import_with_auto_install("gemmi")
    from gemmi import cif
    star = cif.read_file(starFile)
    if len(star) == 2:
        optics = cif.Document()
        optics.add_copied_block(star[0])
        del star[0]
        js = optics.as_json(True)  # True -> preserve case
        optics = pd.read_json(js).T
        d = {c.strip('_'): optics[c].values[0] for c in optics}
        optics = pd.DataFrame(d)
    else:
        optics = None
    js = star.as_json(True)  # True -> preserve case
    data = pd.read_json(js).T
    d = {c.strip('_'): data[c].values[0] for c in data}
    data = pd.DataFrame(d)
    
    assert("rlnImageName" in data)
    tmp = data["rlnImageName"].str.split("@", expand=True)
    indices, filenames = tmp.iloc[:,0], tmp.iloc[:, -1]
    indices = indices.astype(int)-1
    data["pid"] = indices
    data["filename"] = filenames

    if optics is not None:
        og_names = set(optics["rlnOpticsGroup"].unique())
        for gn, g in data.groupby("rlnOpticsGroup", sort=False):
            if gn not in og_names:
                print(f"ERROR: optic group {gn} not available ({sorted(og_names)})")
                sys.exit(-1)
            ptcl_indices = g.index
            og_index = optics["rlnOpticsGroup"] == gn
            if "rlnPixelSize" in optics:
                data.loc[ptcl_indices, "apix"] = optics.loc[og_index, "rlnPixelSize"].astype(float).iloc[0]
    if "rlnPixelSize" in data:
        data.loc[:, "apix"] = data["rlnPixelSize"]
    if "rlnClassNumber" in data:
        data.loc[:, "class"] = data["rlnClassNumber"]
    if "rlnHelicalTubeID" in data:
        data.loc[:, "helicaltube"] = data["rlnHelicalTubeID"].astype(int)-1
    if "rlnAnglePsiPrior" in data:
        data.loc[:, "phi0"] = data["rlnAnglePsiPrior"].astype(float).round(3) - 90.0

    return data

def cs2dataframe(csFile):
    # read CryoSPARC v2/3 meta data
    cs = np.load(csFile)
    import pandas as pd
    data = pd.DataFrame.from_records(cs.tolist(), columns=cs.dtype.names)
    if csFile.find("passthrough_particles") == -1:
        ptf = sorted(pathlib.Path(csFile).parent.glob("*passthrough_particles*.cs"))
        if ptf:
            passthrough_file = ptf[0].as_posix()
            cs = np.load(passthrough_file)
            extra_data = pd.DataFrame.from_records(cs.tolist(), columns=cs.dtype.names)
            data = pd.concat([data, extra_data], axis=1)
    mapping = {"blob/idx":"pid", "blob/psize_A":"apix", "filament/filament_uid":"helicaltube"}
    for key in mapping:
        if key in data:
            data.loc[:, mapping[key]] = data[key]
    if "filament/filament_pose" in data:    # = - rlnAnglePsiPrior
        data.loc[:, "phi0"] = -np.rad2deg(data["filament/filament_pose"]) - 90
    if "alignments2D/class" in data:
        data.loc[:, "class"] = data["alignments2D/class"]-1
    if "blob/path" in data:
        data.loc[:, "filename"] = data["blob/path"].str.decode("utf-8")
    return data

def lst2dataframe(lstFile):
    data = []
    vars = []
    with open(lstFile) as fp:
        for l in fp:
            if l[0] == "#": continue
            items = l.split()
            if len(items) < 2: continue
            d = [ ("pid", int(items[0]) ), ("filename", items[1]) ]
            d+= [ it.split("=") for it in items[2:] ]
            if len(d)>len(vars):    # to preserve order
                vars = [v[0] for v in d]
            data.append(dict(d))
    import pandas as pd
    p = pd.DataFrame(data).loc[:, vars]
    if "phi0" not in p and "euler" in p:
        p.loc[:, "phi0"] = p["euler"].str.split(",", expand=True).iloc[:, -1].astype(float)
    return p

def mrc2dataframe(mrcFile):
    import mrcfile
    with mrcfile.open(mrcFile, mode=u'r', header_only=True) as mrc:
        nz = mrc.header.nz
        apix = mrc.voxel_size.x
    import pandas as pd
    p = pd.DataFrame({"pid":range(nz), "filename":mrcFile, 'apix':apix})
    return p

def image2dataframe(inputFile):
    if not pathlib.Path(inputFile).exists():
        print(f"ERROR: cannot find file {inputFile}")
        sys.exit(-1)
    if inputFile.endswith(".star"):    # relion
        p = star2dataframe(inputFile)
    elif inputFile.endswith(".cs"):  # cryosparc v2.x
        p = cs2dataframe(inputFile)
    elif inputFile.endswith(".lst"):    # jspr
        p = lst2dataframe(inputFile)
    elif inputFile.endswith(".mrc") or inputFile.endswith(".mrcs"):
        p = mrc2dataframe(inputFile)
    else:
        print("ERROR: {inputFile} is in a unsupported format")
        sys.exit(-1)
    
    cols = [c for c in "pid filename apix class helicaltube phi0".split() if c in p]
    p = p.loc[:, cols]

    int_types = "pid helicaltube".split()
    float_types = "apix phi0".split()
    for i in int_types:
        if i in p: p.loc[:, i] = p.loc[:, i].astype(int)
    for f in float_types:
        if f in p: p.loc[:, f] = p.loc[:, f].astype(float)
    
    dir0 = pathlib.Path(inputFile).parent
    mapping = {}
    for f in p["filename"].unique():
        if f in mapping: continue
        fp = pathlib.Path(f)
        name = fp.name
        choices = [fp, dir0/name, dir0/".."/name, dir0/"../.."/name, dir0/".."/fp, dir0/"../.."/fp]
        for choice in choices:
            if choice.exists():
                mapping[f] = choice.resolve().as_posix()
                break
        if f in mapping:
            fp2 = pathlib.Path(mapping[f])
            for fo in fp2.parent.glob("*"+fp.suffix):
                ftmp = (fp.parent / fo.name).as_posix()
                mapping[ftmp] = fo.as_posix()
    for f in p["filename"].unique():
        if f not in mapping:
            print(f"WARNING: {f} is not accessible")
            mapping[f] = f
    p.loc[:, "filename"] = p.loc[:, "filename"].map(mapping)
    return p

def dataframe2lst(data, lstFile):
    int_types = "pid count".split()
    float_types = "apix".split()
    for i in int_types:
        if i in data: data.loc[:, i] = data.loc[:, i].astype(int)
    for f in float_types:
        if f in data: data.loc[:, f] = data.loc[:, f].astype(float)

    keys = list(data)
    keys.remove("pid")
    keys.remove("filename")

    lines = data['pid'].astype(str) + '\t' + data['filename']
    for k in keys:
        mask = data[k].notnull()
        if data[k].dtype in [np.float64, np.float32]:
            lines[mask] += '\t' + k + '=' + data[k][mask].round(6).astype(str)
        else:
            lines[mask] += '\t' + k + '=' + data[k][mask].astype(str)
    lines = lines.str.strip()

    maxlen = max(lines.str.len())
    lines = lines.str.pad(maxlen, side='right')

    with open(lstFile, "w") as lstfp:
        lstfp.write("#LSX\n#If you edit this file, you MUST rerun lstfast.py on it before using it!\n# %d\n" % (maxlen+1))
        lstfp.write('\n'.join(lines))
        lstfp.write('\n')

def get_query_string(query_params):
    from urllib.parse import urlencode
    s = urlencode(query_params, doseq=True)
    return s

def parse_command_line():
    import argparse
    description = "compute the average 2D power spectra of particles"
    epilog  = "Author: Wen Jiang (jiang12@purdue.edu)\n"
    epilog += "Copyright (c) 2021 Purdue University\n"

    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('inputImage', help='input particle lst/star/cs/mrcs file')
    parser.add_argument('--outputPrefix', metavar="<str>", type=str, help="prefix of output files", default="")
    parser.add_argument("--groupby", metavar="<attr>", type=str, nargs="+", help="group particles by these parameters (class, helicaltube etc)", default=[])
    parser.add_argument("--minCount", metavar="<n>", type=int, help="ignore groups of fewer particles than this minimal count. default: %(default)s", default=-1)
    parser.add_argument("--batchSize", metavar="<n>", type=int, help="maximal number of particles per batch. default: %(default)s", default=100)
    parser.add_argument("--apix", metavar="<Å/pixel>", type=float, help="pixel size of input image", default=0)
    parser.add_argument("--diameterMask", metavar="<Å>", type=float, help="masking with this filament/tube diameter (in Angstrom). disabled by default", default=0)
    parser.add_argument("--cutoffRes", metavar="<float>", type=float, help="compute power spectra up to this resolution. default to 2*apix", default=0)
    parser.add_argument("--fftX", metavar="<nx>", type=int, help="set FFT x-dimenstion to this size. default: %(default)s", default=512)
    parser.add_argument("--fftY", metavar="<ny>", type=int, help="set FFT y-dimenstion to this size. default: %(default)s", default=1024)
    parser.add_argument("--align", metavar="<0|1>", type=int, help="center each particle and rotate it to the vertical direction. default: %(default)s", default=0)
    parser.add_argument("--forcePhaseDiff", metavar="<0|1>", type=int, help="compute phase differences across meridian even if in-plane angles are not avilable. default: %(default)s", default=0)
    parser.add_argument("--showPlot", metavar="<0|1>", type=int, help="display power spectra for indexing. default: %(default)s", default=1)
    parser.add_argument("--cpu", metavar="<n>", type=int, help="use this number of cpus/cores. default: %(default)s", default=1)
    parser.add_argument("--verbose", metavar="<n>", type=int, help="verbose level. default: %(default)s", default=1)
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
