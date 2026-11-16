"""CACHE v0.2: training-free H&E stromal architecture, separate continuous and ridge branches.
Research-only; collagen-like appearance, not collagen-specific identification.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage import color, filters, morphology, exposure
import matplotlib.pyplot as plt


def _unit(x, mask=None):
    v=x[mask] if mask is not None and np.any(mask) else x.ravel()
    lo,hi=np.percentile(v,[2,98])
    return np.clip((x-lo)/(hi-lo+1e-9),0,1)


def analyze(rgb, mpp=None, scales_um=(1.5,3,6,12), window_um=30,
            candidate_quantile=0.23, ridge_quantile=0.70):
    rgb=np.asarray(rgb)[...,:3].astype(np.uint8)
    f=rgb.astype(np.float32)/255.
    hed=color.rgb2hed(f); h=np.maximum(hed[...,0],0); e=np.maximum(hed[...,1],0)
    hsv=color.rgb2hsv(f)
    tissue=(hsv[...,1]>.055)&(hsv[...,2]<.985)
    # Exclude strongly saturated non-H&E graphics, including red arrows.
    red_annotation=(f[...,0]>.48)&(f[...,0]>1.65*f[...,1])&(f[...,0]>1.5*f[...,2])&(hsv[...,1]>.48)
    green_annotation=(f[...,1]>1.4*f[...,0])&(f[...,1]>1.3*f[...,2])&(hsv[...,1]>.45)
    valid_tissue=tissue & ~morphology.binary_dilation(red_annotation|green_annotation,morphology.disk(3))
    if valid_tissue.sum()<100: valid_tissue=tissue
    # Nuclei are excluded from candidate mask, but not from the neighborhood tensor itself.
    hn=_unit(h,valid_tissue); en=_unit(e,valid_tissue)
    nuclei=(hn>.66)&(hn>en*.8)
    nuclei=morphology.binary_dilation(nuclei,morphology.disk(2))
    # Eosin-rich nonnuclear candidates. This is NOT collagen-specific.
    threshold=np.quantile(en[valid_tissue],candidate_quantile) if valid_tissue.any() else .25
    candidate=valid_tissue & ~nuclei & (en>threshold)
    # Work in a lightly smoothed eosin OD image, avoiding aggressive normalization.
    signal=gaussian_filter(en,0.7)
    scales_px=np.clip(np.asarray(scales_um)/(mpp if mpp else 1.),1.2,30)
    # Separate tensor and ridge branches: no multiplication for continuous architecture.
    tensor_energy=[]; tensors=[]; ridge_list=[]
    for sigma in scales_px:
        smooth=gaussian_filter(signal,max(.65,sigma*.35))
        gy,gx=np.gradient(smooth)
        jxx=gaussian_filter(gx*gx,sigma); jyy=gaussian_filter(gy*gy,sigma)
        jxy=gaussian_filter(gx*gy,sigma)
        delta=np.sqrt((jxx-jyy)**2+4*jxy*jxy)
        coh=delta/(jxx+jyy+1e-10)
        theta=(.5*np.arctan2(2*jxy,jxx-jyy)+np.pi/2)%np.pi
        energy=np.sqrt(np.maximum(jxx+jyy,0))*sigma
        tensor_energy.append(energy)
        tensors.append((coh,theta))
        # Both bright and dark ridges, normalized separately per scale.
        rr=np.maximum(filters.sato(smooth,sigmas=[max(1.,sigma*.65)],black_ridges=False),
                      filters.sato(smooth,sigmas=[max(1.,sigma*.65)],black_ridges=True))
        ridge_list.append(_unit(rr,valid_tissue))
    energies=np.stack(tensor_energy)
    # Normalize energies per scale to avoid smallest-scale dominance.
    norm_energies=np.stack([_unit(q,valid_tissue) for q in tensor_energy])
    coherence_stack=np.stack([t[0] for t in tensors])
    choice=np.argmax(norm_energies*coherence_stack,axis=0)
    coherence=np.take_along_axis(coherence_stack,choice[None],axis=0)[0]
    theta=np.take_along_axis(np.stack([t[1] for t in tensors]),choice[None],axis=0)[0]
    strength=np.take_along_axis(norm_energies,choice[None],axis=0)[0]
    # Spatially robust confidence independent of ridge response.
    structural_weight=candidate.astype(float)*np.clip(coherence,0,1)*np.sqrt(strength)
    structural_weight=gaussian_filter(structural_weight,1)
    structure_valid=candidate & (strength>.08) & (coherence>.15)
    win_sigma=max(3,window_um/(mpp if mpp else 1.)/4)
    win_sigma=min(win_sigma,80)
    zr=gaussian_filter(structural_weight*np.cos(2*theta),win_sigma)
    zi=gaussian_filter(structural_weight*np.sin(2*theta),win_sigma)
    den=gaussian_filter(structural_weight,win_sigma)
    alignment=np.hypot(zr,zi)/(den+1e-9)
    # Suppress meaningless alignment in regions with too little structural evidence.
    support=den/(gaussian_filter(candidate.astype(float),win_sigma)+1e-9)
    alignment[~valid_tissue]=np.nan
    ridge=np.max(np.stack(ridge_list),axis=0)
    rthr=np.quantile(ridge[candidate],ridge_quantile) if candidate.any() else 1.
    bundles=candidate & (ridge>rthr) & (coherence>.18)
    bundles=morphology.remove_small_objects(bundles,min_size=12)
    w=structural_weight[structure_valid]; a=theta[structure_valid]
    z=np.sum(w*np.exp(2j*a))
    features=dict(candidate_fraction=float(candidate.mean()),orientation_coverage=float(structure_valid.sum()/max(candidate.sum(),1)),
                  ridge_coverage=float(bundles.sum()/max(candidate.sum(),1)),
                  global_alignment=float(abs(z)/(w.sum()+1e-9)),
                  dominant_orientation_deg=float(np.rad2deg(np.angle(z)/2)%180) if len(w) else float('nan'),
                  mean_local_alignment=float(np.nanmean(alignment[candidate])) if candidate.any() else float('nan'),
                  mpp_um=mpp)
    return dict(rgb=rgb,candidate=candidate,structure_valid=structure_valid,bundles=bundles,
                theta=theta,coherence=coherence,structural_weight=structural_weight,
                ridge=ridge,alignment=alignment,features=features)


def plot_review(r,out,title='CACHE v0.2'):
    rgb=r['rgb']; H,W=rgb.shape[:2]
    fig,ax=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    ax[0,0].imshow(rgb);ax[0,0].set_title('A  H&E')
    ax[0,1].imshow(rgb);ax[0,1].imshow(np.ma.masked_where(~r['candidate'],r['candidate']),cmap='autumn',alpha=.27,vmin=0,vmax=1);ax[0,1].set_title('B  Stromal candidate (not collagen-specific)')
    ax[0,2].imshow(r['structural_weight'],cmap='magma',vmin=0,vmax=np.percentile(r['structural_weight'],99.5)+1e-8);ax[0,2].set_title('C  Continuous structure confidence')
    ax[1,0].imshow(rgb);ax[1,0].imshow(np.ma.masked_where(~r['bundles'],r['bundles']),cmap='cool',alpha=.48,vmin=0,vmax=1);ax[1,0].set_title('D  Optional ridge bundle candidates')
    ax[1,1].imshow(rgb,alpha=.72)
    step=max(14,min(H,W)//22); yy,xx=np.mgrid[step//2:H:step,step//2:W:step]
    mask=r['structure_valid'][yy,xx]; t=r['theta'][yy,xx]
    ax[1,1].quiver(xx[mask],yy[mask],np.cos(t[mask]),np.sin(t[mask]),color='lime',angles='xy',scale_units='xy',scale=.11,width=.003,headwidth=0,headlength=0,headaxislength=0)
    ax[1,1].set_title('E  Orientation (independent of ridge)')
    im=ax[1,2].imshow(r['alignment'],cmap='viridis',vmin=0,vmax=1);ax[1,2].set_title('F  Local axial alignment');fig.colorbar(im,ax=ax[1,2],fraction=.045)
    for a in ax.flat:a.axis('off')
    fig.suptitle(title+' | collagen-like architecture, not collagen-specific detection',fontsize=13)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=120);plt.close(fig)
