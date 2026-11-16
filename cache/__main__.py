import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .core import analyze, plot_review


def main():
    p = argparse.ArgumentParser(description='CACHE: collagen-like stromal architecture in H&E')
    p.add_argument('image', type=Path)
    p.add_argument('--output', type=Path, default=Path('cache_output'))
    p.add_argument('--mpp', type=float, default=None, help='Micrometers per pixel (required for physical-scale interpretation)')
    p.add_argument('--scales', type=float, nargs='+', default=[1.5, 3, 6, 12], help='Scales in micrometers if --mpp is set; otherwise pixels')
    p.add_argument('--window', type=float, default=30, help='Window scale in micrometers if --mpp is set; otherwise pixels')
    args = p.parse_args()
    if args.mpp is not None and args.mpp <= 0:
        p.error('--mpp must be positive')
    if any(s <= 0 for s in args.scales) or args.window <= 0:
        p.error('scales and window must be positive')
    im = np.asarray(Image.open(args.image).convert('RGB'))
    r = analyze(im, mpp=args.mpp, scales_um=args.scales, window_um=args.window)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_review(r, args.output / 'review.png', args.image.name)
    with (args.output / 'features.json').open('w') as f:
        json.dump(r['features'], f, indent=2, allow_nan=False)
    for key in ['candidate', 'structure_valid', 'bundles']:
        Image.fromarray((r[key].astype(np.uint8)*255)).save(args.output / f'{key}.png')
    np.savez_compressed(args.output / 'architecture_maps.npz', theta=r['theta'], coherence=r['coherence'], alignment=r['alignment'], structural_weight=r['structural_weight'], ridge=r['ridge'])
    print('CACHE results saved to', args.output)

if __name__ == '__main__':
    main()
