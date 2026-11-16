"""Small deterministic test data; not a substitute for scientific embeddings."""
import argparse
import numpy as np
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--out', default='artifacts/example_vectors.npy')
p.add_argument('--count', type=int, default=2000)
p.add_argument('--dimension', type=int, default=128)
a = p.parse_args()
rng = np.random.default_rng(42)
vectors = rng.standard_normal((a.count, a.dimension)).astype('float32')
path = Path(a.out)
path.parent.mkdir(parents=True, exist_ok=True)
np.save(path, vectors)
print(f'Saved {vectors.shape} vectors to {path}')
