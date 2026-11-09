"""Fetch the published numeric FC-Gram matrices; verify the exact data version.

Data authors: Daniel Leibovici and Valentin Duruisseaux.
Source: neuraloperator/FC_PINO, FC_Gram_Construction/README.md.
Downloaded research data stays under ignored benchmarks/, not vendored source.
"""
from pathlib import Path
import hashlib
import urllib.request

URL='https://raw.githubusercontent.com/neuraloperator/FC_PINO/main/FC_Gram_Construction/FCGram_matrices/FCGram_data_d5_C25.mat'
SHA256='0fe304836d87dadf0b09d452ea97445ceefeb46adcc4b173edaae09a1fe7668c'
if __name__=='__main__':
    data=urllib.request.urlopen(URL,timeout=30).read()
    if hashlib.sha256(data).hexdigest()!=SHA256:
        raise RuntimeError('Upstream FC data changed; review the new version before use')
    path=Path(__file__).resolve().parents[1]/'benchmarks/traveling_shock/prior_art/fc_data/FCGram_data_d5_C25.mat'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    print(path)
