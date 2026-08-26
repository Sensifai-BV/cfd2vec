"""Downstream tasks built on one pretrained CFD2vec network.

    predict      geometry + BC + physics -> fields on any point set (fully masked mode)
    warmstart    super-fidelity with an optional low-fidelity prior, then solver seeding with safety checks
    retrieval    nearest cases in embedding space
"""
