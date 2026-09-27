"""Reference ("expert reviewer") re-implementations of the analyses performed by the
Virtual Biotech agents in case studies 2 (B7-H3 / CD276 in lung cancer) and 3
(OSMR / vixarelimab in ulcerative colitis), following Zhang et al., Science 2026,
Supplementary Methods.

Submodules: ``singlecell``, ``spatial``, ``survival``, ``tf_activity``,
``variance_decomposition``, ``biomarker``, ``cross_disease``.  Heavy optional
dependencies (scanpy, pydeseq2, liana, decoupler, lifelines, cell2location,
cellxgene_census, rpy2) are imported lazily inside the functions that need them.
"""
