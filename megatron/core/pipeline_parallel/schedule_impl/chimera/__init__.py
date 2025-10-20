"""
Chimera: 2-VR Bidirectional Pipeline Parallelism

Chimera implements a simplified bidirectional pipeline with 2 virtual ranks (VR)
per device, eliminating the V-shape pattern for reduced P2P communication overhead.

Variants:
- chimera_2vr.py: Symmetric (uniform layer distribution)
- chimera_2vr_asymmetric.py: Asymmetric (variable layer distribution)
"""

# Will be populated during Chimera implementation
__all__ = []
