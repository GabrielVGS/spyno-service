"""spyno-decision — serviço de decisão do Spyno (TCC).

Política: Thompson Sampling disjoint (uma regressão logística bayesiana por
braço, aproximação de Laplace diagonal, retreino em batch diário). Braços:
{no_notify, spark, facilitator, signal} — taxonomia de gatilhos de Fogg.
"""

__version__ = "0.1.0"
