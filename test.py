import numpy as np
from pyteda.models import SWEModel
from pyteda.observation import LinearSelection, IsotropicDiagonal
# usa tu mismo setup
model = SWEModel(LMAX=21, dt=120.0, state_vars=["u","v","h"])
fs = model.field_size
# toma una observación de cada variable y mira sus vecinos
for var, off in [("u",0),("v",fs),("h",2*fs)]:
    s = off + 100   # un punto cualquiera de esa variable
    ngb = model.get_ngb(s, 4)
    # ¿todos los vecinos caen en el mismo bloque de variable?
    bloques = set(int(g)//fs for g in ngb)
    print(f"{var}: obs s={s}, vecinos en bloques de variable {bloques}, n_vecinos={len(ngb)}")