import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Wave2D import Wave2D_Neumann

mx = my = 2
cfl = 1/np.sqrt(2)

N = 100
Nt = 100

sol = Wave2D_Neumann()
xij, yij = sol.create_mesh(N=N)
data = sol(N=N, Nt=Nt, cfl=cfl, mx=mx, my=my, store_data=1)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
frames = []
for n, val in data.items():
    frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2);
    #frame = ax.plot_surface(xij, yij, val, vmin=-0.5*data[0].max(),
    #                        vmax=data[0].max(), cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    frames.append([frame])

ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                repeat_delay=1000)
ani.save("report/neumannwave.gif", writer="pillow", fps=24)