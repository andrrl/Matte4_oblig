#Matte4_oblig  - løser varmelikningen på et valgfritt område numerisk med to romlige koordinater

#starter med å importere nødvendige biblioteker
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#jeg ser på varmelikningen til en stav med lengder:
lx = 1 #lengde på staven i xretning
ly = 1 #lengde på staven i yretning

# Definerer antall punkter: 
nx = 10  # Antall gridpunkter i x-retning
ny = 10  # Antall gridpunkter i y-retning
nt = 10000  # Antall tidssteg
T = 1 #total lengde av simuleringen

#definerer så steglengder og konstanter for løsningen: 
h = lx/(nx-1)
p = ly/(ny-1)
k = T/(nt-1)

#lager lister for verdier for x, y og t
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
t = np.linspace(0, T, nt)

#lager tom liste for varmen u - som vi skal finne
u = np.zeros((nx, ny, nt))

#definerer funksjonen:

func = lambda x, y: np.sin(3*np.pi*np.sqrt((x-.5)**2 + (y-.5)**2))

func_vals = np.array([[func(x_val, y_val) for y_val in y] for x_val in x])
u[1:-1,1:-1,0] = func_vals[1:-1, 1:-1]



#randkrav og initialbetingelser
#setter at u(x, y, 0) = f(x, y) og at u(x, y, t) = 0 på randen til omega. 

for i in range(1, nx-1):
    for j in range(1, ny-1):
        u[i,j,0] = np.sin(np.pi * x[i]) + np.sin(np.pi * y[i])

u[0, :, 0] = u[-1, :, 0] = 0
u[:, 0, 0] = u[:, -1, 0] = 0


#setter stabilitetsparametre
gamma_x = k/(h**2)
gamma_y = k/(p**2)


#bruker eksplisitt medtode for å løse likningen:
for l in range(nt-1):
    for i in range(1, nx-1): 
        for j in range(1, ny-1):
            u[i,j, l+1] = u [i,j,l] + gamma_x *(-2*u[i,j,l] + u[i+1,j,l] + u[i-1,j,l]) + gamma_y * (-2*u[i,j,l] + u[i,j+1,l] + u[i,j-1,l])
            

fig, ax = plt.subplots() 
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes) 

fig, ax = plt.subplots() 
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes) 

def init(): 
    global cax  
    cax = ax.imshow(u[:, :, 0], extent=(0, lx, 0, ly), origin='lower', vmin=np.min(u), vmax=np.max(u)) 
    fig.colorbar(cax) 
    return cax, time_text 

def update(frame): 
    cax.set_data(u[:, :, frame]) 
    time_text.set_text('Tid: {:.2f}s'.format(frame * k)) 
    return cax, time_text 

ani = animation.FuncAnimation(fig, update, frames=nt, init_func=init, blit=False, interval=50) 
plt.title("Temperaturfordeling over tid") 
plt.xlabel('X') 
plt.ylabel('Y') 
plt.show()



