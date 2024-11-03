import matplotlib.pyplot as plt 


def plot_trajectory(states):
    plt.figure(figsize=(10, 6))
    plt.plot(states[0, :], states[2, :], label='Trajectoire (x, y)')
    plt.xlabel('Position x')
    plt.ylabel('Position y')
    plt.title('Trajectoire curviligne simulée')
    plt.grid(True)
    plt.legend()
    plt.show()