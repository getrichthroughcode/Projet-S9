import numpy as np
import matplotlib.pyplot as plt
def test_1(data):
    N=len(data)
    for i in range(1,N):
        if np.abs(data[i])> 1.95* abs(data[0])/np.sqrt(N):
            print(data[i])
            print(1.95* data[0]/np.sqrt(N))
            return False
    return True

def test_2(data,k):
    N=len(data)
    s=0
    for i in range(1,k):
        s+= data[i]**2
    print(s)
    print((k+1.65*np.sqrt(2*k))*data[0]**2/N)
    return s<=(k+1.65*np.sqrt(2*k))*data[0]**2/N

if __name__ == "__main__":
    BB = np.random.randn(100)
    correlation_BB = np.correlate(BB, BB,'same')
    lags_BB = np.arange(-len(correlation_BB)/2 , len(correlation_BB)/2)
    correlation_BB_dec = np.roll(correlation_BB,int(len(correlation_BB)/2))
    print(test_1(correlation_BB_dec))
    plt.figure(figsize=(10, 6))
    plt.plot(lags_BB, correlation_BB)
    plt.title("Fonction de corrélation en x")
    plt.grid()
    plt.show()