import numpy as np
import matplotlib.pyplot as plt
#IMM algo with noise
def test_1(data):
    N=len(data)
    for i in range(1,N):
        if np.abs(data[i])> 1.95* abs(data[0])/np.sqrt(N):

            return False
    return True


def auto_coord(X):
    L = []
    for i in range(len(X)):
        s = 0

        for t in range(len(X) - i):
            s += X[t] * X[t + i]
        L.append((1 / len(X)) * s)
    out = []
    for i in range(len(X)-1,0,-1):
        out.append(L[i])
    for i in range(0,len(X)):
        out.append(L[i])
    return out





def test_2(data,k):
    N=len(data)
    s=0
    for i in range(1,k):
        s+= data[i]**2
    print(s)
    print((k+1.65*np.sqrt(2*k))*data[0]**2/N)
    return s<=(k+1.65*np.sqrt(2*k))*data[0]**2/N

if __name__ == "__main__":
    L_test1 = []
    L_test2 = []
    for i in range (10000):
        if(i%100 ==0):
            print(i)
        BB = np.random.randn(1000)
        my_correlation_BB = auto_coord(BB.tolist())
        correlation_BB = np.correlate(BB, BB,'full')[len(BB)-1:]

        lags_BB = np.arange(-len(correlation_BB)+1 , len(correlation_BB))
        L_test2.append(int(test_2(my_correlation_BB,10)))
        L_test1.append(int(test_1(my_correlation_BB)))
        plt.figure(figsize=(10, 6))

        plt.plot(my_correlation_BB, label="Générer")

        plt.show()

