import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / e_x.sum(axis=0, keepdims=True)

def main(Nh1, Nh2):

    print(f"For hidden layer nodes: {Nh2}x{Nh1}")
    train_data = np.loadtxt("FashionMNIST2_train_1000.csv", delimiter=',').astype(int)
    # Preprocess training data
    no_classes = 10
    train_data = train_data[train_data[:, 0] < no_classes]
    np.random.shuffle(train_data)

    X_train = (train_data[:, 1:].T / 255.0).astype(np.float32)
    N_train = X_train.shape[1]
    Y_train = np.zeros((10, N_train), dtype=np.float32)
    labels = train_data[:, 0].astype(int)
    Y_train[labels, np.arange(N_train)] = 1

    # Initialize weights and biases
    np.random.seed(42)
    Ni = 784
    No = no_classes

    no_nodes = [Ni, Nh1, Nh2, No]
    W = []          
    b = []
    for i in range(1, len(no_nodes)):
        W.append(np.random.rand(no_nodes[i-1], no_nodes[i]) - 0.5)
        b.append(np.zeros((no_nodes[i], 1)))

    # Activation functions
    sig2 = lambda x: 1 / (1 + np.exp(-x))
    dsig2 = lambda x: sig2(x) * (1 - sig2(x))
    sig3 = softmax

    # Hyperparameters
    N_ep = 150
    lr = 0.00965

    ################################################################################################################################################################################
    # Training the Network
    ################################################################################################################################################################################

    pivec = np.zeros(N_ep)
    for epoch in range(N_ep):
        mixup = np.random.permutation(N_train)
        for j in mixup:
            # Forward pass
            a1 = X_train[:, j].reshape(-1, 1)
            a = [a1]
            n = []
            for i in range(len(W) - 1):
                n.append(W[i].T @ a[i] + b[i])
                a.append(sig2(n[i]))
            n.append(W[-1].T @ a[-1] + b[-1])
            a.append(sig3(n[-1]))

            # Backpropagation
            err = Y_train[:, j].reshape(-1, 1) - a[-1]
            S = [-err]
            for i in range(len(W)-1):
                S.append(dsig2(n[-2-i]) * (W[-1-i] @ S[i]))
            
            # Update weights and biases
            for i in range(len(W)):
                W[-1-i] -= lr * (a[-2-i] @ S[i].T)
                b[-1-i] -= lr * S[i]

        # Calculate performance index
        epoch_loss = 0
        for j in range(N_train):
            a1 = X_train[:, j].reshape(-1, 1)
            a = [a1]
            n = []
            for i in range(len(W) - 1):
                n.append(W[i].T @ a[i] + b[i])
                a.append(sig2(n[i]))
            n.append(W[-1].T @ a[-1] + b[-1])       
            a.append(sig3(n[-1]))

            xent = -np.sum(Y_train[:, j] * np.log(a[-1].flatten() + 1e-8))
            epoch_loss += xent/N_train
        pivec[epoch] = epoch_loss

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    plt.figure()
    plt.plot(pivec)
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Cross-entropy loss')
    plt.title(fr'Training Loss for $\texttt{{Nh1}} = {Nh1}$ and $\texttt{{Nh2}} = {Nh2}$')

    plt.savefig(f'training_loss_{Nh2}x{Nh1}_{no_classes}class.pdf', bbox_inches='tight')
    plt.close()

    ################################################################################################################################################################################
    # Evaluating the network on unseen FMNIST data
    ################################################################################################################################################################################

    # Save and load trained matrices with no regularization
    for idx, w in enumerate(W):
        np.save(f'W_{idx}_{Nh2}x{Nh1}_{no_classes}class.npy', w)

    pretrained = 0
    if pretrained == 1:
        W_load = [np.load(f'W_{idx}_{Nh2}x{Nh1}_{no_classes}class.npy') for idx in range(len(W))]
        W = W_load.copy()



    # Preprocess test data
    test_data = np.loadtxt("FashionMNIST2_test_1000.csv", delimiter=',').astype(int)
    test_data = test_data[test_data[:, 0] < no_classes]
    np.random.shuffle(test_data)

    X_test = (test_data[:, 1:].T / 255.0).astype(np.float32)
    N_test = X_test.shape[1]
    Y_test = np.zeros((10, N_test), dtype=np.float32)
    test_labels = test_data[:, 0].astype(int)
    Y_test[test_labels, np.arange(N_test)] = 1

    # Test predictions
    On = np.zeros((10, N_test))
    test_correct = 0

    pivec2 = 0
    for j in range(N_test):
        a1 = X_test[:, j].reshape(-1, 1)
        a = [a1]
        n = []
        for i in range(len(W) - 1):
            n.append(W[i].T @ a[i] + b[i])
            a.append(sig2(n[i]))
        n.append(W[-1].T @ a[-1] + b[-1])
        a.append(sig3(n[-1]))

        if np.argmax(a[-1]) == np.argmax(Y_test[:, j]):
            test_correct += 1
        # print(f"Predicted: {np.argmax(a[-1])}, Actual: {np.argmax(Y_test[:, j])} \n Accuracy: {test_correct}/{j+1}={test_correct/(j+1)*100:.2f}%")

        xent = -np.sum(Y_test[:, j] * np.log(a[-1].flatten() + 1e-8))
        pivec2 += xent/N_train

    ################################################################################################################################################
    # print(f"\nTest loss: {pivec2}")
    print(f"Test accuracy: {test_correct/N_test*100:.2f}%")
    ################################################################################################################################################


    # unique, counts = np.unique(test_data[:, 0], return_counts=True)
    ################################################################################################################################################
    # print(dict(zip(unique, counts)))
    # print()
    ################################################################################################################################################

    ################################################################################################################################################################################
    # Singular Value Decomposition to observe effects of rank reduction on performance
    ################################################################################################################################################################################

    k=1
    U,S,VT = np.linalg.svd(W[k].copy())
    plt.figure()
    S_ordered = np.sort(S.copy())[::-1]
    ################################################################################################################################################
    # print(f"\nThese are the singular values of W{k}")
    # print(S)
    # print()
    ################################################################################################################################################

    plt.hist(S_ordered, bins=20, edgecolor='black', linewidth=0.5)
    plt.xlabel(r"Singular Values")
    plt.ylabel(r"Frequency")
    plt.title(fr"Distribution of Singular Values in W{k}")

    plt.savefig(f"singval_{Nh2}x{Nh1}_{no_classes}class.pdf", bbox_inches = 'tight')
    plt.close()
    range_s = range(len(S))

    # Measuring test set accuracy of the network at every rank of W_k
    flag = 0
    Testing_Accuracy = []
    for j in range_s:
        sum = np.zeros(W[k].shape)   
        for i in range_s:
            if S[i] > S[j]:
                u_i = U[:,i].reshape(-1,1)
                vt_i = VT[i,:].reshape(-1,1)
                matrix = S[i] * u_i @ vt_i.T
                sum += matrix
        if flag == 0:
            print(f"Rank of W{k}:")
            print(f"Rank of W{k} after reduction:", np.linalg.matrix_rank(sum))
            flag = 1

        
        W_lowrank = W.copy()
        W_lowrank[k] = sum.copy()
        test_correct =0

        # Re-evaulation
        for j in range(N_test):
            a1 = X_test[:, j].reshape(-1, 1)
            a = [a1]
            n = []
            for i in range(len(W) - 1):
                n.append(W_lowrank[i].T @ a[i] + b[i])
                a.append(sig2(n[i]))
            n.append(W_lowrank[-1].T @ a[-1] + b[-1])
            a.append(sig3(n[-1]))
            if np.argmax(a[-1]) == np.argmax(Y_test[:, j]):
                test_correct += 1
            
        Testing_Accuracy.append(test_correct/N_test)

    y_value_at_x1 = Testing_Accuracy[1]
    y_value_at_x2 = Testing_Accuracy[no_classes]
    half_rank = int(len(S)/2)
    y_value_at_halfrank = Testing_Accuracy[half_rank-1]

    plt.figure()
    plt.axhline(y=y_value_at_x1, color='r', linestyle='--', label=fr"Accuracy at $x=1$: {y_value_at_x1:.2f}")
    plt.axhline(y=y_value_at_x2, color='g', linestyle='--', label=fr"Accuracy at $x={no_classes}$: {y_value_at_x2:.2f}")
    plt.axhline(y=y_value_at_halfrank, color='b', linestyle='--', label=fr"Accuracy at $x={half_rank}$: {y_value_at_halfrank:.2f}")
    plt.plot(range_s, Testing_Accuracy)
    plt.xlabel(fr"Rank of W{k}")
    plt.ylabel(r"Accuracy")
    plt.title(fr"{len(no_nodes)} Layers (W{k})")
    plt.legend()

    plt.savefig(f"accuracy_{Nh2}x{Nh1}_{no_classes}class.pdf", bbox_inches='tight')
    plt.close()
    print()

    ################################################################################################################################################################################
    # Confirming the random nature of the lower ranks by comparing the eigenvector entry distribution of the WTW matrix to the Porter-Thomas distribution
    ################################################################################################################################################################################

    from scipy.stats import ks_2samp

    k = 1
    k_m = no_nodes[k]
    k_n = no_nodes[k + 1]


    WTW = W[k].copy().T @ W[k].copy()
    ################################################################################################################################################
    # print("\n This is the shape of my WTW matrix")
    # print(WTW.shape)
    # print()
    ################################################################################################################################################

    eigvalues, eigvecs = np.linalg.eigh(WTW)
    eigvec_collection = (eigvecs[:, 1:10].copy()).flatten()
    eigvec = ((eigvec_collection)**2)

    ################################################################################################################################################
    # print(f"min eigvec: {np.min(eigvec)}")
    # print(f"max eigvec: {np.max(eigvec)}")
    # print(f"mean eigvec: {np.mean(eigvec)}")
    # print(f"std eigvec: {np.std(eigvec)}\n")
    ################################################################################################################################################

    def pt_pdf(x):
        return (np.sqrt(k_n/(2*np.pi*x)))*np.exp((-k_n*x)/2)

    z = np.linspace(0.001,np.max(eigvec), 500)
    porter_thomas = pt_pdf(z)


    # Now we will use Acceptance-Rejection Sampling to compare the two distributions
    # We will sample from the Porter-Thomas distribution and compare it to the eigenvector distribution
    length_eigvec = len(eigvec)
    porter_thomas_sample = []
    lambda_ = 1/np.mean(eigvec)

    def exponen(x):
        return lambda_*np.exp(-lambda_*x)

    upper_bound = np.max(pt_pdf(np.linspace(0.001,np.max(eigvec)))/exponen(np.linspace(0.001,np.max(eigvec))))


    # print("Upper Bound:", upper_bound)
    # print(length_eigvec)

    while len(porter_thomas_sample) < length_eigvec:
        u = np.random.uniform(0, 1)
        y = np.random.exponential(scale=1/lambda_)
        # print(len(porter_thomas_sample))
        
        if u <= (pt_pdf(y)/(exponen(y)*upper_bound)):
            porter_thomas_sample.append(y)

    stat, p_val = ks_2samp(porter_thomas_sample,eigvec)

    ################################################################################################################################################
    # print()
    # print("KS Statistic:", stat)
    # print("P-value:", p_val)

    # if p_val > 0.05:
    #     print("Fail to reject the null (similar distributions)")
    # else:
    #     print("Reject the null (different distribution)")
    ################################################################################################################################################

    # The plot
    plt.figure()
    plt.hist(eigvec, bins=30, edgecolor='black',density=True, label=r"Eigenvector Entries")
    plt.plot(z, porter_thomas, 'r-', label=r"Porter-Thomas Density")
    plt.title(fr"Porter-Thomas against Eigenvectors (W{k})")
    plt.ylabel(r"Density")
    plt.xlabel(r"Eigenvector Entries")
    plt.legend()

    plt.savefig(f"porter_{Nh2}x{Nh1}_{no_classes}class.pdf", bbox_inches='tight')
    plt.close()

    # # Sample from the Porter-Thomas distribution
    # plt.hist(porter_thomas_sample, bins=30, edgecolor='black',density=True)
    # plt.title(f"Porter-Thomas Sample")
    # plt.ylabel("Density")
    # plt.xlabel("Eigenvector Entries")
    # plt.show()

    ################################################################################################################################################################################
    # Confirming the random nature of the lower ranks by comparing the eigenvalue distribution of the S matrix to the Marcenko-Pastur distribution
    ################################################################################################################################################################################

    k_1 = 1
    W_k_copy = W[k_1].copy()
    W_k_norm = (W_k_copy - np.mean(W_k_copy))/(np.std(W_k_copy))

    # This is the number of rows and columns in the matrix
    k_1_m = no_nodes[k_1]
    k_1_n = no_nodes[k_1 + 1]

    S_matrix = (1/k_1_m)* W_k_norm.T @ W_k_norm

    eigsvals, eigsvecs = np.linalg.eigh(S_matrix)
    lambda_vals = np.linspace(min(eigsvals), max(eigsvals), 200)

    aspectratio = k_1_n/k_1_m
    lambdaplus = (1 + np.sqrt(aspectratio))**2
    lambdaminus = (1 - np.sqrt(aspectratio))**2

    def mp_pdf(x):
        return (1/(2*np.pi*aspectratio*x))*np.sqrt((lambdaplus - x)*(x - lambdaminus))

    lambda_vals = np.linspace(lambdaminus, lambdaplus, 200)
    marcenko_pastur = mp_pdf(lambda_vals)

    # The plot
    plt.hist(eigsvals[:-1], bins=30, edgecolor='black',density=True, label=r"Eigenvalues")
    plt.plot(lambda_vals, marcenko_pastur, 'r-', label=r"Marcenko-Pastur Density")
    plt.title(fr"Marcenko-Pastur Density against Eigenvalues (W{k_1})")
    plt.ylabel(r"Density")
    plt.xlabel(r"Eigenvalues")
    plt.legend()

    plt.savefig(f"marcenko_{Nh2}x{Nh1}_{no_classes}class.pdf", bbox_inches='tight')
    plt.close()

    length_eigvals = len(eigsvals[:-no_classes])

    # Sampling from Marcenko-Pastur through the eigenvalues of a GOE ensemble
    goe = np.random.normal(size=(k_1_m,k_1_n))
    S = (1/k_1_m)*np.dot(goe.T, goe)
    marcenko_pastur_sample = np.linalg.eigvals(S)

    # # A histogram of the samples from the Marcenko-Pastur density
    # plt.hist(marcenko_pastur_sample, bins=30, edgecolor='black',density=True)
    # plt.title("Marcenko-Pastur Sample")
    # plt.ylabel("Density")
    # plt.xlabel("Eigenvalues")
    # plt.show()

    stat, p_val = ks_2samp(marcenko_pastur_sample,eigsvals[:-no_classes])

    ################################################################################################################################################
    # print("\n KS Statistic:", stat)
    # print("P-value:", p_val)

    # if p_val > 0.05:
    #     print("Fail to reject the null (similar distributions)")
    # else:
    #     print("Reject the null (different distribution)")
    ################################################################################################################################################

if __name__ == "__main__":
    # Nh = [30, 50, 75, 100, 250, 500, 750, 1000]
    # for nh in Nh:
    #     main(nh,nh)
    main(100, 50)
