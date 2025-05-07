import json
import numpy as np


# average of 364 trips per 10 minutes (from average_trips() method in parse.py)
def average_markov_map():

    average_map = np.array([[0, 0],
                            [0, 0]])

    average_trips = 364

    

    with open('train_1_pops.json', 'r') as f:
        data = json.load(f)

    jumps = len(data) - 1
    
    for i in range(jumps):
        late = data[i][1]
        on_time = average_trips - late

        next_late = data[i+1][1]
        next_on_time = average_trips - next_late

        A = np.array([[1, 1],
              [late, on_time]])
        
        if np.linalg.det(A) != 0: 
            #average_map = average_map + np.dot(np.linalg.inv(A), np.array([[1,1], [next_late, next_on_time]]))
            average_map = average_map + np.linalg.inv(A) @ np.array([[1,1], [next_late, next_on_time]])
        
    return average_map / jumps


def diagonalize_matrix(A):
    # Ensure A is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square to diagonalize.")
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Form the diagonal matrix D and matrix P of eigenvectors
    D = np.diag(eigenvalues)
    P = eigenvectors

    # Check if P is invertible
    if np.linalg.matrix_rank(P) < A.shape[0]:
        print("Matrix is not diagonalizable: eigenvectors are not linearly independent.")
        return None, None, None

    P_inv = np.linalg.inv(P)

    # Reconstruct A for verification
    A_reconstructed = P @ D @ P_inv

    return D, P, P_inv


def raise_power(A, n):
    D, P, P_inv = diagonalize_matrix(A)
    D_power = np.diag(D**n)
    return P @ D_power @ P_inv




seed = np.array([[360], [70]])

M = average_markov_map()

D, P, P_inv = diagonalize_matrix(average_markov_map())

M_n = raise_power(M, 1)

print(M_n @ seed)














        
    





    