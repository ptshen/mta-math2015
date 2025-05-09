import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


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
    
    # Raise each diagonal entry of D to the nth power
    D_power = np.diag(np.diag(D) ** n)
    
    return P @ D_power @ P_inv


print(average_markov_map())
print(raise_power(average_markov_map(), 1))


def plot_markov_map(input_file, save_path):
    # taken from parse.py
    average = 364

    M = average_markov_map()

    with open(input_file, 'r') as f:
        data = json.load(f)    

 

    timestamps = [datetime.fromisoformat(point[0]) for point in data]
    values = [point[1] for point in data]

    x_data = np.array([timestamp.timestamp() for timestamp in timestamps])
    x_fit = np.linspace(min(x_data), max(x_data), 1000)


    # Convert x_fit back to datetime for plotting
    x_fit_dates = [datetime.fromtimestamp(x) for x in x_fit]


    # population according to seed and markov map
    seed = np.array([[data[0][1]], [average - data[0][1]]])
    markov_y_values = [(raise_power(M, i) @ seed)[0][0] for i in range(len(x_fit_dates))]
    #print(markov_y_values)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, values, 'b-', label='MTA Late Trains', alpha=0.6)
    plt.plot(x_fit_dates, markov_y_values, 'r--', label=f'MDP', linewidth=2)
    plt.title(f'Number of Late Trains Over Time with MDP Function')
    plt.xlabel('Time')
    plt.ylabel('Number of Late Trains')
    plt.grid(True)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)  # Saves to the current working directory

    # Show the plot
    plt.show()
    


plot_markov_map('train/train_weekday.json', 'markov_map.png')

    

# seed = np.array([[360], [70]])

# M = average_markov_map()

# M_n = raise_power(M, 6*24)

# print(M_n)


# print(M_n @ seed)


def plot_graph(file):
    with open(file, 'r') as f:
        data = json.load(f)
    
    # Extract timestamps and values
    timestamps = [datetime.fromisoformat(point[0]) for point in data]
    values = [point[1] for point in data]

     # Create x values for the fitted function
    x_data = np.array([timestamp.timestamp() for timestamp in timestamps])
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    

    # Convert x_fit back to datetime for plotting
    x_fit_dates = [datetime.fromtimestamp(x) for x in x_fit]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, values, 'b-', label='MTA Late Trains', alpha=0.6)
    plt.title(f'Number of Late Trains Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Late Trains')
    plt.grid(True)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()
    

def save_graph(option, file, save_path):
    with open(file, 'r') as f:
        data = json.load(f)
    
    # Extract timestamps and values
    timestamps = [datetime.fromisoformat(point[0]) for point in data]
    values = [point[1] for point in data]

     # Create x values for the fitted function
    x_data = np.array([timestamp.timestamp() for timestamp in timestamps])
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    
    # Get the fitted coefficients
    if option.lower() == 'mix':
        a, b, c, d, e = fit_mix_function(file)
        y_fit = evaluate_mix_function(x_fit, [a, b, c, d, e])
    elif option.lower() == 'sin3':
        a, b, c, d = fit_sin3_function(file)
        y_fit = evaluate_sin3_function(x_fit, [a, b, c, d])
    elif option.lower() == 'sine':
        a, b = fit_sine_function(file)
        y_fit = evaluate_sine_function(x_fit, [a, b])
    
   
    
    # Evaluate the fitted function
    if option.lower() == 'mix':
        y_fit = evaluate_mix_function(x_fit, [a, b, c, d, e])
    elif option.lower() == 'sin3':
        y_fit = evaluate_sin3_function(x_fit, [a, b, c, d])
    elif option.lower() == 'sine':
        y_fit = evaluate_sine_function(x_fit, [a, b])
    
    # Convert x_fit back to datetime for plotting
    x_fit_dates = [datetime.fromtimestamp(x) for x in x_fit]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, values, 'b-', label='MTA Late Trains', alpha=0.6)
    plt.plot(x_fit_dates, y_fit, 'r--', label=f'{option.capitalize()} Function', linewidth=2)
    plt.title(f'Number of Late Trains Over Time with {option.capitalize()} Function')
    plt.xlabel('Time')
    plt.ylabel('Number of Late Trains')
    plt.grid(True)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)  # Saves to the current working directory

    # Show the plot
    plt.show()



def fit_mix_function(file):
    
    with open(file, 'r') as f:
        data = json.load(f)

    
    freq = 2 * np.pi / (24 * 3600)  # 24-hour period in radians/second


    timestamps = [datetime.fromisoformat(point[0]) for point in data]
    values = [point[1] for point in data]
    
    x_data = np.array([timestamp.timestamp() for timestamp in timestamps])
    y_data = np.array(values)

    X = np.column_stack([
        np.ones_like(x_data),
        np.sin(freq * x_data),
        np.cos(freq * x_data),
        np.sin(2 * freq * x_data),
        np.cos(2 * freq * x_data),
    ])
    
    # Solve the least squares problem
    coefficients, residuals, rank, s = np.linalg.lstsq(X, y_data, rcond=None)
    
    return coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4]

def evaluate_mix_function(x, coefficients):
    freq = 2 * np.pi / (24 * 3600)
    return (
        coefficients[0] 
        + coefficients[1] * np.sin(freq * x) 
        + coefficients[2] * np.cos(freq * x)
        + coefficients[3] * np.sin(2 * freq * x)
        + coefficients[4] * np.cos(2 * freq * x))


def fit_sin3_function(file):
    
    with open(file, 'r') as f:
        data = json.load(f)

    
    freq = 2 * np.pi / (24 * 3600)  # 24-hour period in radians/second


    timestamps = [datetime.fromisoformat(point[0]) for point in data]
    values = [point[1] for point in data]
    
    x_data = np.array([timestamp.timestamp() for timestamp in timestamps])
    y_data = np.array(values)

    X = np.column_stack([
        np.ones_like(x_data),
        np.sin(freq * x_data),
        np.cos(freq * x_data),
        np.sin(3 * freq * x_data),
    ])
    
    # Solve the least squares problem
    coefficients, residuals, rank, s = np.linalg.lstsq(X, y_data, rcond=None)
    
    return coefficients[0], coefficients[1], coefficients[2], coefficients[3]

def evaluate_sin3_function(x, coefficients):
    freq = 2 * np.pi / (24 * 3600)
    return (
        coefficients[0] 
        + coefficients[1] * np.sin(freq * x) 
        + coefficients[2] * np.cos(freq * x)
        + coefficients[3] * np.sin(3 * freq * x))

def fit_sine_function(file):
    
    with open(file, 'r') as f:
        data = json.load(f)

    freq = 2 * np.pi / (24 * 3600)  # 24-hour period in radians/second
    timestamps = [datetime.fromisoformat(point[0]) for point in data]
    values = [point[1] for point in data]
    
    x_data = np.array([timestamp.timestamp() for timestamp in timestamps])
    y_data = np.array(values)

    X = np.column_stack([
        np.ones_like(x_data),
        np.sin(freq * x_data)
    ])
    
    # Solve the least squares problem
    coefficients, residuals, rank, s = np.linalg.lstsq(X, y_data, rcond=None)
    
    return coefficients[0], coefficients[1]

def evaluate_sine_function(x, coefficients):
    freq = 2 * np.pi / (24 * 3600)
    return (
        coefficients[0] 
        + coefficients[1] * np.sin(freq * x))


#plot_train_data('Sin3', 'filtered_train_1_pops.json', 'sin3_fit.png')

#save_graph('Sine', 'train/train_weekday.json', 'graphs/train/train_weekday_sine.png')













        
    





    