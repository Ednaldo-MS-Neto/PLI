import numpy as np
import random
import math

def read_cost_matrix(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # Remove a linha de cabeçalho e obtenha a dimensão
        dimension_line = lines[0].strip()
        dimension = int(dimension_line.split(':')[1].strip())
        
        # Encontre o índice da linha que contém a matriz de custos
        cost_index = next(i for i, line in enumerate(lines) if line.strip() == 'COST') + 1
        
        # Filtra as linhas que contêm a matriz de custos
        cost_lines = lines[cost_index:]
        
        # Cria a matriz de custo a partir das linhas, ignorando linhas com comprimento inválido
        cost_matrix = []
        for line in cost_lines:
            row = list(map(int, line.split()))
            if len(row) == dimension:
                cost_matrix.append(row)
        
        # Converte para um numpy array
        cost_matrix = np.array(cost_matrix)
        
        # Verifica se a matriz tem o formato correto
        if cost_matrix.shape[0] != dimension or cost_matrix.shape[1] != dimension:
            raise ValueError("A matriz de custo não corresponde à dimensão especificada.")
        
    return dimension, cost_matrix

def calculate_total_distance(tour, cost_matrix):
    return sum(cost_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)) + cost_matrix[tour[-1], tour[0]]

def swap_2opt(tour, i, j):
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour

def swap_3opt(tour, i, j, k):
    if i < j < k:
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:k+1][::-1] + tour[k+1:]
    elif i < k < j:
        new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:j+1][::-1] + tour[j+1:]
    elif j < i < k:
        new_tour = tour[:j] + tour[j:i+1][::-1] + tour[i+1:k+1][::-1] + tour[k+1:]
    else:
        new_tour = tour[:i] + tour[i:j+1] + tour[j+1:k+1] + tour[k+1:]
    return new_tour

def variable_neighborhood_descent(tour, cost_matrix):
    neighborhoods = [swap_2opt, swap_3opt]  # Lista de vizinhanças
    improved = True
    
    while improved:
        improved = False
        for neighborhood in neighborhoods:
            for i in range(len(tour)):
                for j in range(i + 1, len(tour)):
                    if neighborhood == swap_2opt:
                        new_tour = neighborhood(tour, i, j)
                        new_distance = calculate_total_distance(new_tour, cost_matrix)
                        if new_distance < calculate_total_distance(tour, cost_matrix):
                            tour = new_tour
                            improved = True
                    elif neighborhood == swap_3opt:
                        for k in range(j + 1, len(tour)):
                            new_tour = neighborhood(tour, i, j, k)
                            new_distance = calculate_total_distance(new_tour, cost_matrix)
                            if new_distance < calculate_total_distance(tour, cost_matrix):
                                tour = new_tour
                                improved = True
                                
    return tour

def simulated_annealing(cost_matrix, initial_temp=10000, cooling_rate=0.995, num_iterations=1000):
    n = len(cost_matrix)
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_distance = calculate_total_distance(current_tour, cost_matrix)
    
    best_tour = current_tour
    best_distance = current_distance
    
    temperature = initial_temp
    
    for _ in range(num_iterations):
        i, j = sorted(random.sample(range(n), 2))
        new_tour = swap_2opt(current_tour, i, j)
        new_distance = calculate_total_distance(new_tour, cost_matrix)
        
        if new_distance < best_distance:
            best_tour = new_tour
            best_distance = new_distance
        
        if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temperature):
            current_tour = new_tour
            current_distance = new_distance
        
        # Try 3-opt move
        i, j, k = sorted(random.sample(range(n), 3))
        new_tour = swap_3opt(current_tour, i, j, k)
        new_distance = calculate_total_distance(new_tour, cost_matrix)
        
        if new_distance < best_distance:
            best_tour = new_tour
            best_distance = new_distance
        
        if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temperature):
            current_tour = new_tour
            current_distance = new_distance
        
        temperature *= cooling_rate
    
    # Apply VND to the best tour found
    best_tour = variable_neighborhood_descent(best_tour, cost_matrix)
    best_distance = calculate_total_distance(best_tour, cost_matrix)
    
    return best_tour, best_distance

# Usage example:
if __name__ == "__main__":
    # Load the matrix from a file
    dimension, cost_matrix = read_cost_matrix('instances/cost_matrix52.txt')
    
    # Perform simulated annealing with VND
    best_tour, best_distance = simulated_annealing(cost_matrix)
    
    # Print results
    print("Best tour:", best_tour)
    print("Best distance:", best_distance)
