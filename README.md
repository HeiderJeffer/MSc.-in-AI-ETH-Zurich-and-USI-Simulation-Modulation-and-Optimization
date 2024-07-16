<body>
<img src = "https://github-vistors-counter.onrender.com/github?username=https://github.com/HeiderJeffer/MSc.-in-AI-ETH-Zurich-and-USI-Simulation-Modulation-and-Optimization" alt = "Visitors-Counter"/>
</body>

#### Title: "Simulation, Modulation, and Optimization"
#### Master in Artificial Intelligence
#### Speaker: Heider Jeffer
#### instructor: Mehdi Jazayeri 
#### Assistant: Dr. Sasa Nesic

---
### Abstract

The abstract provides a summary of the entire paper, including the objectives, methodology, and findings.

### 1. Introduction

#### 1.1 Background

Explain the importance of simulation, modulation, and optimization in various fields such as operations research, engineering, and economics.

#### 1.2 Objectives

Outline the objectives of this research paper: to explore simulation techniques, modulation methods, and optimization algorithms in Python.

### 2. Simulation Techniques

#### 2.1 Discrete Event Simulation

Discuss discrete event simulation (DES) and its application in modeling systems with discrete, sequential events.

##### Implementation in Python:

```python
# Example code for discrete event simulation
import simpy

# Define a process
def process(env):
    while True:
        # Define process behavior
        yield env.timeout(1)  # Simulate an event occurring every second

# Create a simulation environment
env = simpy.Environment()
env.process(process(env))

# Run the simulation
env.run(until=10)  # Run the simulation for 10 time units
```

#### 2.2 Monte Carlo Simulation

Explain Monte Carlo simulation and its use in probabilistic modeling and risk analysis.

##### Implementation in Python:

```python
# Example code for Monte Carlo simulation
import random

# Define a function to estimate pi using Monte Carlo simulation
def monte_carlo_pi(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples

# Perform Monte Carlo simulation to estimate pi
estimated_pi = monte_carlo_pi(100000)
print("Estimated value of pi:", estimated_pi)
```

### 3. Modulation Methods

#### 3.1 Amplitude Modulation (AM)

Describe AM modulation and its application in telecommunications and signal processing.

##### Implementation in Python:

```python
# Example code for amplitude modulation (AM)
import numpy as np
import matplotlib.pyplot as plt

# Generate a carrier signal
fc = 100  # Carrier frequency
fs = 1000  # Sampling rate
t = np.linspace(0, 1, fs, endpoint=False)
carrier = np.sin(2 * np.pi * fc * t)

# Generate a message signal
fm = 10  # Message frequency
message = np.sin(2 * np.pi * fm * t)

# Perform AM modulation
amplitude_modulated = (1 + message) * carrier

# Plot signals
plt.figure(figsize=(10, 6))
plt.plot(t, carrier, label='Carrier Signal')
plt.plot(t, message, label='Message Signal')
plt.plot(t, amplitude_modulated, label='AM Signal')
plt.title('Amplitude Modulation')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
```

#### 3.2 Frequency Modulation (FM)

Explain FM modulation and its advantages over AM in certain applications.

##### Implementation in Python:

```python
# Example code for frequency modulation (FM)
import numpy as np
import matplotlib.pyplot as plt

# Generate a carrier signal
fc = 100  # Carrier frequency
fs = 1000  # Sampling rate
t = np.linspace(0, 1, fs, endpoint=False)
carrier = np.sin(2 * np.pi * fc * t)

# Generate a message signal
fm = 10  # Message frequency
message = np.sin(2 * np.pi * fm * t)

# Perform FM modulation
frequency_modulated = np.sin(2 * np.pi * (fc + message) * t)

# Plot signals
plt.figure(figsize=(10, 6))
plt.plot(t, carrier, label='Carrier Signal')
plt.plot(t, message, label='Message Signal')
plt.plot(t, frequency_modulated, label='FM Signal')
plt.title('Frequency Modulation')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
```

### 4. Optimization Algorithms

#### 4.1 Genetic Algorithms

Discuss genetic algorithms and their application in optimization problems inspired by natural selection.

##### Implementation in Python:

```python
# Example code for genetic algorithm
import numpy as np

# Define objective function (fitness function)
def fitness_function(x):
    return np.sum(x**2)

# Define genetic algorithm
def genetic_algorithm(population_size, num_generations):
    population = np.random.randint(0, 2, size=(population_size, 10))  # Initial population
    for generation in range(num_generations):
        fitness = np.apply_along_axis(fitness_function, 1, population)
        parents = population[np.argsort(fitness)[:population_size//2]]  # Selecting top half based on fitness
        children = []
        for _ in range(population_size - len(parents)):
            parent1, parent2 = np.random.choice(len(parents), size=2, replace=False)
            crossover_point = np.random.randint(1, 10)
            child = np.concatenate([parents[parent1][:crossover_point], parents[parent2][crossover_point:]])
            mutation_point = np.random.randint(0, 10)
            child[mutation_point] = 1 - child[mutation_point]  # Mutation
            children.append(child)
        population = np.vstack((parents, children))
    best_solution = population[np.argmin(np.apply_along_axis(fitness_function, 1, population))]
    return best_solution

# Example usage of genetic algorithm
best_solution = genetic_algorithm(population_size=100, num_generations=50)
print("Best solution found:", best_solution)
print("Fitness of best solution:", fitness_function(best_solution))
```

#### 4.2 Gradient Descent

Explain gradient descent and its application in finding the minimum of a function.

##### Implementation in Python:

```python
# Example code for gradient descent
import numpy as np

# Define the function to minimize
def function(x):
    return x**2 + 5*x + 6

# Define the derivative of the function
def derivative(x):
    return 2*x + 5

# Implement gradient descent
def gradient_descent(learning_rate, num_iterations):
    x = np.random.uniform(-10, 10)  # Initial guess
    for _ in range(num_iterations):
        x = x - learning_rate * derivative(x)
    return x

# Example usage of gradient descent
minimum = gradient_descent(learning_rate=0.1, num_iterations=100)
print("Minimum of the function:", minimum)
print("Function value at minimum:", function(minimum))
```

### 5. Conclusion

Summarize the key findings and contributions of this research paper. Discuss future directions for research in simulation, modulation, and optimization.

---
