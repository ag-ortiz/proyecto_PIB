import numpy as np
import matplotlib.pyplot as plt

def generate_bingo_card():
    card = np.zeros((3, 9), dtype=int)
    indices = np.random.choice(27, 15, replace=False)
    card[np.unravel_index(indices, card.shape)] = 1
    return card

def check_bingo(card):
    for row in card:
        if np.sum(row) == 9:
            return True
    for col in card.T:
        if np.sum(col) == 3:
            return True
    return False

def check_two_lines(card):
    row_lines = sum(np.sum(row) == 9 for row in card)
    col_lines = sum(np.sum(col) == 3 for col in card.T)
    return row_lines + col_lines >= 2

def simulate_bingo_round():
    card = generate_bingo_card()
    numbers_drawn = np.zeros(90, dtype=int)
    num_count = 0

    while True:
        num = np.random.randint(1, 91)
        if numbers_drawn[num-1] == 0:
            numbers_drawn[num-1] = 1
            num_count += 1
            card[card == num] = 2
            if check_bingo(card) and check_two_lines(card):
                return num_count * 10

def simulate_bingo(n_simulations):
    times = []
    for _ in range(n_simulations):
        time = simulate_bingo_round()
        times.append(time)
    return times

def generate_bingo_card():
    card = np.zeros((3, 9), dtype=int)
    indices = np.random.choice(27, 15, replace=False)
    card[np.unravel_index(indices, card.shape)] = 1
    return card

def check_bingo(card):
    for row in card:
        if np.sum(row) == 9:
            return True
    for col in card.T:
        if np.sum(col) == 3:
            return True
    return False

def check_two_lines(card):
    row_lines = sum(np.sum(row) == 9 for row in card)
    col_lines = sum(np.sum(col) == 3 for col in card.T)
    return row_lines + col_lines >= 2

def simulate_bingo_round():
    card = generate_bingo_card()
    numbers_drawn = np.zeros(99, dtype=int)
    num_count = 0

    while True:
        num = np.random.randint(1, 100)
        if numbers_drawn[num-1] == 0:
            numbers_drawn[num-1] = 1
            num_count += 1
            card[card == num] = 2
            if check_bingo(card) and check_two_lines(card):
                return num_count * 10

def simulate_bingo(n_simulations):
    times = []
    for _ in range(n_simulations):
        time = simulate_bingo_round()
        times.append(time)
    return times


n_simulations_1 = 1
times = simulate_bingo(n_simulations_1)

percentile_90_time = np.percentile(times, 90)
print(f"Tiempo necesario para obtener un bingo y dos líneas con 90% de confiabilidad: {percentile_90_time} segundos")


n_simulations = 100
times = simulate_bingo(n_simulations)

# Calculate the empirical cumulative distribution function (ECDF)
times_sorted = np.sort(times)
ecdf = np.arange(1, n_simulations + 1) / n_simulations

# Plotting the ECDF
plt.figure(figsize=(10, 6))
plt.plot(times_sorted, ecdf, marker='.', linestyle='none')
plt.xlabel('Tiempo transcurrido (segundos)')
plt.ylabel('Probabilidad acumulada')
plt.title('Tiempo transcurrido vs. Probabilidad de obtener 1 bingo y 2 líneas')
plt.grid(True)
plt.show()