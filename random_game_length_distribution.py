import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
import random
import time
from collections import defaultdict
from typing import List, Tuple, Dict

# Global Fibonacci sequence
fibs: List[int] = []
fib_lookup: Dict[int, int] = {}

# Generate Fibonacci numbers up to n
def get_fibs(n: int) -> List[int]:
    result = set()
    a, b = 1, 2
    if 1 <= n:
        result.add(1)
    if 2 <= n:
        result.add(2)
    while b <= n:
        result.add(b)
        a, b = b, a + b
    return sorted(result)

# Initialize Fibonacci data
def init_fibs(max_n: int = 40000):
    global fibs, fib_lookup
    fibs = get_fibs(max_n)
    fib_lookup = {val: idx for idx, val in enumerate(fibs)}

def pretty_state(state: List[int]) -> str:
    return str(state)

# Split
def split_pair_Y(state: List[int], a_index: int, b_index: int) -> List[int]:
    if a_index >= len(state) or b_index >= len(state) or a_index == b_index:
        return state
    a = state[a_index]
    if a != state[b_index]:
        return state
    if a not in fib_lookup:
        return state
    i = fib_lookup[a]
    result = state[:a_index]
    if a == 2:
        result.extend([1, 3])
    elif i >= 2 and i + 1 < len(fibs):
        result.extend([fibs[i - 2], fibs[i + 1]])
    else:
        return state
    result.extend(state[b_index + 1:])
    return result

# Swap
def swapFs_Y(state: List[int], a_index: int, b_index: int) -> List[int]:
    if a_index < len(state) and b_index < len(state):
        state = state[:]
        state[a_index], state[b_index] = state[b_index], state[a_index]
    return state

# Merge
def mergeFs_Y(state: List[int], a_index: int, b_index: int) -> List[int]:
    if a_index >= len(state) or b_index >= len(state):
        return state
    result = []
    min_idx, max_idx = min(a_index, b_index), max(a_index, b_index)
    result.extend(state[:min_idx])
    result.append(state[a_index] + state[b_index])
    result.extend(state[min_idx + 1:max_idx])
    result.extend(state[max_idx + 1:])
    return result

# Get all possible moves
def get_children_Y(state: List[int]) -> List[Tuple[List[int], str]]:
    children = []
    l = len(state)
    for i in range(l - 1):
        a, b = state[i], state[i + 1]
        a_idx = fib_lookup.get(a)
        b_idx = fib_lookup.get(b)

        if a_idx is not None and b_idx is not None and b_idx - a_idx == 1:
            children.append((mergeFs_Y(state, i, i + 1), f"merges {a}, {b}"))
        elif a == 1 and b == 1:
            children.append((mergeFs_Y(state, i, i + 1), "merges 1, 1"))
        elif a == b:
            if a == 2:
                children.append((split_pair_Y(state, i, i + 1), "splits 2,2 -> 1,3"))
            elif a > 2:
                children.append((split_pair_Y(state, i, i + 1), f"splits {a},{a}"))
        elif a > b:
            children.append((swapFs_Y(state, i, i + 1), f"swaps {a}, {b}"))
    return children

# state is terminal?
def is_terminal(state: List[int]) -> bool:
    return not get_children_Y(state)

#Game Stats
class GameStats:
    def __init__(self):
        self.length_distribution = defaultdict(int)
        self.player1_wins = 0
        self.player2_wins = 0
        self.total_games = 0

    def add_game(self, length: int, winner: int):
        self.length_distribution[length] += 1
        if winner == 1:
            self.player1_wins += 1
        elif winner == 2:
            self.player2_wins += 1
        self.total_games += 1

    def print_statistics(self):
        print("\n=== GAME STATISTICS ===")
        print(f"Total games simulated: {self.total_games}")
        print(f"Player 1 wins: {self.player1_wins} ({100.0 * self.player1_wins / self.total_games:.2f}%)")
        print(f"Player 2 wins: {self.player2_wins} ({100.0 * self.player2_wins / self.total_games:.2f}%)\n")
        print("Game Length Distribution:")
        print("Length\tCount\tPercentage")
        for length, count in sorted(self.length_distribution.items()):
            print(f"{length}\t{count}\t{100.0 * count / self.total_games:.2f}%")
        avg_length = sum(length * count for length, count in self.length_distribution.items()) / self.total_games
        print(f"\nAverage game length: {avg_length:.2f}")

# Simulate a game with random moves
def simulate_game(n: int, verbose: bool = False) -> Tuple[int, int]:
    state = [1] * n
    move_count = 0
    current_player = 1
    if verbose:
        print(f"Starting game with N={n}")
        print(f"Initial state: {pretty_state(state)}")
    while not is_terminal(state):
        moves = get_children_Y(state)
        if not moves:
            break
        selected = random.choice(moves)
        if verbose:
            print(f"Player {current_player} selects move: {selected[1]}")
        state = selected[0]
        move_count += 1
        current_player = 2 if current_player == 1 else 1
        if verbose:
            print(f"New state: {pretty_state(state)}\n")
    winner = 2 if current_player == 1 else 1
    if verbose:
        print(f"Game ended after {move_count} moves.")
        print(f"Final state: {pretty_state(state)}")
        print(f"Winner: Player {winner}")
    return move_count, winner

# multiple simulations
def run_simulations(n: int, num_simulations: int) -> GameStats:
    stats = GameStats()
    print(f"Running {num_simulations} simulations with N={n}...")
    for i in range(num_simulations):
        if i % max(1, num_simulations // 10) == 0:
            print(f"Progress: {100 * i // num_simulations}%")
        length, winner = simulate_game(n)
        stats.add_game(length, winner)
    print("Progress: 100%")
    return stats

# Plot statistics with log-normal fitting
def plot_statistics_with_lognormal(stats: GameStats, mu: float = None, sigma: float = None):
    observed_data = []
    for length, count in stats.length_distribution.items():
        observed_data.extend([length] * count)
    observed_data = np.array(observed_data)
    
    if mu is None or sigma is None:
        # Fit log-normal distribution to observed data
        shape, loc, scale = scipy_stats.lognorm.fit(observed_data, floc=0)
        fitted_mu = np.log(scale)
        fitted_sigma = shape
        print(f"Fitted parameters: μ = {fitted_mu:.3f}, σ = {fitted_sigma:.3f}")
    else:
        fitted_mu = mu
        fitted_sigma = sigma
        print(f"Using provided parameters: μ = {fitted_mu:.3f}, σ = {fitted_sigma:.3f}")
    
    # Create log-normal distribution with fitted/provided parameters
    lognorm_dist = scipy_stats.lognorm(s=fitted_sigma, scale=np.exp(fitted_mu))
    
    lengths = sorted(stats.length_distribution.keys())
    counts = [stats.length_distribution[length] for length in lengths]
    percentages = [100 * c / stats.total_games for c in counts]
    
    # theoretical values for comparison
    x_range = np.linspace(max(1, min(lengths) - 2), max(lengths) + 2, 1000)
    
    plt.figure(figsize=(15, 10))
    
    # Bar plot for count distribution with log-normal overlay
    plt.subplot(2, 2, 1)
    plt.bar(lengths, counts, color='skyblue', alpha=0.7, label='Observed Data', width=0.8)
    
    theoretical_pdf = lognorm_dist.pdf(x_range)
    # Scale by total games and approximate bin width
    bin_width = 1.0
    theoretical_counts = theoretical_pdf * stats.total_games * bin_width
    plt.plot(x_range, theoretical_counts, 'r-', linewidth=2, 
             label=f'Log-Normal (μ={fitted_mu:.2f}, σ={fitted_sigma:.2f})')
    
    plt.xlabel("Game Length")
    plt.ylabel("Count")
    plt.title("Distribution of Game Lengths (Counts)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.bar(lengths, percentages, color='salmon', alpha=0.7, label='Observed Data', width=0.8)
    
    theoretical_percentages = theoretical_pdf * 100 * bin_width
    plt.plot(x_range, theoretical_percentages, 'r-', linewidth=2, 
             label=f'Log-Normal (μ={fitted_mu:.2f}, σ={fitted_sigma:.2f})')
    
    plt.xlabel("Game Length")
    plt.ylabel("Percentage (%)")
    plt.title("Distribution of Game Lengths (Percentage)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot
    plt.subplot(2, 2, 3)
    scipy_stats.probplot(observed_data, dist=lognorm_dist, plot=plt)
    plt.title("Q-Q Plot: Observed vs Log-Normal")
    plt.grid(True, alpha=0.3)
    
    # Histogram with log-normal PDF overlay (norm)
    plt.subplot(2, 2, 4)
    plt.hist(observed_data, bins=max(10, len(lengths)), density=True, alpha=0.7, 
             color='lightgreen', label='Observed Histogram')
    plt.plot(x_range, lognorm_dist.pdf(x_range), 'r-', linewidth=2, 
             label=f'Log-Normal PDF (μ={fitted_mu:.2f}, σ={fitted_sigma:.2f})')
    plt.xlabel("Game Length")
    plt.ylabel("Density")
    plt.title("Normalized Histogram with Log-Normal PDF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate goodness of fit stats
    observed_data_sorted = np.sort(observed_data)
    theoretical_cdf = lognorm_dist.cdf(observed_data_sorted)
    empirical_cdf = np.arange(1, len(observed_data_sorted) + 1) / len(observed_data_sorted)
    
    # Kolmogorov-Smirnov test
    ks_statistic = np.max(np.abs(theoretical_cdf - empirical_cdf))
    ks_stat_scipy, p_value = scipy_stats.kstest(observed_data, lognorm_dist.cdf)
    
    print(f"\n=== LOG-NORMAL FIT ASSESSMENT ===")
    print(f"Parameters: μ = {fitted_mu:.3f}, σ = {fitted_sigma:.3f}")
    print(f"Theoretical mean: {lognorm_dist.mean():.2f}")
    print(f"Observed mean: {np.mean(observed_data):.2f}")
    print(f"Theoretical std: {lognorm_dist.std():.2f}")
    print(f"Observed std: {np.std(observed_data):.2f}")
    print(f"Kolmogorov-Smirnov statistic: {ks_statistic:.4f}")
    print(f"KS test p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("Result: Data is consistent with log-normal distribution (p > 0.05)")
    else:
        print("Result: Data significantly deviates from log-normal distribution (p ≤ 0.05)")
    
    return ks_statistic

# Main logic
def main():
    init_fibs()
    print("Fibonacci Game Statistics Calculator with Log-Normal Fitting")
    print("===========================================================")
    n = 150
    num_simulations = 10000
    print(f"Configuration:\n- Starting with N = {n}\n- Number of simulations = {num_simulations}\n- Players select moves randomly\n")
    
    print("\n=== FULL SIMULATION ===")
    start_time = time.time()
    stats = run_simulations(n, num_simulations)
    end_time = time.time()
    stats.print_statistics()
    print(f"\nSimulation completed in {int((end_time - start_time) * 1000)} ms")
    
    # Plot with log-normal fitting (auto-fit parameters)
    print("\n=== FITTING LOG-NORMAL DISTRIBUTION ===")
    ks_stat = plot_statistics_with_lognormal(stats)

if __name__ == "__main__":
    main()
