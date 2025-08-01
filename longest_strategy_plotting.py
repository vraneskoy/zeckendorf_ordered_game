import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from functools import lru_cache

class FibonacciGame:
    def __init__(self, max_n=40000):
        self.fibs = self._generate_fibs(max_n)
        self.fib_lookup = {fib: i for i, fib in enumerate(self.fibs)}

    def _generate_fibs(self, n):
        fibs = []
        a, b = 1, 2

        if 1 <= n:
            fibs.append(1)
        if 2 <= n:
            fibs.append(2)

        while b <= n:
            if b not in fibs:  # Avoid duplicates
                fibs.append(b)
            next_fib = a + b
            a, b = b, next_fib

        return sorted(list(set(fibs)))

    def zeckendorf_decomposition(self, n):
        """Find Zeckendorf decomposition of n"""
        partition = []
        num = n

        for fib in reversed(self.fibs):
            if fib <= num:
                partition.append(fib)
                num -= fib

        return sorted(partition)

    def merge_fs(self, state, a_index, b_index):
        """Merge two elements at given indices"""
        if a_index >= len(state) or b_index >= len(state):
            return state

        result = []
        min_idx = min(a_index, b_index)
        max_idx = max(a_index, b_index)

        # Add elements before min_idx
        result.extend(state[:min_idx])

        # Add merged value
        result.append(state[a_index] + state[b_index])

        # Add elements between indices
        result.extend(state[min_idx + 1:max_idx])

        # Add remaining elements
        result.extend(state[max_idx + 1:])

        return result

    def split_pair(self, state, a_index, b_index):
        """Split identical pair according to rules"""
        if (a_index >= len(state) or b_index >= len(state) or
            a_index == b_index or state[a_index] != state[b_index]):
            return state

        a = state[a_index]
        result = []

        # Add elements before a_index
        result.extend(state[:a_index])

        if a == 2:  # Rule 3: (2,2) -> (1,3)
            result.extend([1, 3])
        elif a in self.fib_lookup and self.fib_lookup[a] >= 2:
            # Rule 2: (F_i,F_i) -> (F_{i-2},F_{i+1})
            i = self.fib_lookup[a]
            if i - 2 >= 0 and i + 1 < len(self.fibs):
                result.extend([self.fibs[i - 2], self.fibs[i + 1]])
            else:
                return state
        else:
            return state

        # Add elements after b_index
        result.extend(state[b_index + 1:])

        return result

    def swap_fs(self, state, a_index, b_index):
        result = state.copy()
        if a_index < len(result) and b_index < len(result):
            result[a_index], result[b_index] = result[b_index], result[a_index]
        return result

    def play_game_with_priority(self, initial_state):
        state = initial_state.copy()
        move_count = 0
        terminal = self.zeckendorf_decomposition(sum(state))

        while state != terminal:
            move_made = False

            # Priority 1: Switch (leftmost switch where left > right)
            for i in range(len(state) - 1):
                if state[i] > state[i + 1]:
                    state = self.swap_fs(state, i, i + 1)
                    move_count += 1
                    move_made = True
                    break

            if move_made:
                continue

            # Priority 2: Merge ones (leftmost)
            for i in range(len(state) - 1):
                if state[i] == 1 and state[i + 1] == 1:
                    state = self.merge_fs(state, i, i + 1)
                    move_count += 1
                    move_made = True
                    break

            if move_made:
                continue

            # Priority 3: Split pairs (rightmost)
            for i in range(len(state) - 2, -1, -1):
                if i + 1 < len(state) and state[i] == state[i + 1] and state[i] >= 2:
                    state = self.split_pair(state, i, i + 1)
                    move_count += 1
                    move_made = True
                    break

            if move_made:
                continue

            # Priority 4: Merge consecutive Fibonacci pairs (leftmost)
            for i in range(len(state) - 1):
                a, b = state[i], state[i + 1]
                if (a in self.fib_lookup and b in self.fib_lookup and
                    self.fib_lookup[b] - self.fib_lookup[a] == 1):
                    state = self.merge_fs(state, i, i + 1)
                    move_count += 1
                    move_made = True
                    break

            if not move_made:
                break

        return move_count

def fit_polynomial_degree_2():
    game = FibonacciGame()

    n_values = []
    moves_list = []

    print("Collecting data...")
    for n in range(2, 1001):
        initial_state = [1] * n
        moves = game.play_game_with_priority(initial_state)

        n_values.append(n)
        moves_list.append(moves)

        if n % 20 == 0:
            print(f"Processed n = {n}, moves = {moves}")

    n_array = np.array(n_values)
    moves_array = np.array(moves_list)

    # Fit polynomial of degree 2
    coefficients = np.polyfit(n_array, moves_array, 2)
    a, b, c = coefficients

    poly_predictions = np.polyval(coefficients, n_array)

    ss_res = np.sum((moves_array - poly_predictions) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((moves_array - np.mean(moves_array)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)  # R-squared

    residuals = moves_array - poly_predictions
    rmse = np.sqrt(np.mean(residuals ** 2))  # Root mean square error
    mae = np.mean(np.abs(residuals))  # Mean absolute error

    print("\n" + "="*70)
    print("POLYNOMIAL FIT RESULTS (Degree 2)")
    print("="*70)
    print(f"Fitted polynomial: moves = {a:.6f}*n² + {b:.6f}*n + {c:.6f}")
    print(f"")
    print(f"Goodness of fit:")
    print(f"  R-squared: {r_squared:.6f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"")
    print(f"Coefficient analysis:")
    print(f"  Leading coefficient (a): {a:.6f}")
    print(f"  Linear coefficient (b): {b:.6f}")
    print(f"  Constant term (c): {c:.6f}")
    print(f"")
    print(f"Asymptotic behavior:")
    print(f"  As n → ∞, moves/n² → {a:.6f}")
    print(f"  Quadratic term dominates for large n")

    # Print some specific predictions vs actual
    print(f"\nSample predictions vs actual:")
    print(f"n\tActual\tPredicted\tError\t%Error")
    print("-" * 50)
    for i in [9, 19, 49, 99, 149, 199]: 
        idx = i - 2  # Adjust for 0-based indexing (we start from n=2)
        if idx < len(n_values):
            n = n_values[idx]
            actual = moves_list[idx]
            predicted = poly_predictions[idx]
            error = actual - predicted
            percent_error = abs(error / actual) * 100
            print(f"{n}\t{actual}\t{predicted:.1f}\t\t{error:.1f}\t{percent_error:.2f}%")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # plot 1: Original data with polynomial fit
    ax1.scatter(n_values, moves_list, alpha=0.6, s=20, label='Actual moves', color='blue')
    ax1.plot(n_values, poly_predictions, 'r-', linewidth=2, label=f'Polynomial fit: {a:.4f}n² + {b:.4f}n + {c:.4f}')
    ax1.set_xlabel('n')
    ax1.set_ylabel('Number of moves')
    ax1.set_title('Polynomial Degree 2 Fit to Moves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # plot 2: Residuals
    ax2.scatter(n_values, residuals, alpha=0.6, s=20, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax2.set_xlabel('n')
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)

    # plot 3: Actual vs Predicted
    ax3.scatter(poly_predictions, moves_list, alpha=0.6, s=20, color='purple')
    min_val = min(min(poly_predictions), min(moves_list))
    max_val = max(max(poly_predictions), max(moves_list))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect fit')
    ax3.set_xlabel('Predicted moves')
    ax3.set_ylabel('Actual moves')
    ax3.set_title('Actual vs Predicted')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # plot 4: moves/n² vs n (showing convergence to leading coefficient)
    ratios = [moves/n**2 for n, moves in zip(n_values, moves_list)]
    ax4.plot(n_values, ratios, 'b-', linewidth=2, label='Actual moves/n²', alpha=0.7)
    ax4.axhline(y=a, color='red', linestyle='--', linewidth=2, label=f'Leading coefficient: {a:.6f}')
    ax4.set_xlabel('n')
    ax4.set_ylabel('moves/n²')
    ax4.set_title('Convergence to Leading Coefficient')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'coefficients': coefficients,
        'r_squared': r_squared,
        'rmse': rmse,
        'mae': mae,
        'n_values': n_values,
        'moves': moves_list,
        'predictions': poly_predictions,
        'residuals': residuals
    }

if __name__ == "__main__":
    results = fit_polynomial_degree_2()
