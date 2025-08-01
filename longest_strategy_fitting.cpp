// Fit a quadratic function to the distribution of move counts when following the longest game strategy.

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <chrono>
#include <cassert>

class OptimizedFibonacciGame {
private:
    std::vector<int> fibs;
    std::unordered_map<int, int> fib_lookup;
    
    // Cache for Zeckendorf decompositions
    mutable std::unordered_map<int, std::vector<int>> zeckendorf_cache;
    
    std::unordered_map<int, int> consecutive_fib_map;
    
    void generate_fibs(int max_n) {
        fibs.clear();
        fibs.reserve(50);
        
        if (max_n >= 1) fibs.push_back(1);
        if (max_n >= 2) fibs.push_back(2);
        
        int a = 1, b = 2;
        while (b <= max_n) {
            int next_fib = a + b;
            if (next_fib > max_n) break;
            fibs.push_back(next_fib);
            a = b;
            b = next_fib;
        }

        fib_lookup.clear();
        consecutive_fib_map.clear();
        for (int i = 0; i < fibs.size(); ++i) {
            fib_lookup[fibs[i]] = i;
            if (i > 0) {
                consecutive_fib_map[fibs[i-1]] = fibs[i]; // fibs[i-1] + fibs[i] = fibs[i+1]
            }
        }
    }
    
public:
    OptimizedFibonacciGame(int max_n = 40000) {
        generate_fibs(max_n);
        zeckendorf_cache.reserve(3000); 
    }
    
    const std::vector<int>& zeckendorf_decomposition(int n) const {
        auto it = zeckendorf_cache.find(n);
        if (it != zeckendorf_cache.end()) {
            return it->second;
        }
        
        std::vector<int> partition;
        partition.reserve(10);
        
        int num = n;
        for (int i = fibs.size() - 1; i >= 0 && num > 0; --i) {
            if (fibs[i] <= num) {
                partition.push_back(fibs[i]);
                num -= fibs[i];
            }
        }
        
        std::sort(partition.begin(), partition.end());
        return zeckendorf_cache[n] = std::move(partition);
    }
    
    // Optimized in-place operations where possible
    inline void merge_fs_inplace(std::vector<int>& state, int a_index, int b_index) {
        if (a_index >= state.size() || b_index >= state.size()) return;
        
        int min_idx = std::min(a_index, b_index);
        int max_idx = std::max(a_index, b_index);
        
        // Merge the values
        state[min_idx] = state[a_index] + state[b_index];
        
        // Remove the second element by shifting
        state.erase(state.begin() + max_idx);
    }
    
    inline bool split_pair_inplace(std::vector<int>& state, int a_index, int b_index) {
        if (a_index >= state.size() || b_index >= state.size() || 
            a_index == b_index || state[a_index] != state[b_index]) {
            return false;
        }
        
        int a = state[a_index];
        
        if (a == 2) {  // Rule 3: (2,2) -> (1,3)
            state[a_index] = 1;
            state[b_index] = 3;
            return true;
        } 
        
        auto lookup_it = fib_lookup.find(a);
        if (lookup_it != fib_lookup.end() && lookup_it->second >= 2) {
            // Rule 2: (F_i,F_i) -> (F_{i-2},F_{i+1})
            int i = lookup_it->second;
            if (i - 2 >= 0 && i + 1 < fibs.size()) {
                state[a_index] = fibs[i - 2];
                state[b_index] = fibs[i + 1];
                return true;
            }
        }
        
        return false;
    }
    
    inline void swap_fs_inplace(std::vector<int>& state, int a_index, int b_index) {
        if (a_index < state.size() && b_index < state.size()) {
            std::swap(state[a_index], state[b_index]);
        }
    }
    
    int play_game_with_priority_optimized(std::vector<int> state) {
        int move_count = 0;
        const int sum = std::accumulate(state.begin(), state.end(), 0);
        const auto& terminal = zeckendorf_decomposition(sum);
        
        // Early termination check
        if (state == terminal) return 0;
        
        // Reserve space to avoid frequent reallocations
        state.reserve(state.size() + 10);
        
        while (state != terminal) {
            bool move_made = false;
            const int size = state.size();
            
            // Priority 1: Switch
            for (int i = 0; i < size - 1; ++i) {
                if (state[i] > state[i + 1]) {
                    swap_fs_inplace(state, i, i + 1);
                    ++move_count;
                    move_made = true;
                    break;
                }
            }
            
            if (move_made) continue;
            
            // Priority 2: Merge ones (leftmost)
            for (int i = 0; i < size - 1; ++i) {
                if (state[i] == 1 && state[i + 1] == 1) {
                    merge_fs_inplace(state, i, i + 1);
                    ++move_count;
                    move_made = true;
                    break;
                }
            }
            
            if (move_made) continue;
            
            // Priority 3: Split pairs (rightmost)
            for (int i = size - 2; i >= 0; --i) {
                if (i + 1 < state.size() && state[i] == state[i + 1] && state[i] >= 2) {
                    if (split_pair_inplace(state, i, i + 1)) {
                        ++move_count;
                        move_made = true;
                        break;
                    }
                }
            }
            
            if (move_made) continue;
            
            // Priority 4: Merge consecutive Fibonacci pairs (leftmost)
            for (int i = 0; i < state.size() - 1; ++i) {
                int a = state[i];
                int b = state[i + 1];
                
                auto it = consecutive_fib_map.find(a);
                if (it != consecutive_fib_map.end() && it->second == b) {
                    merge_fs_inplace(state, i, i + 1);
                    ++move_count;
                    move_made = true;
                    break;
                }
            }
            
            if (!move_made) break;
        }
        
        return move_count;
    }
};

// polynomial fitting using matrix operations
class PolynomialFitter {
private:
    static std::vector<double> solve3x3(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
        // Calculate determinant
        double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                     A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                     A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        
        if (std::abs(det) < 1e-10) {
            throw std::runtime_error("Singular matrix");
        }
        
        std::vector<double> x(3);
        
        // x[0] = det(A0) / det where A0 has b replacing first column
        x[0] = (b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2]) +
                A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2])) / det;
        
        // x[1] = det(A1) / det where A1 has b replacing second column  
        x[1] = (A[0][0] * (b[1] * A[2][2] - A[1][2] * b[2]) -
                b[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * b[2] - b[1] * A[2][0])) / det;
        
        // x[2] = det(A2) / det where A2 has b replacing third column
        x[2] = (A[0][0] * (A[1][1] * b[2] - b[1] * A[2][1]) -
                A[0][1] * (A[1][0] * b[2] - b[1] * A[2][0]) +
                b[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) / det;
        
        return x;
    }

public:
    static std::vector<double> fit_quadratic(const std::vector<int>& x, const std::vector<int>& y) {
        const int n = x.size();
        
        // Build normal equations AtA * coeffs = Aty more efficiently
        double sum_1 = n;
        double sum_x = 0, sum_x2 = 0, sum_x3 = 0, sum_x4 = 0;
        double sum_y = 0, sum_xy = 0, sum_x2y = 0;
        
        for (int i = 0; i < n; ++i) {
            double xi = x[i];
            double yi = y[i];
            double xi2 = xi * xi;
            double xi3 = xi2 * xi;
            double xi4 = xi2 * xi2;
            
            sum_x += xi;
            sum_x2 += xi2;
            sum_x3 += xi3;
            sum_x4 += xi4;
            sum_y += yi;
            sum_xy += xi * yi;
            sum_x2y += xi2 * yi;
        }
        
        std::vector<std::vector<double>> AtA = {
            {sum_1,  sum_x,  sum_x2},
            {sum_x,  sum_x2, sum_x3},
            {sum_x2, sum_x3, sum_x4}
        };
        
        std::vector<double> Aty = {sum_y, sum_xy, sum_x2y};
        
        return solve3x3(AtA, Aty);
    }
    
    static double evaluate(const std::vector<double>& coeffs, double x) {
        return coeffs[0] + coeffs[1] * x + coeffs[2] * x * x;
    }
};

struct OptimizedResults {
    std::vector<double> coefficients;
    double r_squared;
    double rmse;
    double mae;
    std::vector<int> n_values;
    std::vector<int> moves;
    std::vector<double> predictions;
    double computation_time_ms;
};

OptimizedResults fit_polynomial_optimized() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    OptimizedFibonacciGame game;
    
    std::vector<int> n_values, moves_list;
    n_values.reserve(2999);
    moves_list.reserve(2999);
    
    std::cout << "Collecting data (optimized) up to N=3000..." << std::endl;
    
    for (int n = 2; n <= 1500; ++n) {
        std::vector<int> initial_state(n, 1);
        int moves = game.play_game_with_priority_optimized(std::move(initial_state));
        
        n_values.push_back(n);
        moves_list.push_back(moves);
        
        if (n % 100 == 0) {
            std::cout << "Processed n = " << n << ", moves = " << moves << std::endl;
        }
    }
    
    std::vector<double> coefficients = PolynomialFitter::fit_quadratic(n_values, moves_list);
    double c = coefficients[0], b = coefficients[1], a = coefficients[2];
    
    std::vector<double> predictions;
    predictions.reserve(n_values.size());
    for (int n : n_values) {
        predictions.push_back(PolynomialFitter::evaluate(coefficients, n));
    }
    
    double ss_res = 0.0, ss_tot = 0.0, sum_abs_residuals = 0.0;
    double mean_moves = std::accumulate(moves_list.begin(), moves_list.end(), 0.0) / moves_list.size();
    
    for (size_t i = 0; i < moves_list.size(); ++i) {
        double residual = moves_list[i] - predictions[i];
        ss_res += residual * residual;
        ss_tot += (moves_list[i] - mean_moves) * (moves_list[i] - mean_moves);
        sum_abs_residuals += std::abs(residual);
    }
    
    double r_squared = 1.0 - (ss_res / ss_tot);
    double rmse = std::sqrt(ss_res / moves_list.size());
    double mae = sum_abs_residuals / moves_list.size();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double computation_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Print
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "OPTIMIZED POLYNOMIAL FIT RESULTS (Degree 2, N=2 to 3000)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Fitted polynomial: moves = " << a << "*n² + " << b << "*n + " << c << std::endl;
    std::cout << std::endl;
    std::cout << "Performance:" << std::endl;
    std::cout << "  Total computation time: " << std::setprecision(2) << computation_time / 1000.0 << " seconds" << std::endl;
    std::cout << "  Average time per game: " << computation_time / n_values.size() << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << std::setprecision(6);
    std::cout << "Goodness of fit:" << std::endl;
    std::cout << "  R-squared: " << r_squared << std::endl;
    std::cout << "  RMSE: " << std::setprecision(4) << rmse << std::endl;
    std::cout << "  MAE: " << mae << std::endl;
    std::cout << std::endl;
    std::cout << std::setprecision(6);
    std::cout << "Coefficient analysis:" << std::endl;
    std::cout << "  Leading coefficient (a): " << a << std::endl;
    std::cout << "  Linear coefficient (b): " << b << std::endl;
    std::cout << "  Constant term (c): " << c << std::endl;
    std::cout << std::endl;
    std::cout << "Asymptotic behavior:" << std::endl;
    std::cout << "  As n → ∞, moves/n² → " << a << std::endl;
    std::cout << "  Quadratic term dominates for large n" << std::endl;
    
    std::cout << std::endl;
    std::cout << "Sample predictions vs actual:" << std::endl;
    std::cout << "n\tActual\tPredicted\tError\t%Error" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    std::vector<int> sample_indices = {10, 50, 100, 500, 1000, 2000, 3000};
    for (int n : sample_indices) {
        int idx = n - 2;  // Adjust for 0-based indexing (we start from n=2)
        if (idx < n_values.size()) {
            int actual = moves_list[idx];
            double predicted = predictions[idx];
            double error = actual - predicted;
            double percent_error = std::abs(error / actual) * 100;
            std::cout << n << "\t" << actual << "\t" << std::setprecision(1) << predicted 
                      << "\t\t" << std::setprecision(1) << error << "\t" << std::setprecision(2) << percent_error << "%" << std::endl;
        }
    }
    
    std::cout << std::setprecision(4);
    std::cout << std::endl;
    std::cout << "Additional insights:" << std::endl;
    std::cout << "  For large n, the algorithm requires approximately " << a << "*n² moves" << std::endl;
    std::cout << "  The linear term " << b << "*n suggests lower-order corrections" << std::endl;
    std::cout << "  The constant term " << c << " represents base overhead" << std::endl;
    
    return {coefficients, r_squared, rmse, mae, n_values, moves_list, predictions, computation_time};
}

// Benchmark comparison
void benchmark_comparison() {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "PERFORMANCE BENCHMARK" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    OptimizedFibonacciGame game;
    
    std::vector<int> test_sizes = {10, 50, 100, 200, 500, 1000};
    
    for (int n : test_sizes) {
        std::vector<int> initial_state(n, 1);
        
        auto start = std::chrono::high_resolution_clock::now();
        int moves = game.play_game_with_priority_optimized(std::move(initial_state));
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::cout << "n = " << std::setw(4) << n 
                  << ", moves = " << std::setw(6) << moves 
                  << ", time = " << std::setprecision(3) << time_ms << " ms" << std::endl;
    }
}

int main() {
    benchmark_comparison();
    OptimizedResults results = fit_polynomial_optimized();
    return 0;
}
