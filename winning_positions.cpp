/*
 * Fibonacci Game Rules Implementation
 * 
 * Starting state: (F_1, F_1, ..., F_1) where F_1 = 1
 * Fibonacci sequence: F_1 = 1, F_2 = 2, F_{i+1} = F_i + F_{i-1}
 * 
 * Legal moves:
 * 1. Merging: (F_i, F_{i+1}) -> F_{i+2}
 * 2. Splitting: (F_i, F_i) -> (F_{i-2}, F_{i+1}) for i > 2
 * 3. Splitting Twos: (F_2, F_2) -> (F_1, F_3)
 * 4. Splitting Ones: (F_1, F_1) -> F_2
 * 5. Switching: (F_i, F_j) -> (F_j, F_i) if i > j
 * 
 * The game ends when no more legal moves can be performed.
 */

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <functional>
#include <climits>

using namespace std;

// Global Fibonacci sequence
vector<int> fibs;
unordered_map<int, int> fib_lookup;

// Hash function for vector<int>
struct VectorHash {
    size_t operator()(const vector<int>& v) const {
        size_t hash = 0;
        hash_combine(hash, v.size());
        for (int i : v) {
            hash_combine(hash, i);
        }
        return hash;
    }
    
private:
    void hash_combine(size_t& seed, size_t value) const {
        seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
};

// Custom hash for pair<vector<int>, bool>
struct PairHash {
    size_t operator()(const pair<vector<int>, bool>& p) const {
        size_t h1 = VectorHash{}(p.first);
        size_t h2 = hash<bool>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// Custom hash for pair<int, int>
struct IntPairHash {
    size_t operator()(const pair<int, int>& p) const {
        return hash<int>{}(p.first) ^ (hash<int>{}(p.second) << 1);
    }
};

using StateMap = unordered_map<vector<int>, vector<pair<vector<int>, string>>, VectorHash>;
using MemoMap = unordered_map<pair<vector<int>, bool>, int, PairHash>;

string pretty_state(const vector<int>& state) {
    string result = "[";
    for (size_t i = 0; i < state.size(); i++) {
        if (i > 0) result += ", ";
        result += to_string(state[i]);
    }
    result += "]";
    return result;
}

// Generate Fibonacci numbers up to n
vector<int> get_fibs(int n) {
    vector<int> result;
    set<int> unique_fibs;
    int a = 1, b = 2; 
    
    if (1 <= n) unique_fibs.insert(1);
    if (2 <= n) unique_fibs.insert(2);
    
    while (b <= n) {
        unique_fibs.insert(b);
        int next = a + b;
        a = b;
        b = next;
    }
    
    result.assign(unique_fibs.begin(), unique_fibs.end());
    return result;
}

// Initialize global Fibonacci data
void init_fibs(int max_n = 40000) {
    fibs = get_fibs(max_n);
    fib_lookup.clear();
    fib_lookup.reserve(fibs.size());
    for (size_t i = 0; i < fibs.size(); i++) {
        fib_lookup[fibs[i]] = static_cast<int>(i);
    }
}

// Zeckendorf decomposition
vector<int> zeckendorf_decomposition(int n) {
    vector<int> partition;
    partition.reserve(10);
    int num = n;
    
    for (auto it = fibs.rbegin(); it != fibs.rend(); ++it) {
        int fib = *it;
        if (fib <= num) {
            partition.push_back(fib);
            num -= fib;
        }
    }
    
    sort(partition.begin(), partition.end());
    return partition;
}

// Fixed memoization for fibonacci_partitions
unordered_map<pair<int, int>, vector<vector<int>>, IntPairHash> partition_memo;

// Generate all Fibonacci partitions with better memoization
vector<vector<int>> fibonacci_partitions(int n, int max_fib = INT_MAX) {
    if (n == 0) return {{}};
    
    pair<int, int> key = {n, max_fib};
    auto it = partition_memo.find(key);
    if (it != partition_memo.end()) {
        return it->second;
    }
    
    vector<vector<int>> partitions;
    partitions.reserve(100);
    
    for (int fib : fibs) {
        if (fib > n || fib > max_fib) break;
        
        vector<vector<int>> sub_partitions = fibonacci_partitions(n - fib, fib);
        for (const auto& sub_partition : sub_partitions) {
            vector<int> new_partition;
            new_partition.reserve(sub_partition.size() + 1);
            new_partition.push_back(fib);
            new_partition.insert(new_partition.end(), sub_partition.begin(), sub_partition.end());
            partitions.push_back(std::move(new_partition));
        }
    }
    
    partition_memo[key] = partitions;
    return partitions;
}

// Split operation
vector<int> splitPair_Y(const vector<int>& state, int a_index, int b_index) {
    if (a_index >= state.size() || b_index >= state.size() || a_index == b_index) {
        return state;
    }
    
    int a = state[a_index];
    if (a != state[b_index]) return state;
    
    vector<int> result;
    result.reserve(state.size() + 1);

    for (int i = 0; i < a_index; i++) {
        result.push_back(state[i]);
    }

    auto fib_it = fib_lookup.find(a);
    if (fib_it == fib_lookup.end()) return state;
    
    int i = fib_it->second;
    if (a == 2) {  // Special case
        result.push_back(1);
        result.push_back(3);
    }
    else if (i >= 2) {  // General case
        if (i-2 >= 0 && i+1 < fibs.size()) {
            result.push_back(fibs[i-2]);
            result.push_back(fibs[i+1]);
        }
        else return state;
    }
    else return state;
    
    // Copy elements after the pair
    for (int j = b_index + 1; j < state.size(); j++) {
        result.push_back(state[j]);
    }
    
    return result;
}

vector<int> swapFs_Y(vector<int> state, int a_index, int b_index) {
    if (a_index < state.size() && b_index < state.size()) {
        swap(state[a_index], state[b_index]);
    }
    return state;
}

vector<int> mergeFs_Y(const vector<int>& state, int a_index, int b_index) {
    if (a_index >= state.size() || b_index >= state.size()) {
        return state; 
    }
    
    vector<int> result;
    result.reserve(state.size());
    
    int min_idx = min(a_index, b_index);
    int max_idx = max(a_index, b_index);

    result.insert(result.end(), state.begin(), state.begin() + min_idx);
    
    result.push_back(state[a_index] + state[b_index]);
    
    result.insert(result.end(), state.begin() + min_idx + 1, state.begin() + max_idx);
    
    result.insert(result.end(), state.begin() + max_idx + 1, state.end());
    
    return result;
}

vector<pair<vector<int>, string>> get_children_Y(const vector<int>& state) {
    vector<pair<vector<int>, string>> children;
    children.reserve(state.size() * 2);
    
    size_t l = state.size();
    
    // Check for operations between consec pairs
    for (size_t i = 0; i < l - 1; i++) {
        int a = state[i];
        int b = state[i + 1];
        
        auto a_it = fib_lookup.find(a);
        auto b_it = fib_lookup.find(b);
        
        // Merging
        if (b_it != fib_lookup.end() && a_it != fib_lookup.end() && 
            b_it->second - a_it->second == 1) {
            children.emplace_back(mergeFs_Y(state, i, i + 1), 
                               "merges " + to_string(a) + ", " + to_string(b));
        }
        // Splitting Ones
        else if (a == 1 && b == 1) {
            children.emplace_back(mergeFs_Y(state, i, i + 1), 
                               "merges 1, 1");
        }
        // Splitting identical pairs
        else if (a == b) {
            if (a == 2) { 
                children.emplace_back(splitPair_Y(state, i, i + 1),
                                   "splits 2,2 -> 1,3");
            }
            else if (a > 2) { 
                children.emplace_back(splitPair_Y(state, i, i + 1),
                                   "splits " + to_string(a) + "," + to_string(a));
            }
        }
        // Switching
        else if (a > b) {
            children.emplace_back(swapFs_Y(state, i, i + 1), 
                               "swaps " + to_string(a) + ", " + to_string(b));
        }
    }
    
    return children;
}

// Calculate multinomial coefficient
long long multinomial_coefficient(const vector<int>& counts) {
    long long result = 1;
    int total = 0;
    
    for (int count : counts) {
        total += count;
    }
    
    for (int count : counts) {
        for (int i = 1; i <= count; i++) {
            if (result > LLONG_MAX / (total - i + 1)) {
                return LLONG_MAX; // Overflow protection
            }
            result = result * (total - i + 1) / i;
        }
        total -= count;
    }
    
    return result;
}

// Game tree generation
StateMap Y_game_tree(int n) {
    StateMap adjacency_list;
    vector<vector<int>> reg_partitions = fibonacci_partitions(n);

    size_t estimated_states = 0;
    for (const auto& partition : reg_partitions) {
        map<int, int> counts;
        for (int x : partition) counts[x]++;
        
        vector<int> count_values;
        for (const auto& p : counts) {
            count_values.push_back(p.second);
        }
        
        long long perms = multinomial_coefficient(count_values);
        estimated_states += static_cast<size_t>(min(perms, 100000LL)); // Cap estimation
    }
    
    adjacency_list.reserve(estimated_states);
    
    // Generate all permutations of each partition
    for (vector<int> partition : reg_partitions) {
        sort(partition.begin(), partition.end());
        do {
            adjacency_list[partition] = get_children_Y(partition);
        } while (next_permutation(partition.begin(), partition.end()));
    }
    
    vector<int> terminal = zeckendorf_decomposition(n);
    adjacency_list[terminal] = {};
    
    return adjacency_list;
}

// Function to find all winning positions for player 1
void find_all_winning_positions(StateMap& game_tree, const vector<int>& initial_state, 
                               const vector<int>& terminal_state, int n) {
    MemoMap memo;
    memo.reserve(game_tree.size() * 2);
    
    function<int(const vector<int>&, bool)> minimax = [&](const vector<int>& state, bool is_player1_turn) -> int {
        pair<vector<int>, bool> key = {state, is_player1_turn};
        auto memo_it = memo.find(key);
        if (memo_it != memo.end()) return memo_it->second;
        
        if (state == terminal_state) {
            return memo[key] = 0; // Error state
        }
        
        auto game_it = game_tree.find(state);
        if (game_it == game_tree.end()) {
            return memo[key] = (is_player1_turn ? 2 : 1);
        }
        
        bool found_winning_move = false;
        for (const auto& move : game_it->second) {
            // Check if this move leads to terminal state
            if (move.first == terminal_state) {
                found_winning_move = true;
                break;
            }
            // Otherwise, recursively check the position
            int result = minimax(move.first, !is_player1_turn);
            if ((is_player1_turn && result == 1) || (!is_player1_turn && result == 2)) {
                found_winning_move = true;
                break;
            }
        }
        
        int winner;
        if (is_player1_turn) {
            winner = (found_winning_move ? 1 : 2);
        } else {
            winner = (found_winning_move ? 2 : 1);
        }
        
        return memo[key] = winner;
    };
    
    // determine the overall winner
    int overall_winner = minimax(initial_state, true);
    
    if (overall_winner == 2) {
        cout << "Player 2 can force a win from the initial position of " << n << " 1s." << endl;
        cout << "No winning positions exist for Player 1." << endl;
        return;
    }
    
    // Collect all states that Player 1 can move TO and still win
    cout << "Player 1 can win from the initial position of " << n << " 1s." << endl;
    cout << "All positions Player 1 can move TO and force a win (Player 2's turn, but Player 1 wins):" << endl;
    cout << "======================================================================================" << endl;
    
    vector<vector<int>> winning_states;
    for (const auto& memo_entry : memo) {
        const auto& state = memo_entry.first.first;
        bool is_player1_turn = memo_entry.first.second;
        int winner = memo_entry.second;
        
        // We want states where it's player 2's turn but player 1 still wins
        if (!is_player1_turn && winner == 1) {
            winning_states.push_back(state);
        }
    }
    
    // Sort winning states
    sort(winning_states.begin(), winning_states.end(), [](const vector<int>& a, const vector<int>& b) {
        if (a.size() != b.size()) return a.size() < b.size();
        return a < b;
    });
    
    int count = 0;
    for (const auto& state : winning_states) {
        cout << ++count << ". " << pretty_state(state) << endl;
    }
    
    cout << "\nTotal positions Player 1 can move TO and force a win: " << count << endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    init_fibs();
    
    cout << "Fibonacci Game Analysis - All Winning Positions for Player 1" << endl;
    cout << "=============================================================" << endl;
    
    for (int nn = 24; nn <= 24; nn++) {
        cout << "\n--- Analysis for n = " << nn << " ---" << endl;
        
        vector<int> initial_state(nn, 1);
        vector<int> terminal_state = zeckendorf_decomposition(nn);
        
        cout << "Initial state: " << pretty_state(initial_state) << endl;
        cout << "Terminal state: " << pretty_state(terminal_state) << endl;
        cout << endl;
        
        auto game_tree = Y_game_tree(nn);
        find_all_winning_positions(game_tree, initial_state, terminal_state, nn);
        
        cout << "\n" << string(60, '=') << endl;
    }
    
    return 0;
}
