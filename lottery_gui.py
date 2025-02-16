import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
from itertools import combinations
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.arima.model import ARIMA
import warnings
import random

warnings.filterwarnings("ignore")

# Prime numbers in 1-49 range for statistical analysis
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

class EnhancedLotteryAnalyzer:
    def __init__(self, data_path):
        self.raw_data = pd.read_csv(data_path, header=None)  # Store original data
        self.data = None
        self.clean_data()
        self.validate_data()
        self.all_numbers = None
        self.frequencies = None
        self.positional_freq = defaultdict(list)
        self.analyze_all()

    def clean_data(self):
        """Handle missing values and invalid entries"""
        # Remove rows with any missing values
        cleaned = self.raw_data.dropna()
        
        # Convert to numeric and filter valid numbers
        cleaned = cleaned.apply(pd.to_numeric, errors='coerce')
        cleaned = cleaned.dropna()
        
        # Remove duplicates
        cleaned = cleaned.drop_duplicates()
        
        # Filter valid numbers (1-49)
        mask = (cleaned >= 1) & (cleaned <= 49)
        cleaned = cleaned[mask.all(axis=1)]
        
        # Reset index and store
        cleaned.reset_index(drop=True, inplace=True)
        self.data = cleaned

    def validate_data(self):
        """Check if we have valid data after cleaning"""
        if len(self.data) == 0:
            raise ValueError("No valid data remaining after cleaning")
        if len(self.data.columns) != 6:
            raise ValueError("Invalid format - need exactly 6 numbers per row")
        if (self.data < 1).any().any() or (self.data > 49).any().any():
            raise ValueError("All numbers must be between 1 and 49")

    def analyze_all(self):
        # Basic frequency analysis
        self.all_numbers = self.data.values.flatten()
        self.frequencies = pd.Series(self.all_numbers).value_counts().reindex(range(1,50), fill_value=0)
        
        # Positional analysis
        for pos in range(6):
            self.positional_freq[pos] = self.data[pos].value_counts().reindex(range(1,50), fill_value=0)
            
        # Combination analysis
        self.pair_counts = self.count_combinations(2)
        self.triplet_counts = self.count_combinations(3)
        
        # Statistical properties
        self.odd_even_ratio = self.calculate_odd_even()
        self.high_low_ratio = self.calculate_high_low()
        self.prime_stats = self.calculate_prime_percentage()

    def count_combinations(self, combo_size):
        counts = defaultdict(int)
        for _, row in self.data.iterrows():
            sorted_row = sorted(row)
            for combo in combinations(sorted_row, combo_size):
                counts[combo] += 1
        return counts

    def calculate_odd_even(self):
        odds = sum(1 for num in self.all_numbers if num % 2 != 0)
        return odds / len(self.all_numbers)

    def calculate_high_low(self):
        highs = sum(1 for num in self.all_numbers if num > 24)
        return highs / len(self.all_numbers)

    def calculate_prime_percentage(self):
        prime_count = sum(1 for num in self.all_numbers if num in PRIMES)
        return prime_count / len(self.all_numbers)

    def chi_square_test(self):
        observed = self.frequencies.values
        expected = np.full_like(observed, len(self.all_numbers)/49)
        _, p, _, _ = chi2_contingency([observed, expected])
        return p

class LotteryPredictor:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.time_series = self.prepare_time_series()

    def prepare_time_series(self):
        ts_data = []
        for idx, row in self.analyzer.data.iterrows():
            ts_data.extend(row.values)
        return pd.Series(ts_data)

    def arima_forecast(self, steps=10):
        model = ARIMA(self.time_series, order=(5,1,0))
        model_fit = model.fit()
        return model_fit.forecast(steps=steps)

    def cluster_pairs(self, n_clusters=5):
        pairs = list(self.analyzer.pair_counts.keys())
        frequencies = [self.analyzer.pair_counts[p] for p in pairs]
        
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = clustering.fit_predict(list(pairs))
        
        return {cluster: [pair for pair, c in zip(pairs, clusters) if c == cluster] 
                for cluster in set(clusters)}

class AdvancedLotteryGenerator:
    def __init__(self, analyzer, predictor):
        self.analyzer = analyzer
        self.predictor = predictor
        self.weights = self.calculate_dynamic_weights()

    def calculate_dynamic_weights(self):
        base_weights = self.analyzer.frequencies.values
        positional_weights = np.mean([self.analyzer.positional_freq[pos].values 
                                    for pos in range(6)], axis=0)
        return (base_weights * 0.7 + positional_weights * 0.3)

    def generate_combination(self, constraints=None):
        while True:
            numbers = np.random.choice(
                range(1, 50),
                size=6,
                replace=False,
                p=self.weights/np.sum(self.weights)
            )
            if self.validate_constraints(numbers, constraints):
                return sorted(numbers)

    @staticmethod
    def validate_constraints(numbers, constraints):
        if not constraints:
            return True
            
        if 'exclude' in constraints and any(n in constraints['exclude'] for n in numbers):
            return False
            
        odds = sum(1 for n in numbers if n % 2 != 0)
        if 'min_odds' in constraints and odds < constraints['min_odds']:
            return False
        if 'max_evens' in constraints and (6 - odds) > constraints['max_evens']:
            return False
            
        return True

    def generate_optimized_tickets(self, num_tickets, constraints=None):
        return [self.generate_combination(constraints) for _ in range(num_tickets)]

class EnhancedLotteryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lotto Optimizer Pro 6/49")
        self.analyzer = None
        self.predictor = None
        self.generator = None
        self.constraints = {}
        self.ticket_price = 5
        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Analysis Tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")
        self.create_analysis_tab()
        
        # Generation Tab
        self.generation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.generation_frame, text="Generation")
        self.create_generation_tab()
        
        # Simulation Tab
        self.simulation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.simulation_frame, text="Simulation")
        self.create_simulation_tab()

    def create_analysis_tab(self):
        ttk.Button(self.analysis_frame, text="Load Data", 
                 command=self.load_data).pack(pady=5)
                 
        # Visualization canvas
        self.figure = plt.Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.analysis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Statistics display
        self.stats_text = tk.Text(self.analysis_frame, height=10, width=80)
        self.stats_text.pack(pady=5)

    def create_generation_tab(self):
        constraints_frame = ttk.LabelFrame(self.generation_frame, text="Constraints")
        constraints_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        ttk.Label(constraints_frame, text="Min Odd Numbers:").grid(row=0, column=0, sticky=tk.W)
        self.min_odds = ttk.Entry(constraints_frame, width=5)
        self.min_odds.grid(row=0, column=1)
        
        ttk.Label(constraints_frame, text="Max Even Numbers:").grid(row=1, column=0, sticky=tk.W)
        self.max_evens = ttk.Entry(constraints_frame, width=5)
        self.max_evens.grid(row=1, column=1)
        
        ttk.Label(constraints_frame, text="Exclude Numbers:").grid(row=2, column=0, sticky=tk.W)
        self.exclude_nums = ttk.Entry(constraints_frame, width=15)
        self.exclude_nums.grid(row=2, column=1)
        
        ttk.Button(constraints_frame, text="Apply", 
                 command=self.update_constraints).grid(row=3, column=0, columnspan=2, pady=5)
        
        # Generation controls
        control_frame = ttk.Frame(self.generation_frame)
        control_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Budget (RON):").pack()
        self.budget_entry = ttk.Entry(control_frame)
        self.budget_entry.pack(pady=3)
        
        ttk.Label(control_frame, text="OR").pack()
        
        ttk.Label(control_frame, text="Number of Tickets:").pack()
        self.ticket_entry = ttk.Entry(control_frame)
        self.ticket_entry.pack(pady=3)
        
        ttk.Button(control_frame, text="Generate", 
                 command=self.generate_tickets).pack(pady=5)
        ttk.Button(control_frame, text="Export PDF", 
                 command=self.export_pdf).pack(pady=5)
        
        # Results display
        self.results_text = tk.Text(self.generation_frame, width=50)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def create_simulation_tab(self):
        ttk.Button(self.simulation_frame, text="Run Simulation",
                 command=self.run_simulation).pack(pady=5)
        self.sim_results = tk.Text(self.simulation_frame, wrap=tk.WORD)
        self.sim_results.pack(fill=tk.BOTH, expand=True)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.analyzer = EnhancedLotteryAnalyzer(file_path)
                self.predictor = LotteryPredictor(self.analyzer)
                self.generator = AdvancedLotteryGenerator(self.analyzer, self.predictor)
                
                # Show data cleaning report
                original_count = len(self.analyzer.raw_data)
                valid_count = len(self.analyzer.data)
                msg = f"Data loaded successfully!\n\n" \
                      f"Original rows: {original_count}\n" \
                      f"Valid rows after cleaning: {valid_count}\n" \
                      f"Invalid rows removed: {original_count - valid_count}"
                messagebox.showinfo("Data Report", msg)
                
                self.show_analysis()
            except Exception as e:
                error_msg = f"Data loading failed: {str(e)}".replace('%', '%%')
                messagebox.showerror("Error", error_msg)

    def show_analysis(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Create pair frequency heatmap
        heatmap_data = np.zeros((49,49))
        for (a, b), freq in self.analyzer.pair_counts.items():
            heatmap_data[a-1][b-1] = freq
            heatmap_data[b-1][a-1] = freq
            
        im = ax.imshow(heatmap_data, cmap="hot", interpolation='nearest')
        self.figure.colorbar(im, ax=ax)
        ax.set_title("Pair Frequency Heatmap")
        ax.set_xlabel("Numbers")
        ax.set_ylabel("Numbers")
        self.canvas.draw()
        
        # Update statistics report
        stats_report = (
            f"Statistical Analysis Report:\n"
            f"• Chi-Square p-value: {self.analyzer.chi_square_test():.4f}\n"
            f"• Odd/Even Ratio: {self.analyzer.odd_even_ratio:.2f}\n"
            f"• High/Low Ratio: {self.analyzer.high_low_ratio:.2f}\n"
            f"• Prime Numbers: {self.analyzer.prime_stats:.2%}\n"
            f"• Hot Numbers: {self.analyzer.frequencies.nlargest(5).index.tolist()}\n"
            f"• Cold Numbers: {self.analyzer.frequencies.nsmallest(5).index.tolist()}\n"
        )
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_report)

    def update_constraints(self):
        self.constraints = {}
        try:
            if self.min_odds.get():
                self.constraints['min_odds'] = int(self.min_odds.get())
            if self.max_evens.get():
                self.constraints['max_evens'] = int(self.max_evens.get())
            if self.exclude_nums.get():
                self.constraints['exclude'] = [int(n) for n in self.exclude_nums.get().split(",")]
        except ValueError:
            messagebox.showerror("Error", "Invalid constraint values")

    def generate_tickets(self):
        try:
            if self.budget_entry.get():
                budget = float(self.budget_entry.get())
                num_tickets = int(budget // self.ticket_price)
            else:
                num_tickets = int(self.ticket_entry.get())
                
            tickets = self.generator.generate_optimized_tickets(num_tickets, self.constraints)
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Generated {num_tickets} tickets:\n\n")
            for i, ticket in enumerate(tickets, 1):
                self.results_text.insert(tk.END, f"Ticket {i}: {ticket}\n")
                
        except ValueError:
            messagebox.showerror("Error", "Invalid input values")

    def run_simulation(self):
        if not self.analyzer:
            messagebox.showerror("Error", "Load data first!")
            return
            
        test_tickets = self.generator.generate_optimized_tickets(100)
        wins = defaultdict(int)
        
        for _ in range(1000):  # Simulate 1000 draws
            drawn = random.sample(range(1,50), 6)
            for ticket in test_tickets:
                matches = len(set(ticket) & set(drawn))
                wins[matches] += 1
                
        report = "Simulation Results (1000 draws):\n"
        for matches in sorted(wins.keys(), reverse=True):
            report += f"• {matches} matches: {wins[matches]} wins\n"
            
        self.sim_results.delete(1.0, tk.END)
        self.sim_results.insert(tk.END, report)

    def export_pdf(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf")
        if file_path:
            c = canvas.Canvas(file_path, pagesize=letter)
            text = self.results_text.get(1.0, tk.END)
            
            # Header
            c.setFont("Helvetica-Bold", 16)
            c.drawString(100, 750, "Lottery Ticket Report")
            
            # Body
            c.setFont("Helvetica", 12)
            y_position = 700
            for line in text.split("\n"):
                if y_position < 50:
                    c.showPage()
                    y_position = 750
                c.drawString(100, y_position, line)
                y_position -= 15
                
            c.save()
            messagebox.showinfo("Success", f"PDF saved to {file_path}")

if __name__ == "__main__":
    app = EnhancedLotteryApp()
    app.mainloop()