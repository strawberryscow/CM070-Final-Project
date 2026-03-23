import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

warnings.filterwarnings("ignore")

#data classes

@dataclass
class Chromosome:
    """A single trading strategy encoded as a gene array."""
    genes: np.ndarray
    fitness: float = -np.inf
    trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0

    @property
    def entry_threshold(self) -> float:
        return float(np.clip(self.genes[0], 0.50, 0.90))
    
    @property
    def stop_loss_pct(self) -> float:
        return float(np.clip(self.genes[1], 0.01, 0.10))
    
    @property
    def take_profit_pct(self) -> float:
        return float(np.clip(self.genes[2], 0.02, 0.25))
    
    @property
    def holding_days(self) -> int:
        return int(np.clip(round(self.genes[3]), 1, 10))
                   
    def describe(self) -> str:
        return (
            f"entry_threshold={self.entry_threshold:.2f} "
            f"stop_loss={self.stop_loss_pct*100:.1f}% "
            f"take_profit={self.take_profit_pct*100:.1f}% "
            f"holding_days={self.holding_days}d "
        )
    
@dataclass
class GAResult:
    """Full results of on GA run for one asset."""
    asset: str
    best_chromosome: Chromosome
    best_fitness_history: List[float]
    mean_fitness_history: List[float]
    benchmark_sharpe: float
    benchmark_return: float
    generations: int
    population_size: int

#backtesting

def run_backtest(
        prices: np.ndarray,
        probabilities: np.ndarray,
        entry_threshold: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        holding_days: int
) -> Dict:
    n = len(prices)
    capital = 1.0
    daily_capital =[capital]
    trade_returns = []
    in_position = False
    entry_price = 0.0
    days_held = 0

    for i in range(n - 1):
        if not in_position:
            if probabilities[i] >= entry_threshold:
                in_position = True
                entry_price = prices[i]
                days_held = 0
        else:
            days_held += 1
            current_price = prices[i]
            pnl_pct = (current_price - entry_price) / entry_price

            exit_trade = False

            if pnl_pct <= -stop_loss_pct:
                exit_trade = True
            elif pnl_pct >= take_profit_pct:
                exit_trade = True
            elif days_held >= holding_days:
                exit_trade = True
            
            if exit_trade:
                trade_return = pnl_pct
                capital *= (1 + trade_return)
                trade_returns.append(trade_return)
                in_position = False
                days_held = 0

        daily_capital.append(capital)
    
    daily_capital = np.array(daily_capital)
    daily_returns = np.diff(daily_capital) / daily_capital[:-1]

    if in_position:
        final_pnl = (prices[-1] - entry_price) / entry_price
        trade_returns.append(final_pnl)
    
    total_return = daily_capital[-1] - 1.0

    if len(daily_returns) > 1 and np.std(daily_returns) > 1e-10:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    if trade_returns:
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
    else:
        win_rate = 0.0

    #max drawdown
    peak = daily_capital[0]
    max_dd = 0.0
    for val in daily_capital:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
    
    return {
        "daily_returns": daily_returns,
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "trades": len(trade_returns),
        "max_drawdown": max_dd
    }

def buy_and_hold_benchmark(prices: np.ndarray) -> Dict:
    daily_returns = np.diff(prices) / prices[:-1]
    total_return = (prices[-1] / prices[0]) - 1.0
    if len(daily_returns) > 1 and np.std(daily_returns) > 1e-10:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0
    peak = prices[0] 
    max_dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak -p) / peak
        if dd > max_dd:
            max_dd = dd
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd
    }

#Genetic algorithm

class TradingStrategyGA:

    GENE_BOUNDS = [
        (0.50, 0.90),  #entry threshold
        (0.01, 0.10),  #stop_loss_pct
        (0.02, 0.25),  #take_profit_pct
        (1.0, 10.0),   #holding_days
    ]
    N_GENES = 4

    def __init__(
        self, 
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.80,
        mutation_rate: float = 0.15,
        mutation_sigma: float = 0.05,
        tournament_k: int = 5,
        elite_n: int = 3,
        random_state: int = 42
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.tournament_k = tournament_k
        self.elite_n = elite_n
        self.rng = np.random.default_rng(random_state)

#initialisation

    def _random_chromosome(self) -> Chromosome:
        genes = np.array([
            self.rng.uniform(lo, hi)
            for lo, hi in self.GENE_BOUNDS
        ])
        return Chromosome(genes=genes)
    
    def _initialize_population(self) -> List[Chromosome]:
        return [self._random_chromosome() for _ in range(self.population_size)]
    
#Fitness evaluation

    def _evaluate(
        self, 
        chromosome: Chromosome,
        prices: np.ndarray,
        probabilities: np.ndarray        
    ) -> Chromosome:
        result = run_backtest(
            prices, probabilities,
            chromosome.entry_threshold,
            chromosome.stop_loss_pct,
            chromosome.take_profit_pct,
            chromosome.holding_days
        )

        if result["trades"] < 3:
            chromosome.fitness = -1.0
        else:
            chromosome.fitness = result["sharpe_ratio"]
        
        chromosome.trades = result["trades"]
        chromosome.win_rate = result["win_rate"]
        chromosome.total_return = result["total_return"]
        chromosome.max_drawdown = result["max_drawdown"]
        return chromosome
    
    def _evaluate_population(
        self, 
        population: List[Chromosome],
        prices: np.ndarray,
        probabilities: np.ndarray
    ) -> List[Chromosome]:
        return [self._evaluate(c, prices, probabilities) for c in population]

    #Selection

    def _tournament_selection(self, population: List[Chromosome]) -> Chromosome:
        contestants = self.rng.choice(len(population), size=self.tournament_k, replace=False)
        winner = max(contestants, key=lambda i: population[i].fitness)
        return population[winner]
    
    #Crossover
    def _crossover(
            self, 
            parent1: Chromosome,
            parent2: Chromosome
        ) -> Tuple[Chromosome, Chromosome]:
            if self.rng.random() > self.crossover_rate:
                return (
                    Chromosome(genes=parent1.genes.copy()),
                    Chromosome(genes=parent2.genes.copy())
                )
            point = self.rng.integers(1, self.N_GENES)
            child1_genes = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
            child2_genes = np.concatenate([parent2.genes[:point], parent1.genes[point:]])
            return Chromosome(genes=child1_genes), Chromosome(genes=child2_genes)
    
    #mutation

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        genes = chromosome.genes.copy()
        for i, (lo, hi) in enumerate(self.GENE_BOUNDS):
                if self.rng.random() < self.mutation_rate:
                    noise = self.rng.normal(0, self.mutation_sigma * (hi - lo))
                    genes[i] = np.clip(genes[i] + noise, lo, hi)
        chromosome.genes = genes
        return chromosome
    
    #main loop

    def evolve(
        self, 
        prices: np.ndarray,
        probabilities: np.ndarray,
        asset_name: str = "",
        verbose: bool = True
    ) ->  GAResult:

        if verbose:
            print(f"\n{'='*60}")
            print(f"Genetic Algorithm - {asset_name}")
            print(f"{'='*60}\n")
            print(f"Population size: {self.population_size} | Generations: {self.generations}")
            print(f"Validation window: {len(prices)} days")

        population = self._initialize_population()
        population = self._evaluate_population(population, prices, probabilities)
        
        best_overall = max(population, key=lambda c: c.fitness)
        best_overall = Chromosome(
            genes=best_overall.genes.copy(),
            fitness=best_overall.fitness,
            trades=best_overall.trades,
            win_rate=best_overall.win_rate,
            total_return=best_overall.total_return,
            max_drawdown=best_overall.max_drawdown
        )

        best_fitness_history = []
        mean_fitness_history = []

        for gen in range(self.generations):
            sorted_pop = sorted(population, key=lambda c: c.fitness, reverse=True)
            elites = [Chromosome(genes=c.genes.copy(), fitness=c.fitness) for c in sorted_pop[:self.elite_n]]
            
            new_population = []
            while len(new_population) < self.population_size - self.elite_n:
                p1 = self._tournament_selection(population)
                p2 = self._tournament_selection(population)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.extend([c1, c2])
            
            population = elites + new_population[:self.population_size - self.elite_n]
            population = self._evaluate_population(population, prices, probabilities)

            #track best and mean fitness
            gen_best = max(population, key=lambda c: c.fitness)
            gen_fitnesses = [c.fitness for c in population if c.fitness > -1.0]
            gen_mean = np.mean(gen_fitnesses) if gen_fitnesses else -1.0

            best_fitness_history.append(gen_best.fitness)
            mean_fitness_history.append(gen_mean)

            if gen_best.fitness > best_overall.fitness:
                best_overall = Chromosome(
                    genes=gen_best.genes.copy(),
                    fitness=gen_best.fitness,
                    trades=gen_best.trades,
                    win_rate=gen_best.win_rate,
                    total_return=gen_best.total_return,
                    max_drawdown=gen_best.max_drawdown
                )
            
            if verbose and (gen % 10 == 0 or gen == self.generations - 1):
                print(
                    f" Gen {gen+1:>3}/{self.generations} | "
                    f"Best sharpe: {gen_best.fitness:>6.3f} | "
                    f"Mean Sharpe: {gen_mean:>6.3f} | "
                    f"Trades: {gen_best.trades:>3} | "
                )

        benchmark = buy_and_hold_benchmark(prices)

        if verbose:
            print(f"\n{'-'*60}")
            print(f"RESULTS - {asset_name}")
            print(f"{'-'*60}")
            print(f"Best strategy: {best_overall.describe()}")
            print(f"Sharpe Ratio: {best_overall.fitness:.4f} (B&H): {benchmark['sharpe_ratio']:.4f}")
            print(f"Total Return: {best_overall.total_return*100:.2f}% (B&H): {benchmark['total_return']*100:.2f}%")
            print(f"Win Rate: {best_overall.win_rate*100:.1f}%")
            print(f"Trades: {best_overall.trades}")
            print(f"Max drawdown: {best_overall.max_drawdown*100:.2f}%")

        return GAResult(
            asset=asset_name,
            best_chromosome=best_overall,
            best_fitness_history=best_fitness_history,
            mean_fitness_history=mean_fitness_history,
            benchmark_sharpe=benchmark['sharpe_ratio'],
            benchmark_return=benchmark['total_return'],
            generations=self.generations,
            population_size=self.population_size
        ) 
    
    #orchestrator 

class GATradingOptimiser:

    def __init__(
            self, 
            population_size: int = 50,
            generations: int = 100,
            random_state: int = 42
    ):
        self.ga_kwargs = dict(
            population_size=population_size,
            generations=generations,
            random_state=random_state
        )

    def _get_best_model_probabilities(
        self, 
        asset_results: dict,
        X_test:pd.DataFrame
    ) -> Tuple[str, np.ndarray]:
        
        results = asset_results["results"]
        models = asset_results["models"]

        lr_auc = results.get("Logistic Regression", {}).get("test_auc", 0)
        xgb_auc = results.get("XGBoost", {}).get("test_auc", 0)

        if xgb_auc > lr_auc:
            best_name = "XGBoost"
        else:
            best_name = "Logistic Regression"

        best_model = models[best_name]

        feature_names = asset_results["feature_names"]
        X_test = X_test[feature_names]
        proba = best_model.predict_proba(X_test)[:, 1]
        return best_name, proba
    
    def optimise_asset(
        self, 
        asset_name: str,
        asset_results: dict,
        prices_test: np.ndarray,
        X_test: pd.DataFrame,
        verbose: bool = True
    ) -> GAResult:
        model_name, probabilities = self._get_best_model_probabilities(asset_results, X_test)
        if verbose:
            print(f"\nUsing {model_name} probabilities for {asset_name} GA")
        
        ga = TradingStrategyGA(**self.ga_kwargs)
        result = ga.evolve(
            prices=prices_test,
            probabilities=probabilities,
            asset_name=asset_name,
            verbose=verbose
        )
        return result
    
    def optimise_all(
        self, 
        all_asset_results: dict,
        preprocessed_data: dict,
        verbose: bool = True
    ) -> Dict[str, GAResult]:
        ga_results = {}

        print("\n" + "="*80)
        print("Genetic Algorithm - Strategy Optimization")
        print("="*80 + "\n")
        print(f"Population size: {self.ga_kwargs['population_size']}")
        print(f"Generations: {self.ga_kwargs['generations']}")
        print("="*80)

        for asset_name, asset_results in all_asset_results.items():
            try:
                data = preprocessed_data[asset_name]
                X_test = data["X_test"]
                test_dates = data["test_dates"]

                raw_df = pd.read_csv(f"data/raw/{asset_name}.csv")
                raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce')
                raw_df = raw_df[['Date', 'Close']].dropna()
                raw_df = raw_df.sort_values('Date')

                test_date_set = set(test_dates.values)
                mask = raw_df['Date'].isin(test_date_set)
                prices_test = raw_df.loc[mask, 'Close'].values.astype(float)

                if len(prices_test) < 30:
                    print(f" Skipping {asset_name} - not enough test data ({len(prices_test)})")
                    continue

                result = self.optimise_asset(
                    asset_name=asset_name,
                    asset_results=asset_results,
                    prices_test=prices_test,
                    X_test=X_test,
                    verbose=verbose
                )

                ga_results[asset_name] = result

                model_name, probabilities = self._get_best_model_probabilities(asset_results, X_test)
                signals = (probabilities >= result.best_chromosome.entry_threshold).astype(int)

                self.plot_equity_curve(
                    prices_test,
                    signals,
                    asset_name,
                    f"results/figures/{asset_name}"
                )

            except Exception as e:
                print(f" Error optimizing {asset_name}: {e}")
                import traceback; traceback.print_exc()
                continue
        return ga_results
    
    #Plotting

    def plot_convergence(
        self,
        ga_result: GAResult,
        save_dir: str,
    ):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))

        gens = range(1, ga_result.generations + 1)
        ax.plot(gens, ga_result.best_fitness_history, color='blue', 
                linewidth=2, label='Best Sharpe')
        ax.plot(gens, ga_result.mean_fitness_history, color='orange', 
                linewidth=2, label='Mean Sharpe')
        ax.axhline(y=ga_result.benchmark_sharpe, color='green', linewidth=1.5, linestyle='--', label=f'Buy & Hold ({ga_result.benchmark_sharpe:.3f})')
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.set_title(f'GA Convergence - {ga_result.asset}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        best_gen = np.argmax(ga_result.best_fitness_history) + 1
        best_val = max(ga_result.best_fitness_history)
        ax.annotate(
            f'Best: {best_val:.3f}\nGen {best_gen}', 
                    xy=(best_gen, best_val), 
                    xytext=(best_gen + ga_result.generations * 0.5, best_val),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='red'),
                    color='red'
        )

        plt.tight_layout()
        path = f"{save_dir}/ga_convergence.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    def plot_strategy_comparison(
            self,
            ga_results: Dict[str, GAResult],
            save_dir: str = "results/figures"
    ):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        assets = list(ga_results.keys())
        ga_sharpes = [ga_results[a].best_chromosome.fitness for a in assets]
        bh_sharpes = [ga_results[a].benchmark_sharpe for a in assets]

        x = np.arange(len(assets))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width/2, ga_sharpes, width, label='GA Strategy', color='blue')
        bars2 = ax.bar(x + width/2, bh_sharpes, width, label='Buy & Hold', color='orange')

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Asset', fontsize=12)
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.set_title('GA Strategy vs Buy and Hold - Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(assets)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = f"{save_dir}/ga_vs_buyhold.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    def plot_all(
        self, 
        ga_results: Dict[str, GAResult],
        save_dir: str = "results/figures"
    ):
        for asset_name, result in ga_results.items():
            asset_dir = f"{save_dir}/{asset_name}"
            self.plot_convergence(result, asset_dir)
        self.plot_strategy_comparison(ga_results, save_dir)
    
    def plot_equity_curve(self, prices, signals, asset_name, save_dir):
        initial_capital = 10000

        #buy and hold equity
        buyhold_returns = prices[1:] / prices[:-1]
        buyhold_equity = initial_capital * np.cumprod(buyhold_returns)

        #strategy equity
        strategy_returns = []
        capital = initial_capital

        for i in range(1, len(prices)):
            if signals[i-1] == 1:
                capital *= prices[i] / prices[i-1]
            strategy_returns.append(capital)
        strategy_returns = np.array(strategy_returns)

        #plot
        plt.figure(figsize=(10,6))
        plt.plot(strategy_returns, label="GA strategy",  linewidth=2)
        plt.plot(buyhold_equity, label="Buy & Hold",  linewidth=2)

        plt.title(f"{asset_name} Equity Curve: GA vs Buy & Hold")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(alpha=0.3)

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = f"{save_dir}/equity_curve.png"

        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

#summary 

    def print_summary(self, ga_results: Dict[str, GAResult]):
        print(f"\n{'='*90}")
        print("GA OPTIMIZATION SUMMARY")
        print("="*90)
        header = (
            f"{'Asset':<8} | {'Best Sharpe':<12} | {'B&H Sharpe':<10} | "
            f"{'Return %':>8} | {'B&H Return %':>8} | {'Win Rate %':>9} | "
            f"{'Trades':>7} | {'Max DD %':>8} | {'Outperform':>13}"
        )
        print(header)
        print("-"*90)
        for asset_name, result in ga_results.items():
            c = result.best_chromosome
            outperform = "Yes" if c.fitness > result.benchmark_sharpe else "No"
            print(
               f"{asset_name:<8}"
               f"{c.fitness:>12.4f}"
               f"{result.benchmark_sharpe:>10.4f}"
                f"{c.total_return*100:>8.2f}"
                f"{result.benchmark_return*100:>8.2f}"
                f"{c.win_rate*100:>8.1f}"
                f"{c.trades:>7}"
                f"{c.max_drawdown*100:>7.2f}"
                f"{outperform:>13}"
           )
        print("="*90)

    def save_results(self, ga_results: Dict[str, GAResult], path: str = "results/ga_results.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(ga_results, f)
        print(f"Saved GA results to {path}")

#entry point 

if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root/ 'src'))
    from preprocessing import FinancialPreprocessor

    #load trained ml models and results
    results_path = "results/all_assets_results_3d.pkl"
    if not Path(results_path).exists():
        print(f"Error: {results_path} not found. Run the training script first.")
        sys.exit(1)
    
    with open(results_path, "rb") as f:
        all_asset_results = pickle.load(f)

    print(f"Loaded results for {len(all_asset_results)} assets")

    #re-preprocess each asset to get X_test an test dates
    assets = {
        "AAPL": "data/raw/AAPL.csv",
        "MSFT": "data/raw/MSFT.csv",
        "GOOGL": "data/raw/GOOGL.csv",
        "AMZN": "data/raw/AMZN.csv", 
        "TSLA": "data/raw/TSLA.csv", 
        "NVDA": "data/raw/NVDA.csv",
        "META": "data/raw/META.csv",
        "BTC": "data/raw/BTC.csv",
        "ETH": "data/raw/ETH.csv"
    }

    preprocessed_data = {}
    for asset_name, filepath in assets.items():
        if not Path(filepath).exists():
            print(f"Warning: {filepath} not found. Skipping {asset_name}.")
            continue
        preprocessor  = FinancialPreprocessor()
        preprocessed_data[asset_name] = preprocessor.process_asset(filepath)
        print(f" Preprocessed: {asset_name}")


    #run GA optimization
    print("\nStarting GA optimization...")
    optimiser = GATradingOptimiser(
        population_size=50,
        generations=100,
        random_state=42
    )

    ga_results = optimiser.optimise_all(
        all_asset_results=all_asset_results,
        preprocessed_data=preprocessed_data,
        verbose=True
    )

    #save and plot results
    optimiser.save_results(ga_results)
    optimiser.plot_all(ga_results, save_dir="results/figures")
    optimiser.print_summary(ga_results)

    print("\n" + "="*80)
    print("\nGA optimization complete")
    print("="*80)
    print(f"Results saved to: results/ga_results.pkl")
    print(f"Figures saved to: results/figures/")
    print(f"  - Per-asset convergence plots: results/figures/[asset]/ga_convergence.png")
    print(f" - Cross-asset comparison: results/figures/ga_vs_buyhold.png")

