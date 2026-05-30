import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for dist in ["UN", "CL"]:
        CSV_DIR = os.path.join(BASE_DIR, "outputs", "csv", f"synthetic_{dist.lower()}")
        PLOT_DIR = os.path.join(BASE_DIR, "outputs", "plots", f"synthetic_{dist.lower()}")
        os.makedirs(PLOT_DIR, exist_ok=True)
        print(f"Plotting for distribution {dist}")
        
        # 1. Runtime Comparison
        rc_file = os.path.join(CSV_DIR, "runtime_compare.csv")
        if os.path.exists(rc_file):
            df = pd.read_csv(rc_file)
            times = df.groupby('Method')['Runtime'].mean()
            plt.figure(figsize=(8,5))
            times.plot(kind='bar', color=['red', 'salmon', 'blue', 'lightblue'])
            plt.title(f'Average Runtime by Method ({dist})')
            plt.ylabel('Runtime (s)')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "runtime_comparison.png"))
            plt.close()
            
        # 2. Scalability
        sc_file = os.path.join(CSV_DIR, "scalability_results.csv")
        if os.path.exists(sc_file):
            df = pd.read_csv(sc_file)
            
            du = df[df['Parameter'] == 'users']
            plt.figure(figsize=(8,5))
            for m in du['Method'].unique():
                sub = du[du['Method'] == m]
                plt.plot(sub['Value'], sub['Runtime'], marker='o', label=m)
            plt.title(f'Runtime vs Number of Users ({dist})')
            plt.xlabel('Users')
            plt.ylabel('Runtime (s)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "runtime_vs_users.png"))
            plt.close()
            
            dl = df[df['Parameter'] == 'timestamps']
            plt.figure(figsize=(8,5))
            for m in dl['Method'].unique():
                sub = dl[dl['Method'] == m]
                plt.plot(sub['Value'], sub['Runtime'], marker='o', label=m)
            plt.title(f'Runtime vs Timestamps L ({dist})')
            plt.xlabel('L (Timestamps)')
            plt.ylabel('Runtime (s)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "runtime_vs_timestamps.png"))
            plt.close()
            
            plt.figure(figsize=(8,5))
            sub_hyb = du[du['Method'] == 'Hybrid_SSA']
            plt.plot(sub_hyb['Value'], sub_hyb['PruningRatio'], marker='o', color='green')
            plt.title(f'Pruning Ratio vs Users ({dist})')
            plt.xlabel('Users')
            plt.ylabel('Pruning Ratio')
            plt.ylim(0, 1.1)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "pruning_vs_users.png"))
            plt.close()

        # 3. Accuracy Sweep
        sw_file = os.path.join(CSV_DIR, "parameter_sweep.csv")
        if os.path.exists(sw_file):
            df = pd.read_csv(sw_file)
            df_hyb = df[df['Method'] == 'Hybrid_SSA']
            plt.figure(figsize=(8,5))
            for tau in df_hyb['tau'].unique():
                sub = df_hyb[df_hyb['tau'] == tau]
                agg = sub.groupby('c')['Recall'].mean()
                plt.plot(agg.index, agg.values, marker='o', label=f'tau={tau}')
            plt.title(f'Recall vs Approximation parameter c ({dist})')
            plt.xlabel('c (k relaxing factor)')
            plt.ylabel('Recall')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "recall_vs_c.png"))
            plt.close()

if __name__ == "__main__":
    main()
