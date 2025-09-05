import os, pandas as pd, numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt, seaborn as sns, argparse

def compare_models(args):
    file1_path, file2_path = os.path.join(args.results_dir, f"{args.model1}_predictions.csv"), os.path.join(args.results_dir, f"{args.model2}_predictions.csv")
    if not os.path.exists(file1_path) or not os.path.exists(file2_path): return

    df1, df2 = pd.read_csv(file1_path), pd.read_csv(file2_path)
    correct1, correct2 = (df1['true_label'] == df1['predicted_label']), (df2['true_label'] == df2['predicted_label'])
    n11, n10 = np.sum(correct1 & correct2), np.sum(correct1 & ~correct2)
    n01, n00 = np.sum(~correct1 & correct2), np.sum(~correct1 & ~correct2)
    contingency_table = [[n11, n10], [n01, n00]]
    result = mcnemar(contingency_table, exact=False, correction=True)
    
    print(f"\n--- McNemar's Test: {args.model1} vs. {args.model2} ---")
    if result.pvalue < 0.05: print("Conclusion: REJECT null hypothesis (models are significantly different).")
    else: print("Conclusion: FAIL to reject null hypothesis.")
        
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap('cividis')
    labels = [[f"Both Correct\n{n11}", f"{args.model1} Correct\n{args.model2} Incorrect\n(b = {n10})"],
              [f"{args.model1} Incorrect\n{args.model2} Correct\n(c = {n01})", f"Both Incorrect\n{n00}"]]
    text_colors = [['black', 'yellow'], ['yellow', 'yellow']]
    sns.heatmap(contingency_table, annot=False, fmt='', cmap=cmap, cbar=False,
                xticklabels=[f'{args.model2} Correct', f'{args.model2} Incorrect'],
                yticklabels=[f'{args.model1} Correct', f'{args.model1} Incorrect'])
    for i, text in enumerate(np.array(labels).flatten()):
        plt.text(i % 2 + 0.5, i // 2 + 0.5, text, ha='center', va='center', color=np.array(text_colors).flatten()[i], fontsize=14, weight='bold')
    plt.title("McNemar's Test Contingency Table"); plt.xlabel(f"{args.model2} Prediction Outcome"); plt.ylabel(f"{args.model1} Prediction Outcome")
    plt.savefig(os.path.join(args.results_dir, f"mcnemar_test_{args.model1}_vs_{args.model2}_improved.png"), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare two models using McNemar's Test.")
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model2', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='outputs/plots')
    args = parser.parse_args()
    compare_models(args)