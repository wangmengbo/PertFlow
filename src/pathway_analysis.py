import os
import argparse
import pandas as pd
import numpy as np
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns


def parse_conditions(columns):
    """Extract condition names by removing the '_sampleX' suffix."""
    conditions = []
    for col in columns:
        parts = col.split('_')
        condition = '_'.join(parts[:-1])
        conditions.append((col, condition))
    return pd.DataFrame(conditions, columns=['original_col', 'condition'])


def run_gsea_for_condition(gene_scores, condition_name, outdir, gene_sets):
    """Run PyGSEA prerank for a specific condition."""
    rnk = gene_scores.reset_index()
    rnk.columns = ['Gene', 'Score']
    rnk['Gene'] = rnk['Gene'].str.upper()
    rnk = rnk.sort_values(by='Score', ascending=False).reset_index(drop=True)
    
    print(f"Running GSEA for {condition_name}...")
    try:
        res = gp.prerank(
            rnk=rnk,
            gene_sets=gene_sets,
            threads=4,
            min_size=5,
            max_size=1000,
            permutation_num=1000,
            outdir=os.path.join(outdir, f"gsea_{condition_name}"),
            seed=42,
            verbose=False
        )
        df_res = res.res2d
        df_res['Condition'] = condition_name
        return df_res
    except Exception as e:
        print(f"  GSEA failed: {e}")
        return pd.DataFrame()


def run_enrichr_for_condition(gene_scores, condition_name, outdir, gene_sets, top_k=200):
    """Run Enrichr on the top K genes with the highest gradient scores."""
    print(f"Running Enrichr for {condition_name}...")
    
    # Sort descending and take top K genes
    top_genes = gene_scores.sort_values(ascending=False).head(top_k).index.tolist()
    top_genes = [str(g).upper() for g in top_genes]
    
    enrichr_dir = os.path.join(outdir, f"enrichr_{condition_name}")
    os.makedirs(enrichr_dir, exist_ok=True)
    
    try:
        enr = gp.enrichr(
            gene_list=top_genes,
            gene_sets=gene_sets,
            outdir=enrichr_dir,
            cutoff=0.25
        )
        df_res = enr.results
        df_res['Condition'] = condition_name
        return df_res
    except Exception as e:
        print(f"  Enrichr failed: {e}")
        return pd.DataFrame()


def generate_comparative_dotplot(all_gsea_results, output_path, top_n=5):
    """Generate a dotplot comparing pathway enrichment across conditions."""
    if not all_gsea_results:
        print("No significant GSEA results to plot.")
        return

    combined_df = pd.concat(all_gsea_results, ignore_index=True)
    combined_df['NES'] = pd.to_numeric(combined_df['NES'], errors='coerce')
    combined_df['FDR q-val'] = pd.to_numeric(combined_df['FDR q-val'], errors='coerce')
    combined_df = combined_df.dropna(subset=['NES', 'FDR q-val'])
    
    # Filter for positive NES and significance
    combined_df = combined_df[(combined_df['FDR q-val'] < 0.05) & (combined_df['NES'] > 0)]
    
    if combined_df.empty:
        print("No positive significant pathways met the FDR threshold for plotting.")
        return

    top_pathways = set()
    for condition in combined_df['Condition'].unique():
        cond_df = combined_df[combined_df['Condition'] == condition]
        top_pathways.update(cond_df.nlargest(top_n, 'NES')['Term'].tolist())
    
    plot_df = combined_df[combined_df['Term'].isin(top_pathways)].copy()
    plot_df['-log10(FDR)'] = np.round(-np.log10(plot_df['FDR q-val'] + 1e-10), 2)
    plot_df['Term'] = plot_df['Term'].apply(lambda x: x.split(' (GO:')[0] if 'GO:' in x else x)

    plt.figure(figsize=(max(8, len(plot_df['Condition'].unique()) * 1.5), max(6, len(top_pathways) * 0.4)))
    sns.set_theme(style="whitegrid")
    
    scatter = sns.scatterplot(
        data=plot_df,
        x='Condition',
        y='Term',
        size='-log10(FDR)',
        hue='NES',
        palette='Reds',
        sizes=(50, 500),
        edgecolor='gray',
        linewidth=0.5
    )
    
    plt.title('Enriched Pathways by Drug & Cell Line', fontsize=14, pad=20)
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('Pathway', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description="Run Pathway Analysis on gene importance scores.")
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gene_sets', type=str, default='KEGG_2021_Human')
    parser.add_argument('--top_n', type=int, default=5)
    parser.add_argument('--top_k_enrichr', type=int, default=200, help="Top K genes to use for Enrichr ORA")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading gene importance data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv, index_col=0)
    
    mapping_df = parse_conditions(df.columns)
    
    grouped_scores_dict = {}
    for condition in mapping_df['condition'].unique():
        replicate_cols = mapping_df[mapping_df['condition'] == condition]['original_col'].tolist()
        grouped_scores_dict[condition] = df[replicate_cols].mean(axis=1)
        
    grouped_scores = pd.DataFrame(grouped_scores_dict, index=df.index)

    all_gsea_results = []
    for condition in grouped_scores.columns:
        # 1. Run GSEA
        res_df = run_gsea_for_condition(
            gene_scores=grouped_scores[condition], 
            condition_name=condition, 
            outdir=args.output_dir,
            gene_sets=args.gene_sets
        )
        if not res_df.empty:
            all_gsea_results.append(res_df)
            
        # 2. Run Enrichr
        run_enrichr_for_condition(
            gene_scores=grouped_scores[condition],
            condition_name=condition,
            outdir=args.output_dir,
            gene_sets=args.gene_sets,
            top_k=args.top_k_enrichr
        )

    # Generate summary dotplot for GSEA
    plot_path = os.path.join(args.output_dir, f'pathway_dotplot_{args.gene_sets}.png')
    generate_comparative_dotplot(all_gsea_results, plot_path, top_n=args.top_n)


if __name__ == "__main__":
    main()

