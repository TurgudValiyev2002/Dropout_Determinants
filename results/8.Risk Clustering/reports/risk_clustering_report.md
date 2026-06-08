# Risk Clustering Report

This analysis clusters students using the whole dataset, not the train/validation/test split. The goal is not model evaluation, but practical intervention design. The clustering matrix combines key academic, financial, demographic, and course-related risk variables with the tuned full-feature model's predicted dropout probability.

The strongest statistical clustering solution uses **k = 3**. Selection considered silhouette and Calinski-Harabasz scores. Because intervention design may need more granular groups, a separate practical **k = 5** sensitivity solution is also saved under `Practical_5_Clusters/`.

## Most Important Cluster Findings

The highest-risk cluster is **Cluster 0**, labelled **Severe academic non-progression risk**. It contains 898 students, has observed dropout rate 83.6%, mean predicted dropout probability 82.4%, tuition-not-up-to-date rate 30.6%, debtor rate 18.9%, and mean second-semester approved units 0.08. Recommended action: Mandatory academic advising and course-completion intervention.

The lowest-risk cluster is **Cluster 1**, labelled **Low-risk academically stable group**. It contains 3264 students and has observed dropout rate 18.7%, with mean second-semester approved units 5.17 and mean second-semester grade 12.69.

## Paper Use

For the paper, use the k=3 solution if the goal is clean statistical separation, and use the k=5 practical sensitivity solution if the goal is richer intervention groups. The strongest figures are `cluster_dropout_rate.png` and `cluster_profile_heatmap.png`. These results translate the predictive model into actionable intervention groups rather than only reporting accuracy.

## Practical k=5 Sensitivity Summary

- Cluster 1: Severe academic non-progression risk; n=720; dropout=81.5%; predicted risk=81.4%; second-semester approved=0.03; tuition-not-up-to-date=29.3%.
- Cluster 4: Financial-academic emergency risk; n=66; dropout=63.6%; predicted risk=63.8%; second-semester approved=2.29; tuition-not-up-to-date=34.8%.
- Cluster 2: Academic delay with financial vulnerability; n=1129; dropout=52.3%; predicted risk=46.1%; second-semester approved=3.17; tuition-not-up-to-date=20.7%.
- Cluster 3: Older high-workload moderate risk; n=249; dropout=20.5%; predicted risk=22.1%; second-semester approved=10.35; tuition-not-up-to-date=8.4%.
- Cluster 0: Low-risk academically stable group; n=2260; dropout=6.7%; predicted risk=8.2%; second-semester approved=5.88; tuition-not-up-to-date=1.7%.