# Detailed Risk Clustering Interpretation Report

## Why Risk Clustering Was Performed

The purpose of this analysis is different from supervised model evaluation. The predictive models estimate dropout risk, while risk clustering translates those predictions and key student characteristics into practical intervention groups. Therefore, clustering was performed on the whole dataset, not only on the training/test split. This is appropriate because the goal is descriptive segmentation and policy design, not estimating out-of-sample classifier accuracy.

The clustering matrix combined predicted dropout probability from the tuned full-feature model with interpretable risk variables: academic progression, semester grades, tuition status, debtor status, scholarship status, age, displacement, course, application mode, and prior/admission grades.

## Why These Cluster Solutions Were Used

The statistical cluster-selection metrics favored **k = 3**, with the highest silhouette score (0.331). This k=3 solution gives the cleanest separation and is useful for reporting that the data naturally separates into broad risk groups.

However, a university early-warning or retention system often needs more detailed intervention categories. For this reason, a practical **k = 5** sensitivity solution was also created. The k=5 solution is more useful for policy because it separates severe academic collapse, financial-academic emergency, moderate academic delay, older high-workload students, and low-risk students.

## Statistical k=3 Findings

- **Cluster 0: Severe academic non-progression risk**. This group contains 898 students, with observed dropout rate 83.6% and mean predicted dropout probability 82.4%. Mean second-semester approved units are 0.08, mean second-semester grade is 0.59, tuition-not-up-to-date rate is 30.6%, and debtor rate is 18.9%. Recommended action: Mandatory academic advising and course-completion intervention.
- **Cluster 2: Older high-workload moderate risk**. This group contains 262 students, with observed dropout rate 22.9% and mean predicted dropout probability 24.3%. Mean second-semester approved units are 10.18, mean second-semester grade is 12.67, tuition-not-up-to-date rate is 9.5%, and debtor rate is 14.5%. Recommended action: Offer flexible scheduling and light-touch persistence monitoring.
- **Cluster 1: Low-risk academically stable group**. This group contains 3264 students, with observed dropout rate 18.7% and mean predicted dropout probability 17.9%. Mean second-semester approved units are 5.17, mean second-semester grade is 12.69, tuition-not-up-to-date rate is 7.0%, and debtor rate is 9.0%. Recommended action: Maintain light monitoring; use as reference profile.

The k=3 solution shows one very clear high-risk academic non-progression group, one large lower-risk academically stable group, and one smaller moderate-risk/high-workload group. This confirms that academic progression is the dominant segmentation axis, with financial indicators adding additional intervention context.

## Practical k=5 Cluster Findings

### Cluster 1: Severe academic non-progression risk
- Size: 720 students
- Observed dropout rate: 81.5%
- Mean predicted dropout probability: 81.4%
- Mean age: 25.7
- Mean second-semester approved units: 0.03
- Mean second-semester grade: 0.30
- Tuition-not-up-to-date rate: 29.3%
- Debtor rate: 17.8%
- Scholarship rate: 9.9%
- Interpretation: This is the clearest dropout-risk group. Students have almost no second-semester progress and extremely low semester grades, so the main problem is academic non-progression rather than only financial status.
- Recommended action: Mandatory academic advising and course-completion intervention

### Cluster 4: Financial-academic emergency risk
- Size: 66 students
- Observed dropout rate: 63.6%
- Mean predicted dropout probability: 63.8%
- Mean age: 27.4
- Mean second-semester approved units: 2.29
- Mean second-semester grade: 4.98
- Tuition-not-up-to-date rate: 34.8%
- Debtor rate: 22.7%
- Scholarship rate: 13.6%
- Interpretation: This small group combines academic weakness with the highest unpaid-tuition rate. Because the group is small but very high risk, it is suitable for direct case-management intervention.
- Recommended action: Combine financial outreach with urgent academic completion support

### Cluster 2: Academic delay with financial vulnerability
- Size: 1129 students
- Observed dropout rate: 52.3%
- Mean predicted dropout probability: 46.1%
- Mean age: 26.9
- Mean second-semester approved units: 3.17
- Mean second-semester grade: 10.61
- Tuition-not-up-to-date rate: 20.7%
- Debtor rate: 19.7%
- Scholarship rate: 9.8%
- Interpretation: This is a large medium-to-high-risk group. Students are not completely academically inactive, but they have delayed progression and meaningful financial pressure, making them important for scalable advising and payment-status monitoring.
- Recommended action: Prioritize course-completion advising and payment-status monitoring

### Cluster 3: Older high-workload moderate risk
- Size: 249 students
- Observed dropout rate: 20.5%
- Mean predicted dropout probability: 22.1%
- Mean age: 28.5
- Mean second-semester approved units: 10.35
- Mean second-semester grade: 12.70
- Tuition-not-up-to-date rate: 8.4%
- Debtor rate: 13.7%
- Scholarship rate: 12.9%
- Interpretation: This group has many enrolled and approved units and relatively good grades, but students are older and still show some risk. The issue is more likely persistence burden, scheduling, workload, or life constraints rather than academic inability.
- Recommended action: Offer flexible scheduling and light-touch persistence monitoring

### Cluster 0: Low-risk academically stable group
- Size: 2260 students
- Observed dropout rate: 6.7%
- Mean predicted dropout probability: 8.2%
- Mean age: 20.0
- Mean second-semester approved units: 5.88
- Mean second-semester grade: 13.08
- Tuition-not-up-to-date rate: 1.7%
- Debtor rate: 4.6%
- Scholarship rate: 38.8%
- Interpretation: This is the reference low-risk group. Students have strong academic progression, very low unpaid-tuition and debtor rates, and the highest scholarship rate.
- Recommended action: Maintain light monitoring; use as reference profile

## Overall Findings

First, the clustering strongly confirms the supervised-model and SHAP findings: academic progression variables, especially approved curricular units and semester grades, define the strongest dropout-risk segmentation. Students with almost no approved units in the second semester form the highest-risk group.

Second, financial vulnerability is not the only determinant, but it sharpens intervention priorities. The financial-academic emergency cluster has a high dropout rate and the highest tuition-not-up-to-date rate, showing that unpaid tuition and debtor status help universities identify students who may need immediate financial support in addition to academic advising.

Third, not all non-low-risk students require the same intervention. Some students need urgent academic recovery, some need combined academic and financial support, and some older/high-workload students may need flexible scheduling or persistence support rather than remedial academic intervention.

Fourth, the low-risk group is large and clearly distinct: strong academic progress, low unpaid tuition, low debtor rate, and higher scholarship rate. This group can be monitored lightly, allowing universities to allocate more resources to severe and moderate-risk clusters.

## What To Put In The Paper

The paper should present the k=3 solution as the statistically strongest segmentation and the k=5 solution as a practical intervention-oriented sensitivity analysis. The most useful table is the k=5 `risk_cluster_profiles.csv`, because it directly maps each group to dropout rate, predicted risk, academic progression, financial vulnerability, and recommended action. The best figures are `cluster_dropout_rate.png` and `cluster_profile_heatmap.png` from the Practical_5_Clusters folder.