# EDA One-Page Report: Findings to Carry Into the Paper

The cleaned dataset contains 4,424 students, 37 original variables, no missing values, and no duplicate rows. The outcome is moderately imbalanced: 2,209 students graduated (49.9%), 1,421 dropped out (32.1%), and 794 remained enrolled (17.9%). This class structure justifies using macro-F1, balanced accuracy, and dropout-specific metrics rather than relying on accuracy alone.

The strongest descriptive signal is financial vulnerability. Students whose tuition fees were not up to date had an 86.6% dropout rate, compared with 24.7% among students with paid/up-to-date tuition. Debtor students had a 62.0% dropout rate, compared with 28.3% for non-debtors. Scholarship status moves in the opposite direction: students without scholarships had a 38.7% dropout rate, while scholarship holders had only 12.2%. These patterns strongly support a policy-oriented interpretation: dropout risk is not only academic but also financial, and universities may benefit from integrating early financial-support triggers into warning systems.

Academic progression is the clearest early-warning signal. Students in the lowest quintile of first-semester approved units had a 77.9% dropout rate, while the highest quintile was about 10.5%. For first-semester grade, the lowest quintile had a 75.0% dropout rate, compared with 12.1% in the highest quintile. The second-semester variables show an even sharper separation: the lowest quintile of second-semester approved units had an 83.3% dropout rate, and the lowest quintile of second-semester grade had an 82.9% dropout rate. This directly supports the staged early-warning design: pre-enrollment models can identify background risk, but first-semester academic data should substantially improve detection, and second-semester data will likely sharpen it further.

Age also shows a meaningful risk gradient. Students in the youngest quintile had an 18.6% dropout rate, while students in the oldest quintile, ages 27-70, had a 56.3% dropout rate. This suggests that older students may face different constraints, such as work, family, schedule, or financial pressure. This should be discussed as an intervention-relevant group rather than only as a model feature.

Course and application route show institutional heterogeneity. Several courses have dropout rates above 50% among groups with at least 30 students, including course codes 9130 (55.3%), 9119 (54.1%), and 9991 (50.7%). Application mode also differs strongly: mode 7 has 61.2% dropout and mode 39 has 55.4%, while the largest group, mode 1, has 20.2%. These variables are promising for course-risk profiling and course-level intervention design. They also justify the proposed course-risk graph.

The academic correlation table confirms that semester performance variables are highly correlated across semesters: first- and second-semester credited units correlate at 0.945, enrolled units at 0.943, approved units at 0.904, grades at 0.837, and evaluations at 0.779. This is important for the paper because it explains why PCA was not ideal for interpretation and why staged modeling is better: instead of hiding these variables inside components, we can compare what is known at each time point.

Recommended paper visuals/tables:

1. Target distribution plot/table: establishes the multiclass and imbalance problem.
2. Financial dropout-rate table or figure: tuition, debtor, and scholarship status should be included for sure.
3. Academic quintile figure: first-semester grade or approved units, plus second-semester grade/approved units if space allows.
4. Age quintile figure: supports the older-student risk profile.
5. Course-level dropout-rate figure: supports university-level/course-level decision making.
6. Academic correlation heatmap or high-correlation table: use briefly to justify avoiding PCA as the main interpretability method and motivate staged early-warning models.

Most important manuscript message: dropout risk is concentrated around three practical dimensions: academic progression, financial vulnerability, and institutional/course pathway. These EDA results motivate the next modeling stages: staged early-warning prediction, variable-block ablation, dropout-specific SHAP, and risk clustering.
