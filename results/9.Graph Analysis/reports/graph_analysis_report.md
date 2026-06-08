# Graph Analysis Report

This analysis builds two graphs using the whole dataset. The student-risk graph connects students with similar academic, financial, demographic, course, and predicted-risk profiles. The course-risk graph connects courses that have similar dropout-risk profiles.

## Student-Risk Similarity Graph

The student graph contains 4424 nodes and 36303 weighted similarity edges. Greedy modularity community detection found 10 communities with modularity 0.733.

The highest-risk student community is **Community 2**, labelled **Collapsed academic progression community**. It contains 911 students, has dropout rate 82.4%, mean predicted dropout probability 81.2%, tuition-not-up-to-date rate 29.5%, and mean second-semester approved units 0.08.

## Course-Risk Similarity Graph

The highest-dropout course profile is Course 33, with 12 students, dropout rate 66.7%, and mean predicted dropout probability 60.0%.

The highest-risk course community contains courses 33, with mean dropout rate 66.7%. This suggests that course-level risk may cluster across programs with similar academic progression and financial-risk profiles rather than appearing only in isolated courses.

Because the strict 0.90 similarity graph is intentionally conservative, a relaxed top-3-neighbor course graph is also saved. Its highest-risk course community contains courses 33; 171; 8014; 9003; 9119; 9130; 9147; 9853; 9991, with mean dropout rate 46.5%. This relaxed view is more useful for visualization and for finding groups of courses with similar risk profiles.

## Paper Use

Recommended figures: `student_similarity_graph_pca_risk.png`, `student_community_dropout_rates.png`, `course_risk_similarity_graph_relaxed_top3_interpretable.png`, and `course_dropout_rates.png`. Recommended tables: `student_graph_community_profiles.csv`, `course_risk_profiles.csv`, `course_risk_community_profiles_relaxed_top3.csv`, and `course_risk_interpretable_graph_node_summary.csv`.