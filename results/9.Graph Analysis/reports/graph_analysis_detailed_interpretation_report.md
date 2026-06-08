# Detailed Graph Analysis Interpretation Report

## Why Graph Analysis Was Performed

Graph analysis was added to move beyond individual-level prediction and ask whether dropout risk forms connected communities. This is useful for universities because interventions can be designed for groups of similar students or similar courses rather than only for isolated individuals.

Two graph views were created. The student-risk similarity graph connects students with similar academic progression, financial status, demographic profile, course information, and predicted dropout risk. The course-risk graph connects courses with similar dropout profiles, predicted risk, academic progression, and financial vulnerability.

## Student-Risk Graph Findings

The student graph contains 4424 students and 36303 weighted similarity links. Community detection found 10 communities with modularity 0.733. This modularity is high enough to suggest meaningful risk communities rather than random grouping.

- **Community 2: Collapsed academic progression community**. n=911; dropout=82.4%; predicted risk=81.2%; second-semester approved=0.08; tuition-not-up-to-date=29.5%; scholarship=10.5%. This is the dominant high-risk graph community. It is defined by collapsed academic progression and very high predicted dropout probability.
- **Community 0: Mixed-risk community**. n=1280; dropout=29.8%; predicted risk=29.8%; second-semester approved=6.32; tuition-not-up-to-date=17.4%; scholarship=9.5%. This community has moderate observed dropout and should be monitored, especially for academic and payment-status changes.
- **Community 8: Mixed-risk community**. n=37; dropout=29.7%; predicted risk=20.9%; second-semester approved=3.65; tuition-not-up-to-date=2.7%; scholarship=0.0%. This community has moderate observed dropout and should be monitored, especially for academic and payment-status changes.
- **Community 1: Mixed-risk community**. n=1103; dropout=18.7%; predicted risk=16.6%; second-semester approved=4.56; tuition-not-up-to-date=1.9%; scholarship=2.8%. This community is lower-risk but still contains some enrolled or uncertain students.
- **Community 5: Mixed-risk community**. n=82; dropout=9.8%; predicted risk=12.8%; second-semester approved=3.63; tuition-not-up-to-date=0.0%; scholarship=0.0%. This is a low-risk or stable community with strong academic/financial indicators.
- **Community 7: Stable low-risk community**. n=43; dropout=9.3%; predicted risk=12.0%; second-semester approved=4.84; tuition-not-up-to-date=2.3%; scholarship=0.0%. This is a low-risk or stable community with strong academic/financial indicators.
- **Community 4: Stable low-risk community**. n=132; dropout=7.6%; predicted risk=5.5%; second-semester approved=7.28; tuition-not-up-to-date=0.0%; scholarship=68.2%. This is a low-risk or stable community with strong academic/financial indicators.
- **Community 6: Stable low-risk community**. n=75; dropout=6.7%; predicted risk=7.4%; second-semester approved=5.51; tuition-not-up-to-date=1.3%; scholarship=0.0%. This is a low-risk or stable community with strong academic/financial indicators.
- **Community 3: Stable low-risk community**. n=734; dropout=6.0%; predicted risk=6.5%; second-semester approved=5.83; tuition-not-up-to-date=1.6%; scholarship=100.0%. This is a low-risk or stable community with strong academic/financial indicators.
- **Community 9: Stable low-risk community**. n=27; dropout=3.7%; predicted risk=3.1%; second-semester approved=5.11; tuition-not-up-to-date=0.0%; scholarship=100.0%. This is a low-risk or stable community with strong academic/financial indicators.

The most important student-graph result is the identification of a large collapsed academic progression community: 911 students, 82.4% observed dropout, and 81.2% mean predicted dropout probability. This aligns closely with the SHAP and clustering findings, where approved units and semester grades dominated dropout explanation.

## Course-Risk Graph Findings

The strict course graph used a high similarity threshold and produced 3 edges. This conservative graph identifies only very close course-risk similarities. Because that view is sparse, a relaxed top-3-neighbor graph was also created with 36 edges and 2 course communities.

Highest dropout courses:
- Course 33: n=12; dropout=66.7%; predicted risk=60.0%; second-semester approved=4.00; tuition-not-up-to-date=25.0%.
- Course 9130: n=141; dropout=55.3%; predicted risk=50.8%; second-semester approved=3.11; tuition-not-up-to-date=19.1%.
- Course 9119: n=170; dropout=54.1%; predicted risk=49.1%; second-semester approved=2.16; tuition-not-up-to-date=16.5%.
- Course 9991: n=268; dropout=50.7%; predicted risk=50.3%; second-semester approved=3.72; tuition-not-up-to-date=17.5%.
- Course 9853: n=192; dropout=44.3%; predicted risk=39.2%; second-semester approved=4.61; tuition-not-up-to-date=12.0%.
- Course 9003: n=210; dropout=41.0%; predicted risk=40.5%; second-semester approved=4.75; tuition-not-up-to-date=21.9%.

Relaxed course-risk communities:
- **Course community 0**: courses 33; 171; 8014; 9003; 9119; 9130; 9147; 9853; 9991; total students=1803; mean dropout=46.5%; mean predicted risk=44.6%; mean second-semester approved=3.59; mean tuition-not-up-to-date=16.9%.
- **Course community 1**: courses 9070; 9085; 9238; 9254; 9500; 9556; 9670; 9773; total students=2621; mean dropout=28.2%; mean predicted risk=27.2%; mean second-semester approved=4.71; mean tuition-not-up-to-date=10.5%.

## Overall Interpretation

The student graph confirms that dropout risk is not scattered randomly: similar students form clear communities. The highest-risk community is dominated by academic non-progression, while low-risk communities are characterized by stronger grades, more approved units, lower financial vulnerability, and in some communities high scholarship rates.

The course graph suggests that course-level dropout risk also has structure. A high-risk course community has substantially higher mean dropout than the lower-risk course community, and it is associated with weaker academic progression and higher financial vulnerability. This does not prove course causality, but it gives universities a useful program-level monitoring layer.

## Recommended Paper Use

Use the student graph as the main graph contribution because it is stronger and more directly linked to intervention design. Use the course graph as a secondary program-level analysis. Recommended figures: `student_similarity_graph_pca_risk.png`, `student_community_dropout_rates.png`, and `course_risk_similarity_graph_relaxed_top3.png`. Recommended tables: `student_graph_community_profiles.csv` and `course_risk_community_profiles_relaxed_top3.csv`.