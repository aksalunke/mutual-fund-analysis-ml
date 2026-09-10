# Architecture Decisions

Design decisions made while building the clustering and regression models.

---

### ADR-1: K=3 clusters

**Decision:** K-Means was run with K=3 on Beta and Standard Deviation.

**Reasoning:** The elbow-method curve flattens noticeably after K=4, but
plotting K=4 concentrated the data into what visually read as 3 groups,
with the fourth cluster difficult to interpret against fund type and
category. K=3 produced clusters that mapped cleanly onto volatility bands
with clear fund-type composition — the more useful result for the
analysis this project set out to do.

---

### ADR-2: Outliers retained; RobustScaler used

**Decision:** Statistical outliers in the return and ratio columns were
identified but not removed. RobustScaler (median/IQR-based) was used
ahead of K-Means rather than StandardScaler.

**Reasoning:** The outliers are legitimate financial data — a fund with an
unusually large gain or loss is a real data point, not noise — so removing
them would misrepresent the fund universe. RobustScaler was chosen because
it's less sensitive to exactly the outliers being deliberately kept in, a
directly connected pair of decisions rather than two independent defaults.

---

### ADR-3: Single-ratio regression, one per return horizon

**Decision:** For each return horizon (1/2/3 years), Random Forest
feature-importance identified the single strongest financial ratio, which
was then used as the sole predictor in a linear regression.

**Reasoning:** A single, clearly-identified predictor keeps the result
interpretable and explainable — "returns are best explained by Alpha" is
a clean, presentable finding for a coursework scope. See "Possible future
enhancements" in the README for what a multivariate version would add.

---

### ADR-4: Source dataset excluded from the repository

**Decision:** The raw data file isn't committed to this repository.

**Reasoning:** The dataset is third-party data — publishing the source file
isn't this project's call to make. Documenting the methodology and
publishing the code that operates on it is a separate question with a
separate answer.
