# ING Employer-Branding Diagnosis

My objective was to find out what employees say about **ING** compared with its main competitors (Santander, BNP-Paribas and Deutsche-Bank), using the pros and cons they write on Glassdoor. This way, HR knows what to push in employer branding to attract talent and what to fix to stop losing it.

For this purpose, I used the **Glassdoor Job Reviews 1 & 2 — David Gauthier** dataset, available on [Kaggle](https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews).

**TL;DR:** ING is perceived as a better employer than its peers (rating 4.00 vs. 3.59-3.71). The reviews point to two assets to promote (*Culture & people* and *Work-life balance*), one friction to fix (*Processes & technology*) and one risk to keep an eye on (*Management & leadership*). Nothing in the data justifies a pay or career overhaul. I also want to be upfront about something: only one gap in the whole analysis has a 95% interval that excludes zero, so these findings tell HR *where to look*, not what is proven.

## 1. Project Context

### 1.1. Why this project

I picked ING because I work there, so I had a real reason to care about the answer. The peers are large European banks with international presence that compete with ING for similar talent profiles, and all of them have enough reviews to support a fair comparison. One caveat: Deutsche-Bank has a stronger investment banking side, so part of its talent pool isn't identical to ING's.

### 1.2. Workflow

```text
Raw Data (Glassdoor reviews)
   │
   ▼
Balanced Sampling (stratified by year & current/former)
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Sentiment Analysis (DistilBERT) ── quality check of pros/cons
   │
   ▼
Topic Modeling (MiniLM embeddings + BERTopic)
   │
   ├── Pros model
   ├── Cons model
   │
   ▼
Outlier Reassignment & Grouping into Business Categories
   │
   ▼
Peers Comparison (net balance, bootstrap intervals, rating cross-check)
   │
   ▼
Former vs. Current Employees (exit reasons, retention matrix)
   │
   ▼
Prioritisation Matrix & Recommendations for HR
```

### 1.3. Schema

```text
ING-glasdoor-analysis/
│
├── exercise.ipynb       # Complete analysis workflow
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

The notebook also expects the dataset at `data/data.parquet` and creates an `outputs/` folder on its first run (sentiment scores and cached embeddings, see Getting started).

### 1.4. Technologies

| Category            | Technologies                                          |
| ------------------- | ----------------------------------------------------- |
| Language            | Python                                                |
| Data manipulation   | Pandas, NumPy                                         |
| Data visualization  | Matplotlib, Seaborn                                   |
| NLP                 | Hugging Face Transformers, Sentence-Transformers, PyTorch |
| Topic modeling      | BERTopic, UMAP, HDBSCAN, Scikit-learn                 |
| Statistics          | Bootstrap, two-proportion z-test, Benjamini-Hochberg  |
| Environment         | Jupyter Notebook                                      |

## 2. Data

The dataset merges Glassdoor Job Reviews 1 & 2 into one Parquet file (July 2018 - April 2023, firms with 300+ reviews). I kept the four firms above and worked with a balanced sample of **5,998 reviews** (~1,500 per firm). Three decisions shaped the rest of the analysis:

- **Balanced sample:** capping every firm at 1,500 reviews means no firm weighs more than the others in the topic models, and it keeps the transformer computing time reasonable. The sample is stratified by year and current/former status with a fixed seed. Mean rating was *not* used to stratify, so comparing it with the full data is a real test of representativeness.
- **Almost no text cleaning:** I only normalize whitespace. No stemming or stop-word removal, because both the sentiment model and the sentence embedder read the whole sentence (negations, word order) and removing words would only make them understand less.
- **Splitting `current`:** it mixed employment status and tenure (_"Former Employee, more than 3 years"_), so I turned it into two variables.

## 3. Method

There's no model to train here: this is a diagnosis, not a prediction. The pretrained transformers are tools to turn thousands of free-text reviews into something comparable across firms. The work is in how they're chained, checked and interpreted.

### 3.1. Sentiment as a quality check, not a metric

Pros and cons already come with an expected sentiment, so I didn't use sentiment to rank firms. I used `distilbert-base-uncased-finetuned-sst-2-english` to check that the data behaves: if the model called most cons positive, the topic analysis couldn't be trusted. It passes in all four firms: 91.5%-93.7% of pros come out positive and 83.0%-85.1% of cons negative.

I also read the discordant cases by hand. About half of the "positive" cons are just empty answers ("none") and about 40% are real complaints worded neutrally, which polarity can't see. So sentiment is never used to throw reviews away.

### 3.2. Topic modeling

Every review's pros (or cons) is embedded with `all-MiniLM-L6-v2` and clustered with BERTopic, one model for pros and one for cons, both trained on the four firms together so topics are comparable.

The pros settings didn't work for cons. With them, the cons model collapsed into 7 topics with 79.4% of the reviews in one generic cluster, because cons are longer and mix several complaints. Switching HDBSCAN to `leaf` fixed it:

| Setting                          | Pros     | Cons     |
| -------------------------------- | :------: | :------: |
| HDBSCAN `min_cluster_size`       | 30       | 40       |
| HDBSCAN selection method         | `eom`    | `leaf`   |
| Topics found                     | 30       | 24       |
| Outliers (raw)                   | 35.5%    | 45.4%    |
| Reassignment threshold (cosine)  | 0.6      | 0.4      |
| Outliers (after reassignment)    | 14.9%    | 9.8%     |

The outliers weren't noise. Reading a sample showed mostly multi-theme or niche comments, so instead of dropping them I reassigned them to their closest topic, choosing the threshold after testing 0.3-0.7.

### 3.3. From topics to business categories

Topics aren't something HR can act on, so I grouped them by hand into 6 comparable categories (Compensation & benefits, Work-life balance & hours, Career & development, Management & leadership, Culture & people, Processes & technology), defined by the HR lever each one points to rather than by the wording. A few one-sided categories (like *Company brand & prestige*) are shown but kept out of the comparison. Doubtful topics were read by hand and a traceability table links every topic to its category.

Since grouping and outlier reassignment are my decisions, I re-ran ING's gaps without them to see what survived. Strengths held in every scenario; *Processes & technology* is the fragile one.

### 3.4. How I compared firms

Everything uses all of a firm's reviews as denominator, and the benchmark is the mean of the three peers, in percentage points. Groups are small, so gaps come with 95% bootstrap intervals and tests use the Benjamini-Hochberg correction. My rule was simple: a category is only a problem if ING is worse than the market **and** the complaints actually hurt rating or recommendation. Talking about something more isn't the same as having a problem with it.

## 4. Results

### 4.1. Where ING stands

ING's mean rating is 4.00 against 3.59-3.71 for the peers, with the advantage in every year. Among those who express an opinion, 81.9% recommend ING vs. 61.9%-73.6% in the peers, although 43% of ING reviews give no opinion, so I trust the rating more than the recommendation rate.

It's not an illusion held up by people still cashing their paycheck either: current employees rate ING 4.09 and former ones 3.84, and ING's leavers recommend it (74.7%) about as much as the peers' *current* employees.

### 4.2. Diagnosis by category

| Category                      | Verdict                    | Key evidence |
| ----------------------------- | -------------------------- | ------------ |
| **Culture & people**          | Strength                   | Net gap +4.2 pp: more praise and no more complaints. |
| **Work-life balance & hours** | Strength                   | Net gap +3.6 pp: more praise, fewer complaints, and complaining doesn't lower the rating. |
| **Processes & technology**    | Improvement area (fragile) | The only category where ING leavers complain more than the peers' leavers (14.7% vs. 10.3%). Shrinks without outlier reassignment. |
| **Management & leadership**   | Risk to monitor            | Costliest complaint everywhere (-0.68 rating points), but peers show the same penalty and ING gets fewer complaints. |
| **Compensation & benefits**   | Not a weakness             | Top complaint among ING leavers, but equal to peers. The pay problem is BNP-Paribas's. |
| **Career & development**      | Not a weakness             | Fewer complaints than peers, but ING is praised for it less. |

On *Processes & technology*: it looks like chronic friction more than a reason to leave. People who stay complain as much as people who leave, those who complain don't rate ING lower, and it's the same before and after 2020. Quotes point to bureaucracy, layers and slow decisions.

## 5. What I'd actually do about it

- **Build the employer-branding message on culture and work-life balance.** They're the only two areas where employees praise ING more than the peers' employees do, without more complaints. Use real employee testimonials in the careers site, job ads and interviews, and back them with concrete career paths and mobility examples, since career growth and prestige are what ING is least associated with.
- **Run a "less red tape" review.** Start with listening sessions and exit interviews to find *which* approvals and systems cause the friction, then pick 2-3 quick wins and communicate them. Since stayers complain as much as leavers, I wouldn't expect a huge impact on turnover: quick wins, not a transformation program.
- **Watch line management, but don't panic.** Add manager questions to pulse surveys and exit interviews and offer coaching where scores are low. Bad management hurts wherever it appears, and ING suffers it less often than its peers.
- **Don't touch pay or career frameworks because of this data.** Compensation complaints are in line with the market and don't lower ratings.

## 6. Limitations

- **Glassdoor isn't the whole workforce.** Reviews are voluntary and skew positive, so these results describe what *reviewers* say.
- **The data ends in April 2023,** so ING's situation today may differ.
- **Different markets and businesses.** Part of the gaps may reflect business mix and not HR policy.
- **Language:** non-English reviews weren't removed, even though the sentiment model and the embedder are English-only. I didn't measure how many there are.
- **One category per review.** Multi-theme reviews get forced into one category, and a big share stays as *Mixed / generic* (28% of pros, 15% of cons). That's why I only interpret gaps versus peers, never absolute levels.
- **Small groups.** ING has only 491 former-employee reviews, so most intervals in the leavers section include zero.

## 7. Future work

- Validate everything against internal data: turnover by area and tenure, exit interviews, engagement surveys.
- Break down *Processes & technology* by country and business area to find where the friction really is.
- Replace the single-category assignment with a multi-label or aspect-based approach.
- Refresh the analysis with recent reviews and add more peers, including non-bank competitors for digital and tech talent.

## 8. Getting started

```bash
git clone https://github.com/rubengil-dev/ING-glasdoor-analysis.git
cd ING-glasdoor-analysis
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews), merge Job Reviews 1 & 2 into a single Parquet file, place it at `data/data.parquet` (create the `data/` folder if it doesn't exist), and then:

```bash
jupyter notebook exercise.ipynb
```

The first run of the sentiment analysis took around 1:30h, so the notebook saves checkpoints in `outputs/` and reuses them afterwards. Delete those files if you change the sample.
