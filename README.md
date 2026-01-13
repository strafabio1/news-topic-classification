# Project
Data Science Project


paper su SVM: Text Categorization with Support Vector Machines: Learning with Many Relevant Features (Thorsten Joachims, 1998)

paper su Logistic Regression: Large-scale Bayesian logistic regression for text categorization (Genkin et al., 2007)

Paper per multinomial naive bayes: A Comparison of Event Models for Naive Bayes Text Classification (McCallum & Nigam, 1998).

# Feature Engineering
- ⁠creiamo una feature text_length, è utile per distinguere le classi: Classe 5: Media ~228 caratteri, Classe 2: Media ~456 caratteri, Classe 0: Media ~346 caratteri.
- dividiamo timestamp in giorno_settimana e ora
- ⁠prima di usare il most frequent su timestamp, creiamo una feature is_timestamp_valid booleana (altrimenti diciamo che una notizia è uscita sicuramente il mercoledì solo perché è il most frequent)
- source: teniamo le Top N fonti (es. le prime 20 o 50) e raggruppa tutte le altre in una categoria "Other". Poi applica One-Hot Encoding su queste N+1 categorie
- hyperparameter n_sources_meaningful

# PreProcessing
- applichiamo standardScaler

# GRAFICI 
- istogramma con occorrenze labels per capire se è develpoment.csv è balanced/unbalanced
- istogramma di articolo di esempio e relative parole (per dimostrare che le rare vengono esaltate e le frequenti ignorate)
- istogramma top parole tra tutti gli articoli (e relativo peso)
- istogramma pakeRank/label per mostrare che è inutile
