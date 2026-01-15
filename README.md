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





X = train_df[['text_content','source','page_rank','day_sin','day_cos']], source come ordinal  f1= 0.7210

X = train_df[['text_content','source','page_rank','day_sin','day_cos']], source come categorical  f1= 0.7211

X = train_df[['text_content','source','page_rank']] f1=0.7211

X = train_df[['text_content','source','page_rank','hour_sin','hour_cos']] f1=0.7211

X = train_df[['text_content','source','page_rank','hour_sin','hour_cos']]
f1= 0.7211



PREPROCESSING


-TESTO
Uniamo title e article e calcoliamo tfidf con ngram_range=(1,3) e sublinear_tf=True, min_df=3 per togliere outlier(ngrammi che si ripetono in meno di 3 "documenti") e max_df=0.5 per non considerare parole che si ripetono in più della metà dei documenti (il testo lo decodifichiamo solo per ora, perchè tipo in html l'apostrofo si vede come &39 e decodificandolo si vede l'apostrofo)
ngram_range=(1,3) perchè:
Sports: Il modello vedrebbe parole come "red" o "sox" separatamente. Con i trigrammi cattura "boston red sox" o "new york yankees", che sono segnali inequivocabili di Sport.
lito
Health: Parole come "drug" o "administration" sono generiche. Il trigramma "food drug administration" (FDA) o il bigramma "breast cancer" identificano immediatamente la categoria Salute.

International: Ho trovato "prime minister tony" (probabilmente riferito a Blair o Abbott). Questo identifica una figura politica specifica, molto più utile di "tony" da solo.
Abbiamo solo 1 titolo nan e un articolo nan, possiamo scartarli oppure tenerli visto che comunque abbiamo 

-PAGERANK
Lo trattiamo come categorico, perchè se lo trattassimo come numerico da standardizzare o come ordinale SVM e logistiRegression cercherebbero una relazione lineare, mentre abbiamo solo dei picchi che corrispondono alle singole classi. Tipo rank 2 e 3 sono sempre label technology. Possiamo mostrare grafico  


-SOURCE
Facciamo con Onehot-encoding solo prime 50 e others perchè le prime 50 coprono 87% delle notizie.

-DAY OF WEEK
Estriamo il giorno, e creiamo nuova feature 'is_weekend' perchè per esempio solitamente lo sport esce il weekend mentre la finanza durante la settimana.
Inoltre creiamo feature 'timestamp_missing' perchè abbiamo ben 27000 valori null per il timestamp e molto spesso abbiamo classi con molti timestamp nulli, altre classi con tanti timestamp

-HOUR
Estriamo orario e lo encodizziamo con la cosa ciclica hour_sin e hour_cos.




