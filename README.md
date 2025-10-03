# Text Classification – AI vs Human Content

Ky projekt eshte zhvilluar si pjese e 
**LinkPlus IT – AI Internship Challenge (Text Classification Task)**.

Qellimi eshte te trajnohet nje model qe klasifikon tekstet si **Human-written** ose **AI-generated** duke perdorur teknika te thjeshta te perpunimit te gjuhes natyrore (NLP).

---

## Dataset

Dataset-i eshte marre nga Kaggle:  
 [AI vs Human Text Dataset]https://www.kaggle.com/datasets/pratyushpuri/ai-vs-human-content-detection-1000-record-in-2025/data

- Numri i mostrave: **1367**
- Kategori: **0 = Human**, **1 = AI**
- Jane shtuar edhe disa tipare numerike (word count, sentence count, sentiment score, etj.)
- Per trajnime eshte perdorur versioni i pastruar:  
  `ai_human_content_detection_dataset_clean.csv`

---

## Hapat e Implementimit

### 1. Data Preparation
- Heqja e rreshtave me vlera `NaN` në `text_content`.
- Heqja e duplikatëve.
- Zevendesimi i vlerave mungese ne kolonat numerike me **mesatare**.
- Preprocessing teksti:
  - Lowercase
  - Heqje pikesimi & simboleve
  - Heqje stopwords
- Krijimi i kolones `clean_text`.

### 2. Exploratory Data Analysis (EDA)
- Numri i mostrave per kategori.
- Vizualizimi i fjaleve me te shpeshta.
- Korrelacioni midis variablave numerike dhe label-it.
- Heatmap per matricen e korrelacionit.

### 3. Model Training
- Dataset-i u nda ne **train (80%)** dhe **test (20%)**.
- Teksti u transformua me **TF-IDF Vectorizer (max_features=5000)**.
- U trajnuan 4 modele:
  - Logistic Regression (balanced)
  - Naive Bayes
  - Linear SVC
  - Random Forest
- Modelet u vlersuan me:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**

### 4. Krahasimi i Modeleve
- Rezultatet u shfaqen ne nje tabele dhe bar chart.
- **Modeli me i mire sipas F1-score:** Logistic Regression (balanced) (~0.53).

### 5. Prediction Script
- Funksioni `predict_sentence(text)` kthen:
  ```json
  {
    "input": "Teksti i dhënë",
    "prediction": 0,
    "proba": {"0": 0.52, "1": 0.48}
  }


## API
U zhvillua nje API me FastAPI:
GET / → status
POST /predict → pranon nje tekst dhe kthen prediction + probabilitetet
Swagger UI → http://127.0.0.1:8000/docs

## Si te ekzekutohet projekti
```bash

git clone <repo-url>
cd text-classifier

```
2.  Krijo virtual environment (opsional, por e rekomanduar)
```bash

python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac

```

3. Instalo kerkesat
```bash

pip install -r requirements.txt

```

4. Ekzekuto notebook-un
Hap analysis.ipynb në Jupyter/VS Code dhe ekzekuto qelizat.

5. Nis API-në
```bash

uvicorn app:app --reload

```
Pastaj hap:
```bash

http://127.0.0.1:8000
http://127.0.0.1:8000/docs

```


## Kufizime
  Dataset-i eshte i vogel (~1367 mostra) → performanca eshte e kufizuar (accuracy ≈ 0.5).
  Tiparet numerike kane korrelacion te ulet me label-in → ndikim i vogel ne parashikime.

## Permiresime te mundshme:
   Dataset me i madh dhe me i balancuar
   Perdorimi i n-grams dhe embeddings (Word2Vec, BERT)
   Kombinimi i tekstit me tiparet numerike
   Hyperparameter tuning për modelet ekzistuese.

## Shembull Output-i nga API
```json
{
  "input": "Write a short creative story about a dragon.",
  "prediction": 1,
  "proba": {
    "0": 0.41,
    "1": 0.59
  }
}
```

## Përmbajtja e projektit

- `analysis.ipynb` - notebook me analizen dhe trajnimin e modeleve
- `app.py` - API me FastAPI per parashikime
- `model.pkl` - modeli i trajnuar
- `vectorizer.pkl` - TF-IDF vectorizer i ruajtur
- `ai_human_content_detection_dataset_clean.csv` - dataset i pastruar
- `requirements.txt` - paketat Python qe duhen për instalim
- `README.md` - ky dokument

