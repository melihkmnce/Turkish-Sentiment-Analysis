import re
import numpy as np
import pandas as pd
import chardet
import nltk
import pickle
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report



# Dosyanın kodlamasını algıla
with open("comments.csv", "rb") as file:
    file_type = chardet.detect(file.read())
    encoding_type = file_type.get('encoding')

# Veriyi yükle ve etiketleri dönüştür
print("Veri yükleniyor...")
data = pd.read_csv("comments.csv", encoding=encoding_type, delimiter="\t")

# 2 içeren satırları sil
data = data[data['durum'] != 2]

print(data.head())

# Eksik etiketleri (NaN) olan satırları çıkar
data = data.dropna(subset=['durum'])

# Veri uzunluklarını kontrol et
print(f"Length of data['metin']: {len(data['metin'])}")
print(f"Length of data['durum']: {len(data['durum'])}")

# Stopwords setini oluştur
sword = set(stopwords.words("turkish"))
print("Stopwords: ", sword)

# Ön işleme fonksiyonu
def process(comment):
    comment = BeautifulSoup(comment, features="html.parser").get_text()
    comment = re.sub("[^a-zA-ZçÇğĞıİöÖşŞüÜ]", ' ', comment)
    comment = comment.lower()
    comment = comment.split()
    comment = [i for i in comment if i not in sword]
    return " ".join(comment)

# Metinleri işleme
print("Metinler işleniyor...")
# Metinleri işleme
trainxall = []
for i in range(len(data["metin"])):
    if (i + 1) % 1000 == 0:
        print("Process Step:", i + 1)
    try:
        if pd.notna(data["metin"][i]):  # Sadece geçerli metinleri işleyin
            trainxall.append(process(data["metin"][i]))
        else:
            print(f"Skipping row {i} due to missing text.")
    except Exception as e:
        print(f"An error occurred during the operation (line {i}): {e}")

# Özellikler ve etiketler
x = np.array(trainxall)
y = np.array(data["durum"].iloc[:len(trainxall)])  # x ve y boyutlarının uyumlu olmasını sağla

# Veri boyutlarını kontrol et
print(f"Length of x: {len(x)}")
print(f"Length of y: {len(y)}")


# Eğitim ve test setlerine ayırma
print("Veriler bölünüyor...")
trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.15, random_state=33)

# Metinleri sayısal forma dönüştürme
print("Metinler vektörleştiriliyor...")
vectorizer = CountVectorizer(max_features=10000)
trainx1 = vectorizer.fit_transform(trainx).toarray()
testx1 = vectorizer.transform(testx).toarray()

# Modeli oluşturma ve eğitme
print("Model eğitiliyor...")
rf = RandomForestClassifier(n_estimators=300, random_state=33)
rf.fit(trainx1, trainy)

# Test seti tahminleri
print("Tahminler yapılıyor...")
predictions = rf.predict(testx1)

# Model performansı
auc_score = roc_auc_score(testy, rf.predict_proba(testx1)[:, 1])  # Sadece 1. sınıf için
print(f"Binary ROC AUC Score: {auc_score}")

print(classification_report(testy, predictions))

# Model ve vektörleştiriciyi kaydetme
print("Model ve vektörleştirici kaydediliyor...")
with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(rf, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer saved successfully.")

