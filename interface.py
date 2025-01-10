import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pickle
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# Stopword setini oluşturma
sword = set(stopwords.words("turkish"))

# Yorum işleme fonksiyonu
def process_input(comment):
    comment = BeautifulSoup(comment, features="html.parser").get_text()
    comment = re.sub("[^a-zA-ZçÇğĞıİöÖşŞüÜ]", ' ', comment)
    comment = comment.lower()
    comment = comment.split()
    comment = [word for word in comment if word not in sword]
    return " ".join(comment)

# Model ve vektörleştiriciyi yükleme fonksiyonu
def load_model_and_vectorizer():
    try:
        with open("random_forest_model.pkl", "rb") as model_file:
            rf = pickle.load(model_file)
        with open("vectorizer.pkl", "rb") as vec_file:
            vectorizer = pickle.load(vec_file)
        return rf, vectorizer
    except FileNotFoundError as e:
        messagebox.showerror("Error", f"Model or vectorizer file not found: {e}")
        exit()

# Model ve vektörleştiriciyi yükleme
rf, vectorizer = load_model_and_vectorizer()

# Yorum analizi fonksiyonu
def analyze_comment():
    user_comment = comment_entry.get("1.0", tk.END).strip()
    if not user_comment:
        messagebox.showwarning("Input Error", "Please enter a comment.")
        return

    # Yorumu işleme ve tahmin
    processed_input = process_input(user_comment)
    processed_vector = vectorizer.transform([processed_input]).toarray()
    prediction = rf.predict(processed_vector)[0]
    probability = rf.predict_proba(processed_vector)[0][1]  # Pozitif sınıf olasılığı
    result = f"Sonuç: {'Pozitif' if prediction == 1 else 'Negatif'}"

    # Sonucu kullanıcıya gösterme
    result_label.config(text=result, fg="#2ecc71" if prediction == 1 else "#e74c3c")

# Tkinter arayüzü
app = tk.Tk()
app.title("Yorum Duygu Analizi")
app.geometry("600x500")
app.resizable(False, False)
app.configure(bg="#f7f7f7")

# Gradient arka plan
canvas = tk.Canvas(app, width=600, height=500)
canvas.pack(fill="both", expand=True)
canvas.create_rectangle(0, 0, 600, 250, fill="#3498db", outline="")
canvas.create_rectangle(0, 250, 600, 500, fill="#ecf0f1", outline="")

# Başlık
title_label = tk.Label(
    app,
    text="Yorum Duygu Analizi",
    font=("Helvetica", 24, "bold"),
    bg="#3498db",
    fg="white",
)
title_label.place(relx=0.5, y=50, anchor="center")

# Çerçeve
frame = tk.Frame(app, bg="white", relief="flat", borderwidth=0)
frame.place(relx=0.5, rely=0.6, anchor="center", width=550, height=300)

# Yorum girişi
comment_label = tk.Label(
    frame, text="Kutucuğa metni giriniz:", font=("Helvetica", 14), bg="white", fg="#2c3e50"
)
comment_label.place(x=20, y=20)

comment_entry = tk.Text(frame, height=5, width=52, font=("Helvetica", 12), relief="solid", borderwidth=1)
comment_entry.place(x=20, y=60)

# Analiz düğmesi
analyze_button = tk.Button(
    frame,
    text="Analiz",
    font=("Helvetica", 14, "bold"),
    bg="#3498db",
    fg="white",
    activebackground="#2980b9",
    activeforeground="white",
    borderwidth=0,
    command=analyze_comment,
    height=1,
    width=10,
)
analyze_button.place(x=220, y=200)

# Tahmin sonucu
result_label = tk.Label(frame, text="", font=("Helvetica", 16, "bold"), bg="white", fg="#34495e")
result_label.place(x=210, y=250)

# Uygulamayı çalıştır
app.mainloop()
