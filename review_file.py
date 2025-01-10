import pickle
import json
import os

def load_and_inspect_pkl(file_path):
    try:
        # PKL dosyasını açma
        with open(file_path, "rb") as file:
            data = pickle.load(file)

        print("PKL Dosyası Başarıyla Yüklendi!")
        print("-" * 50)

        # İçeriğin türünü kontrol etme
        print(f"PKL dosyasının veri türü: {type(data)}")

        # Veri türüne göre inceleme
        if isinstance(data, dict):
            print(f"Sözlük içeriyor. Anahtar sayısı: {len(data)}")
            for key, value in list(data.items())[:5]:  # İlk 5 anahtar-değeri yazdırma
                print(f"Key: {key}, Value Type: {type(value)}")
        elif isinstance(data, list):
            print(f"Liste içeriyor. Eleman sayısı: {len(data)}")
            print("İlk 5 Eleman:", data[:5])
        elif hasattr(data, "get_params"):  # Makine öğrenimi modelleri için
            print("Makine öğrenimi modeli tespit edildi.")
            print("Model Parametreleri:")
            print(data.get_params())
        else:
            print("Diğer veri tipi tespit edildi:")
            print(data)

        # Kaydetme seçeneği
        save_option = input("Bu içeriği JSON veya metin dosyasına kaydetmek ister misiniz? (e/h): ").lower()
        if save_option == "e":
            save_to_file(data, file_path)
        else:
            print("\nİçerik:",data)

    except Exception as e:
        print(f"Hata oluştu: {e}")


def save_to_file(data, original_file_path):
    # Dosya kaydı için format belirleme
    base_name = os.path.splitext(original_file_path)[0]
    if isinstance(data, dict):
        json_file = f"{base_name}_output.json"
        with open(json_file, "w") as json_out:
            json.dump(data, json_out, indent=4)
        print(f"İçerik JSON formatında kaydedildi: {json_file}")
    else:
        text_file = f"{base_name}_output.txt"
        with open(text_file, "w") as text_out:
            text_out.write(str(data))
        print(f"İçerik metin formatında kaydedildi: {text_file}")


pkl_file_path = "random_forest_model.pkl"
load_and_inspect_pkl(pkl_file_path)
