# V-Fit AI Koç 💪🤖

Bu proje, Google Gemini yapay zekasını kullanarak kişiselleştirilmiş fitness antrenman programları ve beslenme önerileri sunan bir **Yapay Zeka Fitness Koçu** uygulamasıdır.

## 🌟 Özellikler
- **Kişisel Analiz:** Yaş, boy, kilo, cinsiyet ve hedefe göre analiz.
- **Akıllı Antrenman Planı:** 16 haftalık (4 Blok) periodizasyon sistemi.
- **Görsel & Video Destek:** Hareket isimlerini algılar, ilgili PDF/Görsel ve YouTube videolarını sunar.
- **Türkçe & İngilizce Eşleşme:** "Arka Kol" dediğinizde "Triceps" egzersizlerini veritabanından bulur.
- **BMI Hesaplama:** Vücut kitle indeksinizi hesaplar ve durumunuzu gösterir.

## 🚀 Kurulum (Başka Bir Bilgisayarda Nasıl Çalıştırılır?)

Bu projeyi başka bir bilgisayarda çalıştırmak için şu adımları izleyin:

### 1. Projeyi İndirin
Terminali açın ve projeyi bilgisayarınıza çekin:
```bash
git clone https://github.com/gulsoyvolkan-ui/vfit-ai-koc.git
cd vfit-ai-koc
```

### 2. Sanal Ortam Oluşturun (Önerilen)
```bash
# MacOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Kütüphaneleri Yükleyin
Gerekli tüm paketleri tek komutla yükleyin:
```bash
pip install -r requirements.txt
```

### 4. API Anahtarını Ayarlayın
Google Gemini API anahtarınızı `app.py` içinde veya (daha güvenlisi) `.env` dosyasında tanımlayın.
*(Not: Streamlit Cloud üzerindeyseniz, Secrets bölümünden ayarlayabilirsiniz).*

### 5. Uygulamayı Başlatın
```bash
streamlit run app.py
```

## 📂 Proje Yapısı
- `app.py`: Ana uygulama dosyası.
- `data/`: Antrenman bilgileri (PDF/TXT) ve görseller.
- `requirements.txt`: Python kütüphane listesi.
- `packages.txt`: Sistem gereksinimleri (Linux/Cloud için).

## 🛠 Kullanılan Teknolojiler
- Python 3.11
- Streamlit
- LangChain
- Google Gemini (GenAI)
- ChromaDB (Vektör Veritabanı)

---
*Geliştirici: Volkan Gülsoy*
