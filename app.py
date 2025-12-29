import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai
from google.genai import types

# --- Deterministic Video Link System ---
class VideoDatabase:
    def __init__(self):
        self.video_map = {} # {"hareket ismi": "url"}
        self.load_database()
    
    def load_database(self):
        """Tüm TXT dosyalarını tarar ve Hareket -> URL eşleşmesi çıkarır"""
        if not os.path.exists("data"): return
        
        for file in os.listdir("data"):
            if file.endswith(".txt"):
                with open(os.path.join("data", file), "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for i in range(len(lines)):
                        line = lines[i].strip()
                        # Eğer satır bir URL ise (https://youtu...)
                        if line.startswith("https://") and i > 0:
                            # Bir önceki satır hareket ismidir (Örn: "1)Smith Machine...")
                            prev_line = lines[i-1].strip()
                            # İsim temizliği: "1) Hareket" -> "hareket"
                            clean_name = re.sub(r'^\d+\)', '', prev_line).strip()
                            
                            # BOŞ KEY KONTROLÜ: Eğer isim boşsa veya çok kısaysa ekleme!
                            if len(clean_name) > 2:
                                self.video_map[clean_name.lower()] = line
                            
    def get_video_link(self, query_text):
        """Metin içinde geçen hareketleri bulur ve link ekler"""
        if not query_text: return ""

        processed_text = query_text
        sorted_keys = sorted(self.video_map.keys(), key=len, reverse=True)
        
        for exercise in sorted_keys:
            pattern = re.compile(re.escape(exercise), re.IGNORECASE)
            
            if pattern.search(processed_text):
                url = self.video_map[exercise]
                link_md = f" [📺 Video]({url})"
                
                def replace_func(match):
                    end = match.end()
                    snippet_after = processed_text[end:end+5]
                    if snippet_after.startswith("(") or snippet_after.startswith("["):
                         return match.group(0)
                    return f"{match.group(0)}{link_md}"
                
                processed_text = pattern.sub(replace_func, processed_text)
                
        return processed_text

# Video veritabanını başlat
video_db = VideoDatabase()

# ÖNEMLİ: API Anahtarı Ayarı
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("⚠️ API anahtarı bulunamadı! Lütfen Streamlit Secrets'a 'GOOGLE_API_KEY' ekleyin.")
    st.stop()

st.set_page_config(page_title="V-Fit AI Koç", page_icon="💪", layout="wide")

# --- Banner / Kapak Görseli ---
if os.path.exists("data/arkaplan resmi.webp"):
    st.image("data/arkaplan resmi.webp", use_container_width=True)

@st.cache_resource
def init_rag():
    # 1. Klasör Kontrolü
    if not os.path.exists("data") or not os.listdir("data"):
        return None
    
    # 2. PDF ve TXT Dosyalarını Yükle
    docs = []
    
    # PDF Yükleyici
    pdf_loader = PyPDFDirectoryLoader("data")
    docs.extend(pdf_loader.load())
    
    # TXT Yükleyici
    for file in os.listdir("data"):
        if file.endswith(".txt"):
            txt_loader = TextLoader(os.path.join("data", file))
            docs.extend(txt_loader.load())

    # 3. Metinleri Parçalara Böl
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    
    # 4. Ücretsiz Embedding Modeli (HuggingFace)
    # Google Embeddings yerine HuggingFace kullanıyoruz (API kotası yok)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 5. Vektör Veritabanı
    vectorstore = Chroma(embedding_function=embeddings)
    
    # 6. Verileri 5'erli paketler halinde ekle
    progress_bar = st.sidebar.progress(0)
    total_splits = len(splits)
    
    for i in range(0, total_splits, 5):
        chunk = splits[i:i+5]
        vectorstore.add_documents(chunk)
        time.sleep(1)  # Kota koruması
        
        progress = (i + len(chunk)) / total_splits
        progress_bar.progress(progress)
    
    progress_bar.empty()
    return vectorstore

# Google Genai Client oluştur
client = genai.Client(api_key=api_key)

# Arayüz
st.sidebar.title("🏋️‍♂️ Profil ve Ayarlar")
with st.sidebar:
    st.header("Kişisel Bilgiler")
    name = st.text_input("Adınız:", "Misafir")
    gender = st.radio("Cinsiyet:", ("Erkek", "Kadın"))
    age = st.number_input("Yaş:", 10, 100, 25)
    height = st.number_input("Boy (cm):", 100, 250, 175)
    weight = st.number_input("Kilo (kg):", 40, 150, 80)
    goal = st.selectbox("Hedefin:", ("Kas Kütlesi Kazanımı", "Yağ Yakımı", "Kondisyon", "Sağlıklı Yaşam"))
    
    frequency = st.slider("Haftada kaç gün antrenman?", 1, 7, 3)
    
    st.markdown("---")
    st.caption("Kaynak: V-Fit AI & Submaksimal Fitness")
    st.caption("Sürüm: v1.1.0 (Native SDK)")
    
    # BMI Hesaplama
    bmi = weight / ((height/100)**2)
    st.metric("Vücut Kitle İndeksi (BMI)", f"{bmi:.1f}")
    
    # BMI Skalası
    if bmi < 18.5:
        status, color = "Zayıf", "blue"
    elif 18.5 <= bmi < 24.9:
        status, color = "Normal (Fit)", "green"
    elif 25 <= bmi < 29.9:
        status, color = "Kilolu", "orange"
    elif 30 <= bmi < 34.9:
        status, color = "Obez", "red"
    else:
        status, color = "Aşırı Obez", "darkred"
        
    st.markdown(f"**Durum:** <span style='color:{color}; font-size:18px; font-weight:bold'>{status}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Programı İndir Butonu
    if st.button("📥 Programı İndir"):
        if "messages" in st.session_state and st.session_state.messages:
            last_response = st.session_state.messages[-1]["content"]
            st.download_button(
                label="Dosyayı Kaydet",
                data=last_response,
                file_name=f"VFit_Program_{name}.md",
                mime="text/markdown"
            )
        else:
            st.warning("Henüz bir program oluşturulmadı.")

    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()

# RAG Sistemini Başlat
vectorstore = init_rag()

if vectorstore is None:
    st.error("Veri klasörü bulunamadı veya boş! Lütfen 'data' klasörüne PDF/TXT ekleyin.")
else:
    # Retriever oluştur
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Chat Arayüzü
    st.header("🤖 V-Fit Asistanı")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Sorunu sor (Örn: Bana program hazırla)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Konuşma Geçmişini Hazırla
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
        
        # Retrieval: İlgili dökümanları çek
        relevant_docs = retriever.get_relevant_documents(prompt)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Gelişmiş Prompt
        system_instruction = f"""
        BİR ROL YAP: Sen "V-Fit AI Koçu"sun. İnsanlara sağlık, fitness ve beslenme konularında yardımcı olan, ZEKİ, KİBAR, MOTİVE EDİCİ ve PROFESYONEL bir yapay zekasın.
        
        KULLANICI PROFİLİ:
        - İsim: {name}
        - Durum: {gender}, {age} yaşında, {height}cm, {weight}kg.
        - Analiz: BMI {bmi:.1f} ({status}).
        - Hedef: {goal}
        - Zaman: Haftada {frequency} gün antrenman yapabilir.
        
        KURALLAR VE DAVRANIŞLAR:
        1. **Nezaket & Motivasyon:** Her cevaba nazik bir selamlama veya motive edici bir sözle başla.
        2. **Sadece Fitness:** Eğer kullanıcı fitness dışı bir şey söylerse; kibarca "Ben sadece bir fitness antrenörüyüm, lütfen antrenman veya beslenme konuşalım." diyerek konuyu kapat.
        3. **Program Formatı (4+4+4+4):** Kullanıcı "program" istediğinde, süreci 4 bloğa bölerek anlat:
           - **1. Blok (Hafta 1-4):** Adaptasyon ve Alışma.
           - **2. Blok (Hafta 5-8):** Gelişim ve Yüklenme.
           - **3. Blok (Hafta 9-12):** Güç ve Dayanıklılık.
           - **4. Blok (Hafta 13-16):** Definasyon ve Sonuç.
        4. **Video Entegrasyonu:** Sen sadece hareket isimlerini doğru yaz. Linkleri sistem otomatik ekleyecek.
        5. **Dil Desteği:** Kullanıcı "Arka Kol" derse bunu "Triceps", "Ön Kol" derse "Biceps", "Omuz" derse "Shoulder" olarak eşleştir.
        6. **Akıllı Tepki:** Kullanıcı sadece "Merhaba", "Selam" gibi tanışma cümleleri kurarsa, direkt program hazırlama. Hal hatır sor, hedefini teyit et ve motive et.
        7. **Sağlık Uyarısı:** Tıbbi tavsiye vermediğini, spora başlamadan önce doktora danışılması gerektiğini nazikçe hatırlat.
        
        MİSYONUN: Kullanıcıyı hedefine ({goal}) ulaştırmak için en bilimsel ve uygulanabilir yolu çizmek.

        İŞTE KAYNAK BİLGİLER (Bu bilgileri kullanarak cevap ver):
        {context}
        
        Geçmiş Konuşmalar:
        {chat_history}
        
        Kullanıcı Sorusu: {prompt}
        
        Türkçe cevap ver:
        """
        
        with st.spinner('V-Fit Koç düşünüyor ve program hazırlıyor...'):
            try:
                # Google Genai SDK with Native API
                response = client.models.generate_content(
                    model='gemini-2.0-flash-exp',
                    contents=system_instruction
                )
                
                raw_response = response.text
                
            except Exception as e:
                raw_response = f"❌ Üzgünüm, bir hata oluştu: {str(e)}\n\nLütfen API anahtarınızı kontrol edin veya daha sonra tekrar deneyin."
        
        # --- POST-PROCESSING: Link Düzeltme ---
        final_response = video_db.get_video_link(raw_response)
        
        with st.chat_message("assistant"):
            st.markdown(final_response)
            
            # Video linkini göster
            video_links = re.findall(r'(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://youtu\.be/[\w-]+)', final_response)
            if video_links:
                st.video(video_links[0])
            
            # Kas Grubu Görselini veya PDF dosyasını Bul ve Göster
            found_files = []
            
            for file in os.listdir("data"):
                file_lower = file.lower()
                file_name_clean = os.path.splitext(file_lower)[0]
                keywords = file_name_clean.split()
                
                match_count = 0
                for kw in keywords:
                    if kw in raw_response.lower():
                        match_count += 1
                
                if match_count == len(keywords) and len(keywords) > 0:
                    if file not in found_files:
                        st.info(f"💡 İlgili Kaynak Bulundu: {file}")
                        
                        file_path = os.path.join("data", file)
                        if file.endswith((".jpg", ".png", ".jpeg", ".webp")):
                            st.image(file_path, caption=file_name_clean, use_container_width=True)
                        elif file.endswith(".pdf"):
                            with open(file_path, "rb") as pdf_file:
                                st.download_button(label=f"📄 {file} İndir", data=pdf_file, file_name=file, mime="application/pdf")
                        
                        found_files.append(file)

            st.session_state.messages.append({"role": "assistant", "content": final_response})