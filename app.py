import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# ÖNEMLİ: API Anahtarı Ayarı
# Streamlit Cloud üzerinde 'st.secrets' kullanılır. Yerelde ise bu satır çalışır.
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    # Yerel Geliştirme İçin (Canlıya alırken burayı silmeniz önerilir)
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBEIe2cTwCBMvtmwk15n4DYm0kiDWiXCyw"

st.set_page_config(page_title="V-Fit AI Koç", page_icon="💪", layout="wide")

# --- Banner / Kapak Görseli ---
# Kullanıcının eklediği görseli en tepeye yerleştirelim
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
    
    # TXT Yükleyici (Manuel)
    for file in os.listdir("data"):
        if file.endswith(".txt"):
            from langchain_community.document_loaders import TextLoader
            txt_loader = TextLoader(os.path.join("data", file))
            docs.extend(txt_loader.load())

    # 3. Metinleri Parçalara Böl
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    
    # 4. Google Embedding Modelini Tanımla
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # 5. KOTA KORUMALI YÜKLEME (Vektör Veritabanı)
    # Boş bir veritabanı oluştur
    vectorstore = Chroma(embedding_function=embeddings)
    
    # Verileri 5'erli paketler halinde gönder ve aralarda uyu (Sleep)
    progress_bar = st.sidebar.progress(0)
    total_splits = len(splits)
    
    for i in range(0, total_splits, 5):
        chunk = splits[i:i+5]
        vectorstore.add_documents(chunk)
        
        # Kota hatasını önlemek için bekleme süresi
        time.sleep(2) 
        
        # İlerlemeyi göster
        progress = (i + len(chunk)) / total_splits
        progress_bar.progress(progress)
    
    progress_bar.empty() # İşlem bitince barı kaldır
    return vectorstore

# Arayüz
# Arayüz
st.sidebar.title("🏋️‍♂️ Profil ve Ayarlar")
with st.sidebar:
    st.header("Kişisel Bilgiler")
    name = st.text_input("Adınız:", "Misafir")
    gender = st.radio("Cinsiyet:", ("Erkek", "Kadın"))
    age = st.number_input("Yaş:", 10, 100, 25)
    height = st.number_input("Boy (cm):", 100, 250, 175)
    weight = st.number_input("Kutu (kg):", 40, 150, 80)
    goal = st.selectbox("Hedefin:", ("Kas Kütlesi Kazanımı", "Yağ Yakımı", "Kondisyon", "Sağlıklı Yaşam"))
    
    frequency = st.slider("Haftada kaç gün antrenman?", 1, 7, 3)
    
    st.markdown("---")
    st.header("Vücut Analizi")
    
    # BMI Hesaplama
    bmi = weight / ((height/100)**2)
    st.metric("Vücut Kitle İndeksi (BMI)", f"{bmi:.1f}")
    
    # BMI Skalası ve Renkler
    if bmi < 18.5:
        status = "Zayıf"
        color = "blue"
    elif 18.5 <= bmi < 24.9:
        status = "Normal (Fit)"
        color = "green"
    elif 25 <= bmi < 29.9:
        status = "Kilolu"
        color = "orange"
    elif 30 <= bmi < 34.9:
        status = "Obez"
        color = "red"
    else:
        status = "Aşırı Obez"
        color = "darkred"
        
    st.markdown(f"**Durum:** <span style='color:{color}; font-size:18px; font-weight:bold'>{status}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Programı İndir Butonu
    if st.button("📥 Programı İndir"):
        # Son cevabı al
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
        st.experimental_rerun()

# RAG Sistemini Başlat
vectorstore = init_rag()

if vectorstore is None:
    st.error("Veri klasörü bulunamadı veya boş! Lütfen 'data' klasörüne PDF/TXT ekleyin.")
else:
    # Zinciri Kur
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

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
        
        # Gelişmiş Prompt (Persona & Yapı)
        system_instruction = f"""
        BİR ROL YAP: Sen "V-Fit AI Koçu"sun. İnsanlara sağlık, fitness ve beslenme konularında yardımcı olan, ZEKİ, KİBAR, MOTİVE EDİCİ ve PROFESYONEL bir yapay zekasın.
        
        KULLANICI PROFİLİ:
        - İsim: {name}
        - Durum: {gender}, {age} yaşında, {height}cm, {weight}kg.
        - Analiz: BMI {bmi:.1f} ({status}).
        - Hedef: {goal}
        - Zaman: Haftada {frequency} gün antrenman yapabilir.
        
        KURALLAR VE DAVRANIŞLAR:
        1. **Nezaket & Motivasyon:** Her cevaba nazik bir selamlama veya motive edici bir sözle başla. (Örn: "Harika bir hedef {name}!", "Seni azimli gördüm!")
        2. **Sadece Fitness:** Eğer kullanıcı fitness dışı, anlamsız veya hakaret içeren bir şey söylerse; kibarca "Ben sadece bir fitness antrenörüyüm, lütfen antrenman veya beslenme konuşalım." diyerek konuyu kapat. Asla kabalaşma.
        3. **Program Formatı (4+4+4+4):** Kullanıcı "program" istediğinde, TEK BİR 16 haftalık tablo yerine, süreci 4 bloğa bölerek anlat:
           - **1. Blok (Hafta 1-4):** Adaptasyon ve Alışma.
           - **2. Blok (Hafta 5-8):** Gelişim ve Yüklenme.
           - **3. Blok (Hafta 9-12):** Güç ve Dayanıklılık.
           - **4. Blok (Hafta 13-16):** Definasyon ve Sonuç.
           *Tabloyu detaylı hazırla ve haftalık gün sayısına ({frequency} gün) sadık kal.*
        4. **Video Entegrasyonu (Görsel Eğitim):** Hareketleri yazarken, eğer veritabanında o hareketin video linki varsa, **MUTLAKA** hareketin hemen altına tıklanabilir link formatında ekle.
           **ÇOK ÖNEMLİ:** Veritabanındaki linki ASLA DEĞİŞTİRME, UYDURMA veya KISALTMA. Kaynak dosyada (txt) ne yazıyorsa aynısını kopyala.
           Örn:
           *   **Triceps Extension**
               [📺 İzle ve Öğren](https://youtu.be/...)
        5. **Link Formatı:** Asla sadece URL yazma. Mutlaka `[Link Başlığı](URL)` formatını kullan.
        6. **Dil Desteği (ÖNEMLİ):** Kullanıcı "Arka Kol" derse bunu "Triceps", "Ön Kol" derse "Biceps/Forearm", "Omuz" derse "Shoulder/Deltoid" olarak eşleştir. Veritabanındaki İngilizce (veya latince) terimleri kullanıcıya açıkla.
        7. **Akıllı Tepki (YENİ):** Kullanıcı sadece "Merhaba", "Selam", "Nasılsın" gibi tanışma cümleleri kurarsa, direkt program hazırlama. Hal hatır sor, hedefini teyit et ve motive et. Sadece "Program hazırla" veya spesifik bir teknik soru gelirse program moduna geç.
        8. **Sağlık Uyarısı (Disclaimer):** Tıbbi tavsiye vermediğini, spora başlamadan önce doktora danışılması gerektiğini nazikçe hatırlat.
        
        MİSYONUN: Kullanıcıyı hedefine ({goal}) ulaştırmak için en bilimsel ve uygulanabilir yolu çizmek.

        
        Geçmiş Konuşmalar:
        {chat_history}
        
        Kullanıcı Sorusu: {prompt}
        """
        
        # Hata yönetimi için try-except bloğu (zaten dışarıda var sistem tarafından yönetilen, ama promptu güvenli hale getirdik)
        full_query = f"{system_instruction} \n Cevap:"
        response = qa_chain.run(full_query)
        
        with st.chat_message("assistant"):
            st.markdown(response)
            
            # 1. Video linkini ayıkla ve oynat
            video_links = re.findall(r'(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://youtu\.be/[\w-]+)', response)
            if video_links:
                st.video(video_links[0])
            
            # 2. Kas Grubu Görselini veya PDF dosyasını Bul ve Göster
            # Cevap içinde geçen kelimelerle dosya isimlerini eşleştir
            found_files = [] # Aynı dosyayı tekrar tekrar göstermemek için
            
            for file in os.listdir("data"):
                file_lower = file.lower()
                file_name_clean = os.path.splitext(file_lower)[0]
                keywords = file_name_clean.split()
                
                # Eşleşme kontrolü (Anahtar kelime cevapta geçiyor mu?)
                # Basit bir set intersection mantığı veya kelime kelime kontrol
                # Örn: "arka kol" dosyasını bulmak için hem "arka" hem "kol" cevapta geçmeli mi? Evet.
                match_count = 0
                for kw in keywords:
                    if kw in response.lower():
                        match_count += 1
                
                # Eğer dosya ismindeki tüm kelimeler cevapta geçiyorsa (veya en az %80'i)
                if match_count == len(keywords) and len(keywords) > 0:
                    if file not in found_files:
                        st.info(f"💡 İlgili Kaynak Bulundu: {file}")
                        
                        file_path = os.path.join("data", file)
                        if file.endswith((".jpg", ".png", ".jpeg", ".webp")):
                            st.image(file_path, caption=file_name_clean, use_container_width=True)
                        elif file.endswith(".pdf"):
                            # PDF indirme butonu koyalım veya görüntüleyelim (Streamlit PDF viewer gerekebilir, şimdilik indirme)
                            with open(file_path, "rb") as pdf_file:
                                st.download_button(label=f"📄 {file} İndir", data=pdf_file, file_name=file, mime="application/pdf")
                        
                        found_files.append(file)

            st.session_state.messages.append({"role": "assistant", "content": response})