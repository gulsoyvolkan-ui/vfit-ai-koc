import streamlit as st
import os
import time
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# ÖNEMLİ: API Anahtarını buraya yapıştır
os.environ["GOOGLE_API_KEY"] = "AIzaSyBEIe2cTwCBMvtmwk15n4DYm0kiDWiXCyw"

st.set_page_config(page_title="V-Fit AI Koç", page_icon="💪", layout="wide")



@st.cache_resource
def init_rag():
    # 1. Klasör Kontrolü
    if not os.path.exists("data") or not os.listdir("data"):
        return None
    
    # 2. PDF ve TXT Dosyalarını Yükle
    docs = []
    # PDF'leri yükle
    pdf_loader = PyPDFDirectoryLoader("data")
    docs.extend(pdf_loader.load())
    
    # TXT (Video Linklerini) yükle
    from langchain_community.document_loaders import TextLoader
    for file in os.listdir("data"):
        if file.endswith(".txt"):
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
    
    # Basit bir bar göstergesi
    st.progress(min(bmi / 40, 1.0))
    
    # Cinsiyete Göre Görsel
    if gender == "Erkek":
        st.info("💪 Güç ve Disiplin!")
    else:
        st.info("🧘‍♀️ Denge ve Güç!")

    st.markdown("---")
    st.caption("Kaynak: V-Fit AI & Submaksimal Fitness")

st.title("💡 V-Fit AI: Akıllı Antrenör")
vectorstore = init_rag()

if vectorstore:
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Nasıl yardımcı olabilirim? (Örn: 'Bana 8 haftalık program yaz')"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
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
        4. **Video Entegrasyonu (Görsel Eğitim):** Hareketleri yazarken, eğer veritabanında o hareketin video linki varsa, **MUTLAKA** hareketin hemen altına linkini ekle. Örn:
           *   **Bench Press**
               (📺 İzle: https://youtube.com/...)
        5. **Link Formatı:** Linkleri tıklanabilir yap.
        6. **Dil Desteği (ÖNEMLİ):** Kullanıcı "Arka Kol" derse bunu "Triceps", "Ön Kol" derse "Biceps/Forearm", "Omuz" derse "Shoulder/Deltoid" olarak eşleştir. Veritabanındaki İngilizce (veya latince) terimleri kullanıcıya açıkla.
        7. **Sağlık Uyarısı (Disclaimer):** Tıbbi tavsiye vermediğini, spora başlamadan önce doktora danışılması gerektiğini nazikçe hatırlat.
        
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
                # 'Kası' gibi genel kelimeleri hariç tutarak kontrol et
                if any(word in response.lower() for word in keywords if len(word) > 3 and word not in ["kası", "genel", "egzersizleri"]):
                    
                    if file_name_clean not in found_files:
                        found_files.append(file_name_clean)
                        
                        file_path = os.path.join("data", file)
                        
                        # Görsel ise göster
                        if file_lower.endswith(('.png', '.jpg', '.jpeg')):
                            st.image(file_path, caption=f"Hedef Bölge: {os.path.splitext(file)[0]}")
                        
                        # PDF ise indirilebilir link sun (Görsel yoksa alternatif kaynak)
                        elif file_lower.endswith('.pdf'):
                            # PDF dosyasını okumak için binary modda aç
                            with open(file_path, "rb") as pdf_file:
                                PDFbyte = pdf_file.read()
                            
                            st.download_button(label=f"📄 '{file}' Dosyasını İncele",
                                                data=PDFbyte,
                                                file_name=file,
                                                mime='application/octet-stream')
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar Footer (Butonlar)
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()

    # Sohbet İndirme
    chat_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
    st.sidebar.download_button(
        label="📥 Programı İndir (TXT)",
        data=chat_text,
        file_name=f"VFit_Program_{name}.txt",
        mime="text/plain"
    )

elif not vectorstore:
    st.warning("Lütfen 'data' klasörüne dosyaları yükleyip sayfayı yenile!")