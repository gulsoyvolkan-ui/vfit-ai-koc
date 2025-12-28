import os
import google.generativeai as genai

os.environ["GOOGLE_API_KEY"] = "AIzaSyBEIe2cTwCBMvtmwk15n4DYm0kiDWiXCyw"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("List of available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
