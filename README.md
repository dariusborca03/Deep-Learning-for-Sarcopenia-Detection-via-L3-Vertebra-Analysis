# Deep Learning for Sarcopenia Detection via L3 Vertebra Analysis

## Descriere
Acest proiect are ca scop **detectarea sarcopeniei** prin analizarea imaginilor CT la nivelul vertebrei **L3**.  
Folosim tehnici de **Deep Learning** pentru:
- Detectarea automată a vertebrei L3 într-un volum CT.
- Segmentarea masei musculare relevante.
- Calculul indicatorilor clinici asociați cu sarcopenia.

Proiectul include scripturi pentru **preprocesarea datelor (DICOM → PNG)**, **detectarea anatomică a L3** și o **interfață grafică (GUI)** pentru vizualizare.

---

## 📂 Structura proiectului
Deep-Learning-for-Sarcopenia-Detection-via-L3-Vertebra-Analysis/
- dicom_to_png_converter.py # Conversie DICOM → PNG
- l3_y3_detector_anatomic.py # Detectare vertebra L3
- futuristic_y3_gui_optimized.py # Interfața grafică
- data/ # Directorul de date (creat de utilizator)
  -> images/ # Aici se vor stoca imaginile PNG convertite
- README.md






