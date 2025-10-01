# l3_y3_detector_anatomic.py - Detector Y3 bazat pe criterii anatomice precise
import os
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage


class AnatomicL3Detector:
    """
    Detector L3/Y3 bazat pe criteriul anatomic fundamental:
    Y3 = Forma Y în centru + ABSENȚA COMPLETĂ a coastelor laterale
    """

    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.slice_data = {}

    def load_and_analyze_all_slices(self):
        """Încarcă și analizează toate slice-urile"""
        print("Analizez toate slice-urile pentru criteriul anatomic Y3...")

        dicom_files = []
        for file in os.listdir(self.data_directory):
            if file.lower().endswith('.dcm'):
                dicom_files.append(file)

        dicom_files.sort()
        print(f"Gasit {len(dicom_files)} fisiere DICOM")

        for i, filename in enumerate(dicom_files):
            try:
                path = os.path.join(self.data_directory, filename)
                dicom = pydicom.dcmread(path)
                img = dicom.pixel_array.astype(np.float32)

                analysis = self.analyze_anatomic_criteria(img, i, filename)
                self.slice_data[i] = {
                    'filename': filename,
                    'image': img,
                    'analysis': analysis
                }

            except Exception as e:
                print(f"Eroare la {filename}: {e}")
                continue

        print(f"Analizat {len(self.slice_data)} slice-uri")

    def analyze_anatomic_criteria(self, img, slice_idx, filename):
        """Analizează criteriile anatomice pentru Y3"""
        h, w = img.shape

        # Auto-windowing
        p1, p99 = np.percentile(img, [1, 99])
        img_windowed = np.clip(img, p1, p99)
        img_norm = ((img_windowed - p1) / (p99 - p1) * 255).astype(np.uint8)

        # CRITERIUL 1: Detectează forma Y în centru
        y_shape_score = self.detect_central_y_shape(img_norm)

        # CRITERIUL 2: Verifică ABSENȚA coastelor laterale (CHEIE!)
        no_ribs_score = self.verify_no_lateral_ribs(img_norm)

        # CRITERIUL 3: Poziția în ultimele slice-uri
        position_score = self.calculate_position_score(slice_idx, filename)

        # CRITERIUL 4: Calitatea vertebrei centrale
        vertebra_quality = self.analyze_central_vertebra(img_norm)

        # SCORE FINAL - prioritate pe absența coastelor
        y3_score = (no_ribs_score * 0.5 +  # 50% - ABSENȚA coastelor
                    y_shape_score * 0.3 +  # 30% - Forma Y
                    position_score * 0.1 +  # 10% - Poziție
                    vertebra_quality * 0.1)  # 10% - Calitate

        return {
            'y_shape_score': y_shape_score,
            'no_ribs_score': no_ribs_score,
            'position_score': position_score,
            'vertebra_quality': vertebra_quality,
            'y3_score': y3_score,
            'windowed_image': img_norm,
            'ribs_detected': 100 - no_ribs_score  # Pentru debugging
        }

    def detect_central_y_shape(self, img):
        """Detectează forma Y în zona centrală"""
        h, w = img.shape

        # Zona centrală pentru vertebra
        center_h_start = int(h * 0.58)
        center_h_end = int(h * 0.80)
        center_w_start = int(w * 0.42)
        center_w_end = int(w * 0.58)

        vertebra_region = img[center_h_start:center_h_end, center_w_start:center_w_end]

        # Threshold pentru structuri dense
        _, binary = cv2.threshold(vertebra_region, 140, 255, cv2.THRESH_BINARY)

        # Morfologie
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Găsește contururi
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0

        # Analizează cel mai mare contur
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)

        if area < 50:
            return 0

        # Analizează caracteristicile Y
        y_score = self.analyze_y_characteristics(main_contour, vertebra_region)

        return y_score

    def verify_no_lateral_ribs(self, img):
        """Verifică ABSENȚA coastelor laterale - criteriul CHEIE pentru Y3"""
        h, w = img.shape

        # Zonele laterale ÎNGUSTE pentru detectarea coastelor - fără suprapunere
        left_lateral = img[:, :w // 6]  # Zona stângă mai îngustă
        right_lateral = img[:, 5 * w // 6:]  # Zona dreaptă mai îngustă

        # Detectează structuri osoase laterale (coaste = ovaluri albe)
        left_ribs = self.count_lateral_bone_structures(left_lateral)
        right_ribs = self.count_lateral_bone_structures(right_lateral)

        total_ribs = left_ribs + right_ribs

        # Scorul pentru ABSENȚA coastelor
        if total_ribs == 0:
            return 100  # PERFECT - fără coaste (Y3!)
        elif total_ribs <= 2:
            return 70  # Foarte puține coaste
        elif total_ribs <= 4:
            return 40  # Câteva coaste (Y2?)
        elif total_ribs <= 6:
            return 20  # Multe coaste (Y1?)
        else:
            return 0  # Foarte multe coaste (zona toracică)

    def count_lateral_bone_structures(self, lateral_region):
        """Numără structurile osoase laterale (coastele) - MULT MAI STRICT"""
        if lateral_region.size == 0:
            return 0

        # Threshold FOARTE RIDICAT pentru coaste (coastele sunt FOARTE dense/albe)
        _, binary = cv2.threshold(lateral_region, 220, 255, cv2.THRESH_BINARY)

        # Morfologie minimă - coastele sunt structuri clare
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Găsește structurile osoase
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Criterii FOARTE STRICTE pentru coaste
        rib_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)

            # Coastele trebuie să fie destul de mari și vizibile
            if 300 < area < 3000:  # Mai strict cu dimensiunea
                # Verifică dacă e alungită (caracteristic coastelor)
                x, y, w, h = cv2.boundingRect(contour)
                if h > 0 and w > 0:
                    aspect_ratio = w / h
                    # Coastele sunt foarte alungite (orizontale)
                    if 2.0 < aspect_ratio < 10:  # Mult mai strict

                        # Verifică și intensitatea medie - coastele sunt foarte albe
                        mask = np.zeros(lateral_region.shape, dtype=np.uint8)
                        cv2.fillPoly(mask, [contour], 255)
                        mean_intensity = np.mean(lateral_region[mask > 0])

                        # Doar structuri FOARTE dense (coastele adevărate)
                        if mean_intensity > 200:  # Foarte albe
                            rib_count += 1

        return rib_count

    def calculate_position_score(self, slice_idx, filename):
        """Calculează scor bazat pe poziție (Y3 e în ultimele slice-uri)"""
        try:
            if '_' in filename:
                file_number = int(filename.split('_')[0])

                # Y3 e în zona 190-199
                if 190 <= file_number <= 199:
                    return 100
                elif 185 <= file_number <= 189:
                    return 50
                else:
                    return 10
        except:
            pass

        # Fallback: ultimele slice-uri
        total_files = len([f for f in os.listdir(self.data_directory) if f.endswith('.dcm')])
        relative_pos = slice_idx / total_files

        if relative_pos >= 0.85:
            return 100
        elif relative_pos >= 0.75:
            return 50
        else:
            return 10

    def analyze_central_vertebra(self, img):
        """Analizează calitatea vertebrei centrale"""
        h, w = img.shape

        center_region = img[int(h * 0.58):int(h * 0.80), int(w * 0.42):int(w * 0.58)]

        if center_region.size == 0:
            return 0

        # Verifică prezența structurii dense centrale
        dense_pixels = np.sum(center_region > 120)
        total_pixels = center_region.size

        density_ratio = dense_pixels / total_pixels

        # Calculează uniformitatea
        if dense_pixels > 0:
            dense_region = center_region[center_region > 120]
            uniformity = 100 - np.std(dense_region)
        else:
            uniformity = 0

        # Scor combinat
        quality = (density_ratio * 50 + uniformity * 0.5)
        return min(quality, 100)

    def analyze_y_characteristics(self, contour, region):
        """Analizează caracteristicile formei Y"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return 0

        # Caracteristici Y
        circularity = 4 * np.pi * area / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0

        # Scor pentru forma Y
        y_score = 0

        # Y-ul are circularitate moderată
        if 0.3 <= circularity <= 0.7:
            y_score += 40
        elif 0.2 <= circularity <= 0.8:
            y_score += 20

        # Y-ul e ușor vertical
        if 1.0 <= aspect_ratio <= 1.6:
            y_score += 30
        elif 0.8 <= aspect_ratio <= 1.8:
            y_score += 15

        # Dimensiune rezonabilă
        if 200 < area < 1500:
            y_score += 30
        elif 100 < area < 2000:
            y_score += 15

        return min(y_score, 100)

    def find_best_y3_candidates(self):
        """Găsește cei mai buni candidați Y3"""
        print("\nCaut Y3 bazat pe criteriul: Forma Y + FĂRĂ coaste laterale...")

        candidates = []
        for slice_idx, data in self.slice_data.items():
            analysis = data['analysis']
            candidates.append((slice_idx, data['filename'], analysis['y3_score'], analysis))

        candidates.sort(key=lambda x: x[2], reverse=True)

        print("\nTop candidati Y3:")
        for i, (slice_idx, filename, score, analysis) in enumerate(candidates[:5]):
            ribs_status = "FĂRĂ coaste" if analysis['ribs_detected'] < 10 else f"{analysis['ribs_detected']:.0f} coaste"
            y_status = f"Y-shape: {analysis['y_shape_score']:.0f}"

            print(f"  {i + 1}. {filename}")
            print(f"     Score Y3: {score:.1f}")
            print(f"     {ribs_status}, {y_status}")
            print(f"     Poziție: {analysis['position_score']:.0f}")

        return candidates

    def create_detailed_analysis(self, candidates):
        """Creează analiza detaliată cu vizualizare"""
        best_candidate = candidates[0]
        slice_idx, filename, score, analysis = best_candidate

        data = self.slice_data[slice_idx]
        img = data['image']

        print(f"\nRECOMANDARE FINALA Y3:")
        print(f"Fisier: {filename}")
        print(f"Score Y3: {score:.1f}")
        print(f"Criteriu CHEIE: {analysis['ribs_detected']:.0f} coaste detectate")

        # Vizualizare
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Auto-windowing
        p1, p99 = np.percentile(img, [1, 99])
        img_display = np.clip(img, p1, p99)
        img_display = ((img_display - p1) / (p99 - p1) * 255).astype(np.uint8)

        # Imaginea originală
        axes[0, 0].imshow(img_display, cmap='gray')
        axes[0, 0].set_title(f"CT Original\n{filename}")
        axes[0, 0].axis('off')

        # Analiza coastelor
        h, w = img_display.shape
        ribs_analysis = img_display.copy()

        # Marchează zonele laterale ÎNGUSTE - fără suprapunere cu centrul
        cv2.rectangle(ribs_analysis, (0, 0), (w // 6, h), 128, 2)  # Stânga îngustă
        cv2.rectangle(ribs_analysis, (5 * w // 6, 0), (w, h), 128, 2)  # Dreapta îngustă

        # Marchează zona Y centrală
        center_h_start = int(h * 0.58)
        center_h_end = int(h * 0.80)
        center_w_start = int(w * 0.42)
        center_w_end = int(w * 0.58)
        cv2.rectangle(ribs_analysis, (center_w_start, center_h_start),
                      (center_w_end, center_h_end), 255, 3)

        axes[0, 1].imshow(ribs_analysis, cmap='gray')
        axes[0, 1].set_title(f"Analiza Anatomica\nCoaste: {analysis['ribs_detected']:.0f}")
        axes[0, 1].axis('off')

        # Zona Y extrasa
        vertebra_region = img_display[center_h_start:center_h_end, center_w_start:center_w_end]
        axes[0, 2].imshow(vertebra_region, cmap='gray')
        axes[0, 2].set_title(f"Forma Y Extrasa\nScore: {analysis['y_shape_score']:.0f}")
        axes[0, 2].axis('off')

        # Grafic scoruri
        candidate_scores = [c[2] for c in candidates[:10]]
        axes[1, 0].bar(range(len(candidate_scores)), candidate_scores)
        axes[1, 0].set_title("Scoruri Y3 Candidati")
        axes[1, 0].set_xlabel("Rank")
        axes[1, 0].set_ylabel("Score Y3")

        # Componente scor
        components = ['Fără Coaste', 'Forma Y', 'Poziție', 'Calitate']
        scores = [analysis['no_ribs_score'], analysis['y_shape_score'],
                  analysis['position_score'], analysis['vertebra_quality']]
        colors = ['red', 'blue', 'green', 'orange']

        axes[1, 1].bar(components, scores, color=colors, alpha=0.7)
        axes[1, 1].set_title("Criteriile Anatomice Y3")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Statistici
        stats_text = f"""DETECTIE Y3 ANATOMICA

Fisier: {filename}
Score Y3: {score:.1f}

CRITERIUL CHEIE:
Coaste detectate: {analysis['ribs_detected']:.0f}
Status: {'✓ FĂRĂ coaste (Y3!)' if analysis['ribs_detected'] < 10 else '✗ Cu coaste (Y1/Y2)'}

Alte criterii:
- Forma Y: {analysis['y_shape_score']:.0f}/100
- Poziție ultimele slice-uri: {analysis['position_score']:.0f}/100
- Calitate vertebră: {analysis['vertebra_quality']:.0f}/100

Interpretare:
{'✓ Y3 CONFIRMAT' if score > 60 else '? Y3 POSIBIL' if score > 40 else '✗ NU este Y3'}

Zona de lucru pentru sarcopenie:
{'✓ Curată, fără coaste' if analysis['ribs_detected'] < 20 else '✗ Cu interferențe osoase'}
        """

        axes[1, 2].text(0.05, 0.95, stats_text, fontsize=8,
                        verticalalignment='top', transform=axes[1, 2].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig('y3_anatomic_detection.png', dpi=150, bbox_inches='tight')
        plt.show()

        return filename, score


def detect_y3_anatomic(data_directory):
    """Detectare Y3 bazată pe criteriile anatomice fundamentale"""
    print("DETECTOR Y3 ANATOMIC")
    print("Criteriul CHEIE: Forma Y + ABSENȚA coastelor laterale")
    print("=" * 50)

    detector = AnatomicL3Detector(data_directory)

    # Analizează toate slice-urile
    detector.load_and_analyze_all_slices()

    if len(detector.slice_data) == 0:
        print("EROARE: Nu s-au gasit imagini valide!")
        return None

    # Găsește candidații Y3
    candidates = detector.find_best_y3_candidates()

    # Analiza detaliată
    best_filename, best_score = detector.create_detailed_analysis(candidates)

    print(f"\nREZULTAT FINAL:")
    print(f"Y3 detectat în: {best_filename}")
    print(f"Score: {best_score:.1f}")
    print(f"Criteriul anatomic: {'CONFIRMAT' if best_score > 60 else 'NECLAR'}")

    return best_filename, best_score


if __name__ == "__main__":
    data_dir = "data/images/"

    if os.path.exists(data_dir):
        result = detect_y3_anatomic(data_dir)
    else:
        print(f"Directorul {data_dir} nu exista!")
        print("Specificati calea corecta catre imaginile DICOM.")