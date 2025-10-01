# futuristic_y3_gui_optimized.py - GUI Optimizat pentru detectarea progresivă Y3
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pydicom
import os
import cv2
from threading import Thread
import time
from datetime import datetime

# Import detector
from l3_y3_detector_anatomic import AnatomicL3Detector

# Set theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class OptimizedY3GUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Y3 VERTEBRA ANALYZER - Progressive Zone Detection")
        # Reducere 30% de la dimensiunea originală (1600x1000 -> 1120x700)
        self.root.geometry("1120x700")

        # Date
        self.ct_directory = "data/images/"
        self.current_slice_idx = 0
        self.dicom_files = []
        self.slice_data = {}
        self.detector = None
        self.y3_detected = False
        self.best_y3_slice = None

        # Zona Y3 progresivă - îmbunătățită
        self.y3_zone_start = None
        self.y3_zone_end = None
        self.y3_zone_slices = []
        self.y3_progression_quality = 0
        self.formation_stages = {}  # Stadiile formării Y3

        # Colors optimizate
        self.colors = {
            'bg_primary': '#0a0a0a',
            'bg_secondary': '#1a1a1a',
            'accent_cyan': '#00ffff',
            'accent_green': '#00ff41',
            'accent_red': '#ff0040',
            'accent_orange': '#ff8c00',
            'accent_purple': '#8a2be2',
            'text_primary': '#ffffff',
            'text_secondary': '#cccccc',
            'zone_highlight': '#004400'
        }

        self.setup_gui()
        self.load_dicom_files()

    def setup_gui(self):
        """Configurează interfața optimizată"""
        # Main container - dimensiuni reduse
        self.main_frame = ctk.CTkFrame(self.root, fg_color=self.colors['bg_primary'])
        self.main_frame.pack(fill='both', expand=True, padx=8, pady=8)

        # Header mai compact
        self.setup_compact_header()

        # Content area
        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color=self.colors['bg_secondary'])
        self.content_frame.pack(fill='both', expand=True, padx=8, pady=8)

        # Paneluri optimizate pentru spațiu redus
        self.setup_compact_left_panel()
        self.setup_compact_center_panel()
        self.setup_y3_zone_panel()
        self.setup_compact_bottom_panel()

    def setup_compact_header(self):
        """Header compact pentru fereastră mai mică"""
        header_frame = ctk.CTkFrame(self.main_frame, height=60, fg_color=self.colors['bg_secondary'])
        header_frame.pack(fill='x', padx=8, pady=(0, 8))
        header_frame.pack_propagate(False)

        # Title mai compact
        title_label = ctk.CTkLabel(
            header_frame,
            text="◆ Y3 PROGRESSIVE ANALYZER ◆",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.colors['accent_cyan']
        )
        title_label.pack(side='left', padx=15, pady=15)

        # Status indicators - layout îmbunătățit
        self.status_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        self.status_frame.pack(side='right', padx=15, pady=15)

        self.zone_status = ctk.CTkLabel(
            self.status_frame,
            text="● Y3 ZONE: READY",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=self.colors['accent_green']
        )
        self.zone_status.pack(anchor='e', pady=2)

        self.formation_status = ctk.CTkLabel(
            self.status_frame,
            text="● FORMATION: READY",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=self.colors['accent_cyan']
        )
        self.formation_status.pack(anchor='e', pady=2)

    def setup_compact_left_panel(self):
        """Panel stânga compact"""
        self.left_panel = ctk.CTkFrame(self.content_frame, width=280, fg_color=self.colors['bg_primary'])
        self.left_panel.pack(side='left', fill='y', padx=(0, 8))
        self.left_panel.pack_propagate(False)

        # Scan Info compact
        info_label = ctk.CTkLabel(
            self.left_panel,
            text="▼ SCAN DATA",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['accent_cyan']
        )
        info_label.pack(pady=(15, 8))

        self.info_display = ctk.CTkTextbox(
            self.left_panel,
            height=80,
            font=ctk.CTkFont(family="Consolas", size=9),
            fg_color=self.colors['bg_secondary']
        )
        self.info_display.pack(fill='x', padx=15, pady=(0, 15))

        # Navigation compact
        nav_label = ctk.CTkLabel(
            self.left_panel,
            text="▼ NAVIGATION",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['accent_cyan']
        )
        nav_label.pack(pady=(8, 8))

        # Slice slider
        self.slice_var = tk.IntVar()
        self.slice_slider = ctk.CTkSlider(
            self.left_panel,
            from_=0,
            to=100,
            variable=self.slice_var,
            command=self.on_slice_change,
            progress_color=self.colors['accent_cyan'],
            button_color=self.colors['accent_green']
        )
        self.slice_slider.pack(fill='x', padx=15, pady=8)

        # Slice counter și zone indicator
        counter_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        counter_frame.pack(pady=5)

        self.slice_counter = ctk.CTkLabel(
            counter_frame,
            text="SLICE: 0 / 0",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.colors['text_primary']
        )
        self.slice_counter.pack()

        self.zone_indicator = ctk.CTkLabel(
            counter_frame,
            text="",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=self.colors['accent_green']
        )
        self.zone_indicator.pack()

        # Navigation buttons compact
        nav_buttons_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        nav_buttons_frame.pack(pady=8)

        self.prev_btn = ctk.CTkButton(
            nav_buttons_frame,
            text="◀ PREV",
            command=self.prev_slice,
            fg_color=self.colors['accent_red'],
            hover_color="#cc0033",
            width=80
        )
        self.prev_btn.pack(side='left', padx=3)

        self.next_btn = ctk.CTkButton(
            nav_buttons_frame,
            text="NEXT ▶",
            command=self.next_slice,
            fg_color=self.colors['accent_green'],
            hover_color="#00cc34",
            width=80
        )
        self.next_btn.pack(side='right', padx=3)

        # Y3 Zone Detection
        detect_label = ctk.CTkLabel(
            self.left_panel,
            text="▼ Y3 ZONE DETECTION",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['accent_cyan']
        )
        detect_label.pack(pady=(20, 8))

        self.detect_btn = ctk.CTkButton(
            self.left_panel,
            text="🔍 FIND Y3 ZONE",
            command=self.start_y3_zone_detection,
            fg_color=self.colors['accent_orange'],
            hover_color="#cc6600",
            height=40,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.detect_btn.pack(fill='x', padx=15, pady=8)

        # Quick actions
        actions_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        actions_frame.pack(fill='x', padx=15, pady=5)

        self.goto_zone_btn = ctk.CTkButton(
            actions_frame,
            text="➤ ZONE",
            command=self.goto_y3_zone,
            fg_color=self.colors['accent_green'],
            width=70,
            state="disabled"
        )
        self.goto_zone_btn.pack(side='left', padx=2)

        self.goto_best_btn = ctk.CTkButton(
            actions_frame,
            text="★ BEST",
            command=self.goto_best_y3,
            fg_color=self.colors['accent_purple'],
            width=70,
            state="disabled"
        )
        self.goto_best_btn.pack(side='right', padx=2)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            self.left_panel,
            progress_color=self.colors['accent_cyan']
        )
        self.progress_bar.pack(fill='x', padx=15, pady=8)
        self.progress_bar.set(0)

    def setup_compact_center_panel(self):
        """Panel central compact"""
        self.center_panel = ctk.CTkFrame(self.content_frame, fg_color=self.colors['bg_primary'])
        self.center_panel.pack(side='left', fill='both', expand=True)

        # Matplotlib figure mai mică
        self.fig = Figure(figsize=(8, 6), facecolor=self.colors['bg_primary'])
        self.canvas = FigureCanvasTkAgg(self.fig, self.center_panel)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=8)

        # Controls compacte
        controls_frame = ctk.CTkFrame(self.center_panel, height=40, fg_color=self.colors['bg_secondary'])
        controls_frame.pack(fill='x', padx=8, pady=(0, 8))
        controls_frame.pack_propagate(False)

        # View controls
        ctk.CTkLabel(controls_frame, text="VIEW:", text_color=self.colors['text_secondary']).pack(side='left', padx=8,
                                                                                                  pady=8)

        self.zoom_in_btn = ctk.CTkButton(controls_frame, text="🔍+", width=40, command=self.zoom_in)
        self.zoom_in_btn.pack(side='left', padx=2, pady=5)

        self.zoom_out_btn = ctk.CTkButton(controls_frame, text="🔍-", width=40, command=self.zoom_out)
        self.zoom_out_btn.pack(side='left', padx=2, pady=5)

        self.reset_view_btn = ctk.CTkButton(controls_frame, text="↺", width=40, command=self.reset_view)
        self.reset_view_btn.pack(side='left', padx=2, pady=5)

    def setup_y3_zone_panel(self):
        """Panel dedicat pentru zona Y3 - cea mai importantă îmbunătățire"""
        self.right_panel = ctk.CTkFrame(self.content_frame, width=300, fg_color=self.colors['bg_primary'])
        self.right_panel.pack(side='right', fill='y', padx=(8, 0))
        self.right_panel.pack_propagate(False)

        # Y3 Zone Status
        zone_label = ctk.CTkLabel(
            self.right_panel,
            text="▼ Y3 PROGRESSIVE ZONE",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['accent_cyan']
        )
        zone_label.pack(pady=(15, 8))

        # Zone status display îmbunătățit
        self.zone_status_frame = ctk.CTkFrame(self.right_panel, fg_color=self.colors['bg_secondary'])
        self.zone_status_frame.pack(fill='x', padx=15, pady=8)

        self.zone_detection_label = ctk.CTkLabel(
            self.zone_status_frame,
            text="● ZONE STATUS: READY TO SCAN",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self.colors['text_secondary']
        )
        self.zone_detection_label.pack(pady=8)

        self.formation_quality_label = ctk.CTkLabel(
            self.zone_status_frame,
            text="● FORMATION: NOT ANALYZED",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self.colors['text_secondary']
        )
        self.formation_quality_label.pack(pady=(0, 8))

        # Progressive Formation Analysis
        formation_label = ctk.CTkLabel(
            self.right_panel,
            text="▼ FORMATION ANALYSIS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.colors['accent_green']
        )
        formation_label.pack(pady=(15, 8))

        self.formation_display = ctk.CTkTextbox(
            self.right_panel,
            height=180,
            font=ctk.CTkFont(family="Consolas", size=9),
            fg_color=self.colors['bg_secondary']
        )
        self.formation_display.pack(fill='x', padx=15, pady=(0, 15))

        # Zone Recommendations
        recommendations_label = ctk.CTkLabel(
            self.right_panel,
            text="▼ RECOMMENDATIONS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.colors['accent_purple']
        )
        recommendations_label.pack(pady=(8, 8))

        self.recommendations_display = ctk.CTkTextbox(
            self.right_panel,
            height=110,  # Redus cu 10px pentru a preveni overflow
            font=ctk.CTkFont(family="Consolas", size=9),
            fg_color=self.colors['bg_secondary'],
            wrap="word"  # Adaugă word wrapping
        )
        self.recommendations_display.pack(fill='x', padx=15, pady=(0, 12))

        # Quick Export
        export_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        export_frame.pack(fill='x', padx=15, pady=8)

        self.save_zone_btn = ctk.CTkButton(
            export_frame,
            text="💾 SAVE ZONE",
            command=self.save_zone_result,
            fg_color=self.colors['accent_green'],
            width=130
        )
        self.save_zone_btn.pack(side='left', padx=2)

        self.export_img_btn = ctk.CTkButton(
            export_frame,
            text="📸 EXPORT",
            command=self.export_y3_image,
            fg_color=self.colors['accent_orange'],
            width=130
        )
        self.export_img_btn.pack(side='right', padx=2)

    def setup_compact_bottom_panel(self):
        """Panel jos compact"""
        self.bottom_panel = ctk.CTkFrame(self.main_frame, height=45, fg_color=self.colors['bg_secondary'])
        self.bottom_panel.pack(fill='x', padx=8, pady=(8, 0))
        self.bottom_panel.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            self.bottom_panel,
            text="◆ Y3 PROGRESSIVE ANALYZER READY - Load DICOM files to find Y3 zone",
            font=ctk.CTkFont(size=11),
            text_color=self.colors['accent_green']
        )
        self.status_label.pack(side='left', padx=15, pady=12)

        # Time display compact
        self.time_label = ctk.CTkLabel(
            self.bottom_panel,
            text="",
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color=self.colors['text_secondary']
        )
        self.time_label.pack(side='right', padx=15, pady=12)

        self.update_time()

    def update_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.configure(text=f"⧖ {current_time}")
        self.root.after(1000, self.update_time)

    def start_y3_zone_detection(self):
        """Începe detectarea zonei Y3 cu analiză progresivă îmbunătățită"""
        self.detect_btn.configure(state="disabled", text="🔍 ANALYZING ZONE...")
        self.zone_detection_label.configure(text="● ZONE STATUS: SCANNING SLICES...",
                                            text_color=self.colors['accent_orange'])
        self.formation_quality_label.configure(text="● FORMATION: ANALYZING...",
                                               text_color=self.colors['accent_orange'])
        self.progress_bar.set(0)

        # Start detection in thread
        thread = Thread(target=self.run_y3_zone_detection)
        thread.daemon = True
        thread.start()

    def run_y3_zone_detection(self):
        """Rulează detectarea zonei Y3 cu focus pe progresie"""
        try:
            self.update_status("◆ INITIALIZING Y3 ZONE DETECTOR...")

            # Initialize detector
            self.detector = AnatomicL3Detector(self.ct_directory)

            self.update_status("◆ SCANNING FOR Y3 PROGRESSIVE ZONE...")
            self.detector.load_and_analyze_all_slices()

            self.update_status("◆ ANALYZING Y3 FORMATION PROGRESSION...")
            candidates = self.detector.find_best_y3_candidates()

            # Update UI in main thread
            self.root.after(0, self.y3_zone_detection_complete, candidates)

        except Exception as e:
            self.root.after(0, self.y3_zone_detection_error, str(e))

    def y3_zone_detection_complete(self, candidates):
        """Y3 Zone detection completă cu analiză progresivă îmbunătățită"""
        try:
            total_slices = len(self.dicom_files)

            # Calculează zona Y3 progresivă (ultimele 15% pentru mai multă precizie)
            self.y3_zone_start = int(total_slices * 0.85)  # Start mai devreme
            self.y3_zone_end = total_slices - 1
            zone_size = self.y3_zone_end - self.y3_zone_start + 1

            # Filtrează candidații din zona Y3
            y3_zone_candidates = []
            for slice_idx, filename, score, analysis in candidates:
                if slice_idx >= self.y3_zone_start:
                    y3_zone_candidates.append((slice_idx, filename, score, analysis))

            # Analizează progresivitatea formării
            if y3_zone_candidates:
                # Sortează după index pentru analiză progresivă
                sorted_zone = sorted(y3_zone_candidates, key=lambda x: x[0])

                # Analizează calitatea progresiei
                self.analyze_formation_progression(sorted_zone)

                # Găsește cel mai bun candidat
                best_candidate = max(y3_zone_candidates, key=lambda x: x[2])
                slice_idx, filename, score, analysis = best_candidate

                self.best_y3_slice = filename
                self.y3_detected = True
                self.y3_zone_slices = y3_zone_candidates

                # Update status cu informații despre zonă
                status_color = self.colors['accent_green'] if score > 65 else self.colors['accent_orange']

                zone_status = f"● Y3 ZONE DETECTED: {len(y3_zone_candidates)}/{zone_size} valid slices"
                formation_status = f"● FORMATION: {self.get_formation_quality_text()}"

                self.zone_detection_label.configure(text=zone_status, text_color=status_color)
                self.formation_quality_label.configure(text=formation_status, text_color=status_color)

                # Update formation analysis display
                self.update_formation_analysis_display(sorted_zone, best_candidate)

                # Update recommendations
                self.update_recommendations_display(best_candidate, len(y3_zone_candidates), zone_size)

                # Enable navigation buttons
                self.goto_zone_btn.configure(state="normal")
                self.goto_best_btn.configure(state="normal")

                self.update_status(
                    f"◆ Y3 ZONE COMPLETE - {len(y3_zone_candidates)} slices | Best: {filename} (Score: {score:.1f})")

            else:
                # Zona Y3 nu găsită în locația așteptată
                self.zone_detection_label.configure(
                    text="● Y3 ZONE: NOT IN EXPECTED LOCATION",
                    text_color=self.colors['accent_red']
                )
                self.formation_quality_label.configure(
                    text="● FORMATION: REQUIRES MANUAL REVIEW",
                    text_color=self.colors['accent_red']
                )

                # Afișează alternativa
                if candidates:
                    alt_slice_idx, alt_filename, alt_score, alt_analysis = candidates[0]
                    self.formation_display.delete("1.0", "end")
                    self.formation_display.insert("1.0", f"""Y3 ZONE ANALYSIS:

⚠ ZONE NOT IN EXPECTED RANGE
Expected: Slices {self.y3_zone_start + 1}-{self.y3_zone_end + 1}

ALTERNATIVE FOUND:
Best: {alt_filename} (Slice {alt_slice_idx + 1})
Score: {alt_score:.1f}/100

This may indicate:
- Different patient anatomy
- Non-standard scan coverage
- Requires manual verification

MANUAL REVIEW RECOMMENDED""")

            # Update progress
            self.progress_bar.set(1.0)

            # Reset button
            self.detect_btn.configure(state="normal", text="🔍 FIND Y3 ZONE")

            # Load current slice to update analysis
            self.load_current_slice()

        except Exception as e:
            self.y3_zone_detection_error(str(e))

    def analyze_formation_progression(self, sorted_zone_candidates):
        """Analizează progresivitatea formării Y3"""
        if len(sorted_zone_candidates) < 2:
            self.y3_progression_quality = 0
            return

        # Calculează trend-ul scorurilor
        scores = [candidate[2] for candidate in sorted_zone_candidates]
        positions = list(range(len(scores)))

        # Calculează progresivitatea (trend crescător = bun)
        if len(scores) >= 3:
            # Verifică dacă scorurile cresc în general
            increases = sum(1 for i in range(1, len(scores)) if scores[i] > scores[i - 1])
            total_comparisons = len(scores) - 1
            progression_ratio = increases / total_comparisons if total_comparisons > 0 else 0

            # Calculează stabilitatea (variația nu e prea mare)
            score_variance = np.var(scores) if len(scores) > 1 else 0
            stability_factor = max(0, 1 - score_variance / 1000)  # Normalizat

            # Calitatea finală a progresiei
            self.y3_progression_quality = (progression_ratio * 0.7 + stability_factor * 0.3) * 100
        else:
            self.y3_progression_quality = 50  # Progresie limitată

        # Salvează stadiile formării pentru afișare
        self.formation_stages = {
            'early': sorted_zone_candidates[0] if sorted_zone_candidates else None,
            'middle': sorted_zone_candidates[len(sorted_zone_candidates) // 2] if len(
                sorted_zone_candidates) > 2 else None,
            'optimal': sorted_zone_candidates[-1] if sorted_zone_candidates else None
        }

    def get_formation_quality_text(self):
        """Returnează textul pentru calitatea formării"""
        if self.y3_progression_quality >= 80:
            return "EXCELLENT PROGRESSION"
        elif self.y3_progression_quality >= 60:
            return "GOOD PROGRESSION"
        elif self.y3_progression_quality >= 40:
            return "MODERATE PROGRESSION"
        else:
            return "LIMITED PROGRESSION"

    def update_formation_analysis_display(self, sorted_zone, best_candidate):
        """Update formation analysis cu informații detaliate"""
        slice_idx, filename, score, analysis = best_candidate

        formation_text = f"""Y3 PROGRESSIVE FORMATION ANALYSIS:

ZONE COVERAGE:
• Total slices in zone: {len(sorted_zone)}
• Formation quality: {self.y3_progression_quality:.1f}%
• Progression status: {self.get_formation_quality_text()}

FORMATION STAGES:"""

        if self.formation_stages.get('early'):
            early = self.formation_stages['early']
            formation_text += f"""
▸ EARLY: Slice {early[0] + 1} (Score: {early[2]:.1f})"""

        if self.formation_stages.get('middle'):
            middle = self.formation_stages['middle']
            formation_text += f"""
▸ DEVELOPING: Slice {middle[0] + 1} (Score: {middle[2]:.1f})"""

        if self.formation_stages.get('optimal'):
            optimal = self.formation_stages['optimal']
            formation_text += f"""
▸ OPTIMAL: Slice {optimal[0] + 1} (Score: {optimal[2]:.1f})"""

        formation_text += f"""

BEST SLICE ANALYSIS:
• File: {filename}
• Y3 Score: {score:.1f}/100
• Y-Shape: {analysis['y_shape_score']:.1f}
• No Ribs: {analysis['no_ribs_score']:.1f}
• Position: {analysis['position_score']:.1f}
• Quality: {analysis['vertebra_quality']:.1f}

ANATOMIC VALIDATION:
{'✓ Clean Y3 formation' if analysis['ribs_detected'] < 15 else '⚠ Some interference detected'}
{'✓ Optimal vertebra quality' if analysis['vertebra_quality'] > 70 else '? Moderate quality'}"""

        self.formation_display.delete("1.0", "end")
        self.formation_display.insert("1.0", formation_text)

    def update_recommendations_display(self, best_candidate, valid_slices, total_zone_slices):
        """Update recommendations cu ghiduri specifice - versiune compactă"""
        slice_idx, filename, score, analysis = best_candidate

        recommendations_text = f"""CLINICAL RECOMMENDATIONS:

PRIMARY RECOMMENDATION:
★ Use Slice {slice_idx + 1} ({filename[:20]}...)
★ Confidence: {score:.1f}/100

ZONE ASSESSMENT:
• Valid slices: {valid_slices}/{total_zone_slices}
• Coverage: {(valid_slices / total_zone_slices) * 100:.1f}%
• Formation: {self.get_formation_quality_text()[:20]}

FOR SARCOPENIA ANALYSIS:
"""

        if score >= 70:
            recommendations_text += """✓ EXCELLENT - Proceed with confidence
✓ High anatomic accuracy expected
✓ Suitable for automated analysis"""
        elif score >= 55:
            recommendations_text += """✓ GOOD - Suitable for analysis  
? Consider manual verification
✓ Good anatomic landmarks"""
        else:
            recommendations_text += """? MODERATE - Manual review advised
⚠ Verify anatomic landmarks
? Consider alternative slices"""

        recommendations_text += f"""

ALTERNATIVES:
{f'• {valid_slices - 1} other slices available' if valid_slices > 1 else '• Limited alternatives'}

QUALITY INDICATORS:
{f'✓ Y-shape visible' if analysis['y_shape_score'] > 60 else '? Y-shape moderate'}
{f'✓ Rib-free zone' if analysis['no_ribs_score'] > 70 else '⚠ Some interference'}
{f'✓ Good position' if analysis['position_score'] > 70 else '? Position OK'}"""

        self.recommendations_display.delete("1.0", "end")
        self.recommendations_display.insert("1.0", recommendations_text)

    def y3_zone_detection_error(self, error_msg):
        """Eroare în detectarea zonei Y3"""
        self.detect_btn.configure(state="normal", text="🔍 FIND Y3 ZONE")
        self.zone_detection_label.configure(text="● ZONE STATUS: ERROR OCCURRED",
                                            text_color=self.colors['accent_red'])
        self.formation_quality_label.configure(text="● FORMATION: ANALYSIS FAILED",
                                               text_color=self.colors['accent_red'])
        self.progress_bar.set(0)
        self.update_status(f"❌ Y3 ZONE DETECTION ERROR: {error_msg}")

    def goto_y3_zone(self):
        """Navighează la începutul zonei Y3"""
        if self.y3_zone_start is not None:
            self.current_slice_idx = self.y3_zone_start
            self.slice_var.set(self.y3_zone_start)
            self.load_current_slice()
            self.update_status(f"◆ NAVIGATED TO Y3 ZONE START: Slice {self.y3_zone_start + 1}")

    def goto_best_y3(self):
        """Navighează la cel mai bun Y3 detectat"""
        if not self.best_y3_slice:
            return

        try:
            y3_index = self.dicom_files.index(self.best_y3_slice)
            self.current_slice_idx = y3_index
            self.slice_var.set(y3_index)
            self.load_current_slice()
            self.update_status(f"◆ NAVIGATED TO BEST Y3: {self.best_y3_slice}")
        except ValueError:
            self.update_status("❌ ERROR: Best Y3 slice not found")

    def load_dicom_files(self):
        """Încarcă fișierele DICOM"""
        try:
            if not os.path.exists(self.ct_directory):
                self.update_status("❌ ERROR: data/images/ directory not found!")
                return

            self.dicom_files = [f for f in os.listdir(self.ct_directory) if f.endswith('.dcm')]
            self.dicom_files.sort()

            if len(self.dicom_files) == 0:
                self.update_status("❌ ERROR: No DICOM files found!")
                return

            # Update slider
            self.slice_slider.configure(to=len(self.dicom_files) - 1)

            # Load first slice
            self.load_current_slice()

            self.update_status(f"◆ LOADED {len(self.dicom_files)} DICOM files - Ready for Y3 zone analysis")

        except Exception as e:
            self.update_status(f"❌ ERROR loading files: {e}")

    def load_current_slice(self):
        """Încarcă slice-ul curent cu indicatori de zonă"""
        if not self.dicom_files:
            return

        try:
            filename = self.dicom_files[self.current_slice_idx]
            filepath = os.path.join(self.ct_directory, filename)

            # Load DICOM
            dicom = pydicom.dcmread(filepath)
            img = dicom.pixel_array.astype(np.float32)

            # Update displays
            self.update_slice_display_with_zone_info(img, filename)
            self.update_info_display(filename, img)
            self.update_slice_counter_with_zone()

            # Analyze current slice if detector is ready
            if self.detector and hasattr(self.detector, 'slice_data'):
                self.analyze_current_slice_for_zone(img, filename)

        except Exception as e:
            self.update_status(f"❌ ERROR loading slice: {e}")

    def update_slice_display_with_zone_info(self, img, filename):
        """Actualizează afișajul slice-ului cu informații despre zonă"""
        self.fig.clear()

        # Auto-windowing
        p1, p99 = np.percentile(img, [1, 99])
        img_display = np.clip(img, p1, p99)
        img_display = ((img_display - p1) / (p99 - p1) * 255).astype(np.uint8)

        ax = self.fig.add_subplot(111)
        ax.imshow(img_display, cmap='gray')

        # Determină tipul slice-ului
        slice_type = self.get_slice_type_info(filename)
        title_color = slice_type['color']
        title_text = slice_type['text']

        ax.set_title(f"{title_text} {filename}",
                     color=title_color, fontsize=11, fontweight='bold')
        ax.axis('off')

        # Add zone overlay dacă este în zona Y3
        if slice_type['in_zone'] and slice_type['is_best']:
            self.add_y3_detection_overlay(ax, img_display.shape)
        elif slice_type['in_zone']:
            self.add_zone_highlight_overlay(ax, img_display.shape)

        self.fig.patch.set_facecolor(self.colors['bg_primary'])
        ax.set_facecolor(self.colors['bg_primary'])

        self.canvas.draw()

    def get_slice_type_info(self, filename):
        """Determină tipul și statusul slice-ului"""
        info = {
            'color': '#ffffff',
            'text': '',
            'in_zone': False,
            'is_best': False
        }

        # Check dacă este cel mai bun Y3
        if self.best_y3_slice and filename == self.best_y3_slice:
            info.update({
                'color': self.colors['accent_green'],
                'text': '★ BEST Y3 ★',
                'in_zone': True,
                'is_best': True
            })
        # Check dacă este în zona Y3
        elif (self.y3_zone_start is not None and
              self.current_slice_idx >= self.y3_zone_start and
              self.current_slice_idx <= self.y3_zone_end):
            info.update({
                'color': self.colors['accent_orange'],
                'text': '◆ Y3 ZONE ◆',
                'in_zone': True,
                'is_best': False
            })
        # Check dacă este candidat Y3
        elif any(candidate[1] == filename for candidate in self.y3_zone_slices):
            info.update({
                'color': self.colors['accent_cyan'],
                'text': '◇ Y3 CANDIDATE ◇',
                'in_zone': True,
                'is_best': False
            })
        else:
            info.update({
                'color': '#ffffff',
                'text': '◦',
                'in_zone': False,
                'is_best': False
            })

        return info

    def add_y3_detection_overlay(self, ax, img_shape):
        """Adaugă overlay pentru Y3 detectat"""
        h, w = img_shape
        center_h_start = int(h * 0.58)
        center_h_end = int(h * 0.80)
        center_w_start = int(w * 0.42)
        center_w_end = int(w * 0.58)

        # Draw detection box
        from matplotlib.patches import Rectangle
        rect = Rectangle((center_w_start, center_h_start),
                         center_w_end - center_w_start,
                         center_h_end - center_h_start,
                         fill=False, edgecolor=self.colors['accent_green'],
                         linewidth=3, linestyle='--')
        ax.add_patch(rect)

        # Add center point
        center_x = (center_w_start + center_w_end) // 2
        center_y = (center_h_start + center_h_end) // 2
        ax.plot(center_x, center_y, 'o', color=self.colors['accent_green'], markersize=8)
        ax.text(center_x + 20, center_y, 'Y3', color=self.colors['accent_green'],
                fontsize=12, fontweight='bold')

    def add_zone_highlight_overlay(self, ax, img_shape):
        """Adaugă highlight subtil pentru zona Y3"""
        h, w = img_shape
        # Highlight zona centrală unde se formează Y3
        center_h_start = int(h * 0.55)
        center_h_end = int(h * 0.85)
        center_w_start = int(w * 0.35)
        center_w_end = int(w * 0.65)

        from matplotlib.patches import Rectangle
        rect = Rectangle((center_w_start, center_h_start),
                         center_w_end - center_w_start,
                         center_h_end - center_h_start,
                         fill=False, edgecolor=self.colors['accent_orange'],
                         linewidth=1, linestyle=':')
        ax.add_patch(rect)

    def update_slice_counter_with_zone(self):
        """Actualizează contorul cu informații despre zonă"""
        base_text = f"SLICE: {self.current_slice_idx + 1} / {len(self.dicom_files)}"
        self.slice_counter.configure(text=base_text)

        # Zone indicator
        zone_text = ""
        if (self.y3_zone_start is not None and
                self.current_slice_idx >= self.y3_zone_start and
                self.current_slice_idx <= self.y3_zone_end):
            zone_position = self.current_slice_idx - self.y3_zone_start + 1
            zone_total = self.y3_zone_end - self.y3_zone_start + 1
            zone_text = f"Y3 ZONE: {zone_position}/{zone_total}"

        self.zone_indicator.configure(text=zone_text)

    def update_info_display(self, filename, img):
        """Actualizează afișajul informațiilor cu detalii zone"""
        zone_info = ""
        if (self.y3_zone_start is not None and
                self.current_slice_idx >= self.y3_zone_start and
                self.current_slice_idx <= self.y3_zone_end):
            zone_info = f"\nZONE: Y3 Progressive Zone"

        info_text = f"""FILE: {filename}
SIZE: {img.shape[0]} x {img.shape[1]}
RANGE: {np.min(img):.0f} - {np.max(img):.0f}
MEAN: {np.mean(img):.1f}{zone_info}"""

        self.info_display.delete("1.0", "end")
        self.info_display.insert("1.0", info_text)

    def analyze_current_slice_for_zone(self, img, filename):
        """Analizează slice-ul curent în contextul zonei Y3"""
        if not self.detector:
            return

        try:
            # Quick analysis of current slice
            analysis = self.detector.analyze_anatomic_criteria(img, self.current_slice_idx, filename)

            # Determine zone context
            zone_context = "OUTSIDE Y3 ZONE"
            if (self.y3_zone_start is not None and
                    self.current_slice_idx >= self.y3_zone_start and
                    self.current_slice_idx <= self.y3_zone_end):
                zone_context = "IN Y3 ZONE"

            # Check if this is a known candidate
            is_candidate = any(candidate[1] == filename for candidate in self.y3_zone_slices)
            candidate_status = "VALIDATED CANDIDATE" if is_candidate else "NOT CANDIDATE"

            analysis_text = f"""CURRENT SLICE ANALYSIS:

ZONE STATUS: {zone_context}
CANDIDATE: {candidate_status}

Y3 SCORE: {analysis['y3_score']:.1f}/100
├─ Y-Shape: {analysis['y_shape_score']:.1f}
├─ No Ribs: {analysis['no_ribs_score']:.1f}
├─ Position: {analysis['position_score']:.1f}
└─ Quality: {analysis['vertebra_quality']:.1f}

FORMATION ASSESSMENT:
{self.get_formation_assessment(analysis['y3_score'])}

RIBS: {analysis['ribs_detected']:.0f} detected
STATUS: {self.get_slice_status(analysis['y3_score'], zone_context)}"""

        except Exception as e:
            analysis_text = f"SLICE ANALYSIS ERROR: {e}"

        # Update doar dacă formation display nu e folosit pentru zone info
        if not self.y3_detected:
            self.formation_display.delete("1.0", "end")
            self.formation_display.insert("1.0", analysis_text)

    def get_formation_assessment(self, score):
        """Returnează assessment-ul formării"""
        if score >= 70:
            return "✓ EXCELLENT Y3 formation"
        elif score >= 55:
            return "✓ GOOD Y3 formation"
        elif score >= 40:
            return "? MODERATE formation"
        else:
            return "✗ POOR formation"

    def get_slice_status(self, score, zone_context):
        """Returnează statusul slice-ului"""
        if zone_context == "IN Y3 ZONE":
            if score > 60:
                return "★ EXCELLENT Y3"
            elif score > 40:
                return "✓ GOOD Y3"
            else:
                return "? WEAK Y3"
        else:
            if score > 60:
                return "! UNEXPECTED Y3"
            elif score > 40:
                return "? POSSIBLE Y3"
            else:
                return "✗ NOT Y3"

    def on_slice_change(self, value):
        """Handler pentru schimbarea slice-ului"""
        self.current_slice_idx = int(value)
        self.load_current_slice()

    def prev_slice(self):
        """Slice anterior"""
        if self.current_slice_idx > 0:
            self.current_slice_idx -= 1
            self.slice_var.set(self.current_slice_idx)
            self.load_current_slice()

    def next_slice(self):
        """Slice următor"""
        if self.current_slice_idx < len(self.dicom_files) - 1:
            self.current_slice_idx += 1
            self.slice_var.set(self.current_slice_idx)
            self.load_current_slice()

    def zoom_in(self):
        """Zoom in placeholder"""
        self.update_status("◆ ZOOM IN")

    def zoom_out(self):
        """Zoom out placeholder"""
        self.update_status("◆ ZOOM OUT")

    def reset_view(self):
        """Reset view"""
        self.load_current_slice()
        self.update_status("◆ VIEW RESET")

    def save_zone_result(self):
        """Salvează rezultatul zonei Y3"""
        if not self.y3_detected:
            messagebox.showwarning("Warning", "No Y3 zone detected yet!")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Y3_zone_analysis_{timestamp}.txt"

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"""Y3 PROGRESSIVE ZONE ANALYSIS REPORT
==========================================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Best Y3 Slice: {self.best_y3_slice}
Formation Quality: {self.y3_progression_quality:.1f}%

ZONE INFORMATION:
{self.formation_display.get("1.0", "end")}

CLINICAL RECOMMENDATIONS:
{self.recommendations_display.get("1.0", "end")}

Generated by Y3 Progressive Analyzer
""")

            messagebox.showinfo("Success", f"Zone analysis saved: {filename}")
            self.update_status(f"◆ ZONE ANALYSIS SAVED: {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")

    def export_y3_image(self):
        """Exportă imaginea Y3 cu overlay"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Y3_zone_slice_{timestamp}.png"

            self.fig.savefig(filename, dpi=300, bbox_inches='tight',
                             facecolor=self.colors['bg_primary'], edgecolor='none')

            messagebox.showinfo("Success", f"Y3 image exported: {filename}")
            self.update_status(f"◆ Y3 IMAGE EXPORTED: {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")

    def update_status(self, message):
        """Actualizează status-ul"""
        self.status_label.configure(text=message)
        self.root.update_idletasks()

    def run(self):
        """Pornește aplicația"""
        self.root.mainloop()


if __name__ == "__main__":
    app = OptimizedY3GUI()
    app.run()