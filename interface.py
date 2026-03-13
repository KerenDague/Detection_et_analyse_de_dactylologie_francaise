"""
Interface PySide6
"""

import sys
import cv2
import os
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QDialog, QVBoxLayout,
    QGridLayout, QFrame, QScrollArea, QProgressBar, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QPropertyAnimation, QEasingCurve, QSize
from PySide6.QtGui import (
    QFont, QPixmap, QImage, QColor, QPalette, QLinearGradient,
    QPainter, QBrush, QPen, QIcon, QFontDatabase
)
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import QUrl

#------------------------------------------------------------
# Style
COLORS = {
    "bg_dark":       "#EEE8DC",
    "bg_card":       "#E4DED0",
    "bg_hover":      "#D6CCBC",
    "accent":        "#8B2240",
    "accent2":       "#263660",
    "accent3":       "#7888B8",
    "text_primary":  "#432718",
    "text_secondary":"#907060",
    "border":        "#C6B8A2",
    "success":       "#607040",
    "warning":       "#A07820",
}

STYLESHEET = f"""
QMainWindow, QWidget#main_container {{
    background-color: {COLORS['bg_dark']};
}}
QWidget {{
    background-color: transparent;
    color: {COLORS['text_primary']};
    font-family: 'Segoe UI', sans-serif;
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background: {COLORS['bg_dark']};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['border']};
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['accent']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
"""

#------------------------------------------------------------
# Pour la caméra

class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)
    error_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.cap = None

    def run(self):
        self._running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.error_signal.emit("Impossible d'ouvrir la caméra.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.frame_ready.emit(frame)
            self.msleep(30)
        if self.cap:
            self.cap.release()

    def stop(self):
        self._running = False
        self.wait()

#------------------------------------------------------------
# Card
class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 4px;
            }}
        """)

#------------------------------------------------------------
# Interface Principal

class MainInterface(QWidget):
    def __init__(self):
        super().__init__()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20,20,20,20)
        main_layout.setSpacing(20)

        # ----- TITRE GLOBAL -----
        title = QLabel("Détection et analyse de dactylologie française")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        title.setStyleSheet(f"color: {COLORS['accent']};")

        main_layout.addWidget(title)

        # ----- BARRE DE NAVIGATION -----
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(12)

        self.btn_accueil = QPushButton("Accueil")
        self.btn_donnees = QPushButton("Données")
        self.btn_scripts = QPushButton("Scripts")
        self.btn_demo = QPushButton("Démo")

        buttons = [
            self.btn_accueil,
            self.btn_donnees,
            self.btn_scripts,
            self.btn_demo
        ]

        for b in buttons:
            b.setFixedHeight(36)
            b.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['bg_card']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 8px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['bg_hover']};
                }}
            """)
            nav_layout.addWidget(b)

        nav_layout.addStretch()
        main_layout.addLayout(nav_layout)

        # ----- STACKED WIDGET -----
        self.pages = QStackedWidget()

        self.page_accueil = PageAccueil
        self.page_donnees = PageDonnées
        self.page_scripts = PageScripts
        self.page_demo = PageDemo()

        self.pages.addWidget(self.page_accueil)
        self.pages.addWidget(self.page_donnees)
        self.pages.addWidget(self.page_scripts)
        self.pages.addWidget(self.page_demo)

        main_layout.addWidget(self.pages)

        # ----- CONNEXIONS -----
        self.btn_accueil.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        self.btn_donnees.clicked.connect(lambda: self.pages.setCurrentIndex(1))
        self.btn_scripts.clicked.connect(lambda: self.pages.setCurrentIndex(2))
        self.btn_demo.clicked.connect(lambda: self.pages.setCurrentIndex(3))


#------------------------------------------------------------
# Barre de navigation
class NavigationBar(QWidget):
    page_changed = Signal(int)

    def __init__(self):
        super().__init__()

        layout = QHBoxLayout(self)
        layout.setSpacing(12)

        self.buttons = []

        names = ["Accueil", "Données", "Scripts", "Démo"]

        for i, name in enumerate(names):
            btn = QPushButton(name)
            btn.setFixedHeight(36)

            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['bg_card']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 8px;
                    padding: 6px 16px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['bg_hover']};
                }}
            """)

            btn.clicked.connect(lambda _, idx=i: self.page_changed.emit(idx))

            layout.addWidget(btn)
            self.buttons.append(btn)

        layout.addStretch()



#------------------------------------------------------------
# Page Démonstration
class PageDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.camera_active = False
        self._setup_ui()

    def _setup_ui(self):
        main_lay = QHBoxLayout(self)
        main_lay.setContentsMargins(32, 32, 32, 32)
        main_lay.setSpacing(24)

        # Gauche :caméra 
        left = QVBoxLayout()
        left.setSpacing(16)

        cam_title = QLabel("Flux Caméra en Direct")
        cam_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        cam_title.setStyleSheet(f"color: {COLORS['text_primary']};")
        left.addWidget(cam_title)

        self.cam_label = QLabel()
        self.cam_label.setFixedSize(480, 360)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet(f"""
            background-color: {COLORS['bg_card']};
            border: 2px solid {COLORS['border']};
            border-radius: 12px;
            color: {COLORS['text_secondary']};
            font-size: 14px;
        """)
        self.cam_label.setText("📷\n\nCaméra inactive\nCliquez sur Démarrer")
        left.addWidget(self.cam_label)

        self.toggle_btn = QPushButton("▶  Démarrer la caméra")
        self.toggle_btn.setFixedHeight(44)
        self.toggle_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: {COLORS['bg_dark']};
                border: none;
                border-radius: 10px;
            }}
            QPushButton:hover {{
                background-color: #3db99a;
            }}
        """)
        self.toggle_btn.clicked.connect(self._toggle_camera)
        left.addWidget(self.toggle_btn)

        # Conseils
        tips_card = Card()
        tips_lay = QVBoxLayout(tips_card)
        tips_lay.setContentsMargins(20, 16, 20, 16)
        tips_lay.setSpacing(8)
        tips_title = QLabel("Conseils pour une bonne reconnaissance")
        tips_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        tips_title.setStyleSheet(f"color: {COLORS['accent']};")
        tips_lay.addWidget(tips_title)
        tips = [
            "Placez votre main bien dans le cadre",
            "Privilégiez un éclairage homogène et un fond neutre de préférence",
            "Gardez votre main à 30–50 cm de la caméra",
            "Restez immobile 1–2 secondes pour la prédiction",
        ]
        for tip in tips:
            t = QLabel(tip)
            t.setFont(QFont("Segoe UI", 10))
            t.setStyleSheet(f"color: {COLORS['text_secondary']};")
            tips_lay.addWidget(t)

        left.addWidget(tips_card)

        main_lay.addLayout(left)

        #Droite : résultat + guide
        right = QVBoxLayout()
        right.setSpacing(16)

        pred_title = QLabel("Prédiction du Modèle")
        pred_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        pred_title.setStyleSheet(f"color: {COLORS['text_primary']};")
        right.addWidget(pred_title)

        # Grande lettre prédite
        pred_card = Card()
        pred_card.setFixedHeight(200)
        pred_lay = QVBoxLayout(pred_card)
        pred_lay.setAlignment(Qt.AlignCenter)

        self.pred_letter = QLabel("?")
        self.pred_letter.setFont(QFont("Segoe UI", 100, QFont.Bold))
        self.pred_letter.setStyleSheet(f"color: {COLORS['accent']};")
        self.pred_letter.setAlignment(Qt.AlignCenter)

        self.pred_conf = QLabel("En attente…")
        self.pred_conf.setFont(QFont("Segoe UI", 11))
        self.pred_conf.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.pred_conf.setAlignment(Qt.AlignCenter)

        pred_lay.addWidget(self.pred_letter)
        pred_lay.addWidget(self.pred_conf)
        right.addWidget(pred_card)

        # Barre de confiance
        conf_card = Card()
        conf_lay = QVBoxLayout(conf_card)
        conf_lay.setContentsMargins(20, 16, 20, 16)
        conf_lbl = QLabel("Confiance")
        conf_lbl.setFont(QFont("Segoe UI", 10))
        conf_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.conf_bar = QProgressBar()
        self.conf_bar.setFixedHeight(10)
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['bg_dark']};
                border-radius: 9px;
                border: none;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent']};
                border-radius: 9px;
            }}
        """)
        self.conf_pct_lbl = QLabel("0 %")
        self.conf_pct_lbl.setFont(QFont("Segoe UI", 10))
        self.conf_pct_lbl.setStyleSheet(f"color: {COLORS['accent']};")
        conf_lay.addWidget(conf_lbl)
        conf_lay.addWidget(self.conf_bar)
        conf_lay.addWidget(self.conf_pct_lbl)
        right.addWidget(conf_card)

        # bouton affichage des exemples
        btn_examples_layout = QHBoxLayout()
        btn_examples_layout.addStretch()
        self.btn_examples = QPushButton("Exemples de signes")
        self.btn_examples.setFixedHeight(40)
        self.btn_examples.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_examples.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: {COLORS['bg_dark']};
                border: none;
                border-radius: 10px;
            }}
            QPushButton:hover {{
                background-color: ##3db99a;
            }}
        """)
        self.btn_examples.clicked.connect(self.show_examples)
        btn_examples_layout.addWidget(self.btn_examples)
        right.addLayout(btn_examples_layout)

        right.addStretch()
        main_lay.addLayout(right)

    def show_examples(self):
    # Exemple simple : ouverture d'une petite fenêtre avec des boutons pour chaque vidéo
        dialog = QDialog(self)
        dialog.setWindowTitle("Exemples vidéos")
        dialog.resize(800, 400)
        layout = QVBoxLayout(dialog)

        # Choix Consonnes / Voyelles
        type_layout = QHBoxLayout()
        btn_consonnes = QPushButton("Consonnes")
        btn_voyelles = QPushButton("Voyelles")
        type_layout.addWidget(btn_consonnes)
        type_layout.addWidget(btn_voyelles)
        layout.addLayout(type_layout)

        # Grid de miniatures (juste des boutons ici)
        grid = QGridLayout()
        layout.addLayout(grid)

        # Exemple : listes de vidéos
        consonnes_videos = ["C1.mp4", "C2.mp4"]
        voyelles_videos = ["V1.mp4", "V2.mp4"]

        def load_videos(videos):
            # Efface la grille
            for i in reversed(range(grid.count())):
                widget = grid.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            # Ajoute les boutons
            for idx, path in enumerate(videos):
                btn = QPushButton(os.path.basename(path))
                btn.setFixedSize(120, 90)
                btn.clicked.connect(lambda checked, p=path: play_video(p))
                grid.addWidget(btn, idx // 5, idx % 5)

        def play_video(path):
            # Lecture vidéo simple dans QDialog ou QVideoWidget
            video_widget = QVideoWidget()
            player = QMediaPlayer()
            player.setVideoOutput(video_widget)
            player.setSource(QUrl.fromLocalFile(path))
            player.play()
            # Affiche le lecteur dans le dialogue
            layout.addWidget(video_widget)

        btn_consonnes.clicked.connect(lambda: load_videos(consonnes_videos))
        btn_voyelles.clicked.connect(lambda: load_videos(voyelles_videos))

        load_videos(consonnes_videos)  # par défaut

        dialog.exec()

    def _toggle_camera(self):
        if not self.camera_active:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self._on_frame)
        self.camera_thread.error_signal.connect(self._on_camera_error)
        self.camera_thread.start()
        self.camera_active = True
        self.toggle_btn.setText("⏹  Arrêter la caméra")
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent3']};
                color: white;
                border: none;
                border-radius: 10px;
            }}
            QPushButton:hover {{ background-color: #e05555; }}
        """)

    def _stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.camera_active = False
        self.cam_label.setText("📷\n\nCaméra inactive\nCliquez sur Démarrer")
        self.cam_label.setPixmap(QPixmap())
        self.pred_letter.setText("?")
        self.pred_conf.setText("En attente…")
        self.conf_bar.setValue(0)
        self.conf_pct_lbl.setText("0 %")
        self.toggle_btn.setText("▶  Démarrer la caméra")
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: {COLORS['bg_dark']};
                border: none;
                border-radius: 10px;
            }}
            QPushButton:hover {{ background-color: #3db99a; }}
        """)

    def _on_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            480, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.cam_label.setPixmap(pixmap)
        self._run_inference(frame)

    def _run_inference(self, frame: np.ndarray):
        """
       pipeline ici
        """

    def _on_camera_error(self, msg: str):
        self.cam_label.setText(f"Erreur caméra :\n{msg}")
        self._stop_camera()

    def closeEvent(self, event):
        self._stop_camera()
        super().closeEvent(event)

#------------------------------------------------------------
# Pages de description
class SimplePage(QWidget):

    def __init__(self, text):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        label = QLabel(text)
        label.setFont(QFont("Segoe UI",28,QFont.Bold))
        layout.addWidget(label)

#------------------------------------------------------------
# Barre de navigation
class NavigationBar(QWidget):
    page_changed = Signal(int)

    def __init__(self):
        super().__init__()

        layout = QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)

        names = ["Accueil", "Données", "Scripts", "Démo"]
        for i, name in enumerate(names):
            btn = QPushButton(name)
            btn.setMinimumHeight(50)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda _, idx=i: self.page_changed.emit(idx))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['bg_card']};
                    border: 1px solid {COLORS['border']};
                    border-right: none;
                    font-size: 14px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['bg_hover']};
                }}
            """)
            layout.addWidget(btn)

        layout.itemAt(layout.count()-1).widget().setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_hover']};
            }}
        """)

#------------------------------------------------------------
# Interface principal
class Interface(QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        title = QLabel("Détection et analyse de dactylologie française")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI",26,QFont.Bold))
        title.setStyleSheet(f"color:{COLORS['accent']};")
        main_layout.addWidget(title)
        # barre de navigation
        self.navbar = NavigationBar()
        main_layout.addWidget(self.navbar)
        # pages sur lesquelles on peut naviguer
        self.stack = QStackedWidget()
        self.page_accueil = SimplePage("Accueil")
        self.page_donnees = SimplePage("Données")
        self.page_scripts = SimplePage("Scripts")
        self.page_demo = PageDemo()
        self.stack.addWidget(self.page_accueil)
        self.stack.addWidget(self.page_donnees)
        self.stack.addWidget(self.page_scripts)
        self.stack.addWidget(self.page_demo)
        main_layout.addWidget(self.stack)
        self.navbar.page_changed.connect(self.stack.setCurrentIndex)

#------------------------------------------------------------
# Main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    window = QMainWindow()
    window.setWindowTitle("Projet de Détection et analyse de dactylologie française")
    window.resize(1000, 600)

    interface = Interface()
    window.setCentralWidget(interface)
    window.show()

    sys.exit(app.exec())
