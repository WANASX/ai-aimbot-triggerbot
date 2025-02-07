import sys
import os
import subprocess
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox
)
from PyQt5.QtCore import Qt

class Loader(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loader")
        self.setFixedSize(400, 200)
        self.init_ui()

    def init_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)


        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter) 

 
        self.logitech_button = QPushButton("Logitech")
        self.logitech_button.setFixedSize(200, 50) 
        self.logitech_button.clicked.connect(self.run_logitech)
        layout.addWidget(self.logitech_button)


        self.no_logitech_button = QPushButton("No Logitech (For Win11 Gays)")
        self.no_logitech_button.setFixedSize(230, 50) 
        self.no_logitech_button.clicked.connect(self.run_no_logitech)
        layout.addWidget(self.no_logitech_button)

        central_widget.setLayout(layout)

       
        style_sheet = """
        QWidget {
            background-color: #2b2b2b;
            color: #FFFFFF;
            font-size: 14px;
        }

        QLabel {
            color: #32CD32;
        }

        QGroupBox {
            border: 1px solid #32CD32;
            margin-top: 10px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }

        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #3c3c3c;
            color: #FFFFFF;
            border: 1px solid #32CD32;
            padding: 2px;
            border-radius: 5px;
        }

        QCheckBox {
            color: #FFFFFF;
        }

        QPushButton {
            background-color: #32CD32;
            color: #FFFFFF;
            border: none;
            padding: 8px;
            border-radius: 5px;
            font-size: 16px;
        }

        QPushButton:hover {
            background-color: #28a428;
        }

        QScrollBar:vertical, QScrollBar:horizontal {
            background-color: #3c3c3c;
            width: 15px;
            margin: 15px 3px 15px 3px;
            border: 1px solid #32CD32;
        }

        QScrollBar::handle {
            background-color: #32CD32;
            min-height: 20px;
            border-radius: 7px;
        }

        QComboBox QAbstractItemView {
            background-color: #3c3c3c;
            selection-background-color: #32CD32;
            selection-color: #FFFFFF;
            border: 1px solid #32CD32;
        }

        QToolTip {
            background-color: #3c3c3c;
            color: #FFFFFF;
            border: 1px solid #32CD32;
        }
        """
        self.setStyleSheet(style_sheet)

    def run_logitech(self):
        """Runs the Logitech.py script."""
        script_name = "Logitech.py"
        self.run_script(script_name)

    def run_no_logitech(self):
        """Runs the No-Logitech.py script."""
        script_name = "No-Logitech.py"
        self.run_script(script_name)

    def run_script(self, script_name):
        """
        Executes a Python script located in the same directory as loader.py.

        Args:
            script_name (str): The name of the script to execute.
        """
        # Get the absolute path to the script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)

        if os.path.exists(script_path):
            try:
                # Execute the script using the same Python interpreter
                subprocess.Popen([sys.executable, script_path], cwd=os.path.dirname(script_path))
                logger.info(f"Loader: Successfully started {script_name}")
            except Exception as e:
                logger.error(f"Loader: Failed to start {script_name}: {e}")
                QMessageBox.critical(self, "Error", f"Failed to start {script_name}:\n{e}")
        else:
            logger.error(f"Loader: {script_name} not found at {script_path}")
            QMessageBox.critical(self, "Error", f"{script_name} not found at {script_path}")

def setup_logger():
    """
    Sets up the logger to log messages to both a file and the console.
    """
    logger = logging.getLogger("Loader")
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler("Loader.log")
    console_handler = logging.StreamHandler(sys.stdout)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Initialize the logger
logger = setup_logger()

def main():
    app = QApplication(sys.argv)
    loader = Loader()
    loader.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
