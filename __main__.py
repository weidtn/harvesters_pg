#!/usr/bin/env python3
import sys
from PyQt6 import QtWidgets
from ui import MainWindow

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
