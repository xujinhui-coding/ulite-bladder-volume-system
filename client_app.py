import os
import sys

from PySide6.QtWidgets import QApplication

from client.main_window import MainWindow


def main() -> None:
    import PySide6
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH",
                          os.path.join(os.path.dirname(PySide6.__file__), "plugins", "platforms"))
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
