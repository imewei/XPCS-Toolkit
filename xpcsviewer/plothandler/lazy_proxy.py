from xpcsviewer.gui.qt_compat import QVBoxLayout, QWidget


class LazyMplCanvasBarV(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.real_widget = None
        self._is_loaded = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._is_loaded:
            import importlib

            module = importlib.import_module("xpcsviewer.plothandler.matplot_qt")
            self.real_widget = module.MplCanvasBarV(self)
            self.layout.addWidget(self.real_widget)
            self._is_loaded = True

    def clear(self):
        if self._is_loaded:
            self.real_widget.clear()

    def apply_theme(self, theme: str):
        if self._is_loaded:
            self.real_widget.apply_theme(theme)

    def __getattr__(self, name):
        if not self._is_loaded:
            import importlib

            module = importlib.import_module("xpcsviewer.plothandler.matplot_qt")
            self.real_widget = module.MplCanvasBarV(self)
            self.layout.addWidget(self.real_widget)
            self._is_loaded = True
        return getattr(self.real_widget, name)
