# Qt imports via compatibility layer
from xpcsviewer.gui.qt_compat import QtCore
from xpcsviewer.utils.logging_config import get_logger

logger = get_logger(__name__)


class ListDataModel(QtCore.QAbstractListModel):
    def __init__(self, input_list=None, max_display=16384) -> None:
        super().__init__()
        if input_list is None:
            self.input_list = []
        else:
            self.input_list = input_list
        self.max_display = max_display

    # overwrite parent method
    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            content = self.input_list[index.row()]
            return str(content)
        return None

    # overwrite parent method
    # index is optional so list-like Python callers can use rowCount() without a
    # QModelIndex; Qt always passes one. (ListDataModel ignores the parent index.)
    def rowCount(self, index=None):
        return min(self.max_display, len(self.input_list))

    def extend(self, new_input_list):
        self.input_list.extend(new_input_list)
        self.layoutChanged.emit()

    def append(self, new_item):
        self.input_list.append(new_item)
        self.layoutChanged.emit()

    def replace(self, new_input_list):
        self.input_list.clear()
        self.extend(new_input_list)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, i):
        return self.input_list[i]

    def pop(self, i=-1):
        return self.input_list.pop(i)

    def insert(self, i, item):
        self.input_list.insert(i, item)
        self.layoutChanged.emit()

    def copy(self):
        return self.input_list.copy()
        self.layoutChanged.emit()
        return None

    def remove(self, x):
        self.input_list.remove(x)
        self.layoutChanged.emit()

    def clear(self):
        self.input_list.clear()
        self.layoutChanged.emit()


def test():
    a = ["a", "b", "c"]
    model = ListDataModel(a)
    for n in range(len(model)):
        logger.debug(f"Test model item {n}: {model[n]}")


if __name__ == "__main__":
    test()
