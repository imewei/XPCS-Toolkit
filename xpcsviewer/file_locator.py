# Standard library imports
import datetime
import os
import time
import traceback

# Local imports
from .fileIO.qmap_utils import QMapManager
from .helper.listmodel import ListDataModel
from .utils.logging_config import get_logger
from .xpcs_file import XpcsFile as XF

logger = get_logger(__name__)


def create_xpcs_dataset(fname, **kwargs):
    """
    Create a xpcs_file object from a given path.
    Returns None if file cannot be loaded as valid XPCS data.
    """
    try:
        temp = XF(fname, **kwargs)
        return temp
    except KeyError as e:
        logger.warning(
            f"Failed to load file {fname}: Missing required HDF5 dataset - {e}"
        )
    except OSError as e:
        logger.warning(f"Failed to load file {fname}: File I/O error - {e}")
    except Exception as e:
        logger.warning(f"Failed to load file {fname}: {type(e).__name__} - {e}")
        logger.debug("Traceback: %s", traceback.format_exc())

    return None


class FileLocator:
    def __init__(self, path):
        self.path = path
        self.source = ListDataModel()
        self.source_search = ListDataModel()
        self.target = ListDataModel()
        self.qmap_manager = QMapManager()
        self.cache = {}
        self.timestamp = None

    def set_path(self, path):
        self.path = path

    def clear(self):
        self.source.clear()
        self.source_search.clear()

    def get_xf_list(self, rows=None, filter_atype=None, filter_fitted=False):
        """
        get the cached xpcs_file list;
        :param rows: a list of index to select; if None is given, then use
        :return: list of xpcs_file objects;
        """
        if rows is None:
            selected = list(range(len(self.target)))
        elif isinstance(rows, int):
            selected = [rows]
        else:
            selected = rows

        # If rows was explicitly passed as an empty list, return empty result
        if rows is not None and len(selected) == 0:
            return []

        # If no target files are selected but we have cached files, use the cache as fallback
        # This handles cases where files are loaded but target list is temporarily empty
        if not selected and self.cache:
            logger.info(
                "Using cached files as fallback for plotting (target list empty)"
            )
            ret = [xf for xf in self.cache.values() if xf is not None]
            # Apply filters if specified
            if filter_fitted:
                ret = [xf for xf in ret if xf.fit_summary is not None]
            if filter_atype is not None:
                ret = [xf for xf in ret if filter_atype in getattr(xf, "atype", [])]
            return ret

        ret = []
        for n in selected:
            if n < 0 or n >= len(self.target):
                continue
            full_fname = os.path.normpath(os.path.join(self.path, self.target[n]))
            if full_fname not in self.cache:
                xf_obj = create_xpcs_dataset(full_fname, qmap_manager=self.qmap_manager)
                self.cache[full_fname] = xf_obj
            xf_obj = self.cache[full_fname]

            # Skip None objects (failed to load)
            if xf_obj is None:
                logger.warning(
                    f"Skipping invalid file {self.target[n]} (failed to load)"
                )
                continue

            if xf_obj.fit_summary is None and filter_fitted:
                continue
            if filter_atype is None or filter_atype in xf_obj.atype:
                ret.append(xf_obj)

        return ret

    def get_hdf_info(self, fname, filter_str=None):
        """
        get the hdf information / hdf structure for fname
        :param fname: input filename
        :param fstr: list of filter string;
        :return: list of strings that contains the hdf information;
        """
        xf_obj = create_xpcs_dataset(
            os.path.normpath(os.path.join(self.path, fname)),
            qmap_manager=self.qmap_manager,
        )
        return xf_obj.get_hdf_info(filter_str)

    def add_target(self, alist, threshold=256, preload=True):
        if not alist:
            return
        if preload and len(alist) <= threshold:
            t0 = time.perf_counter()
            for fn in alist:
                if fn in self.target:
                    continue
                full_fname = os.path.normpath(os.path.join(self.path, fn))
                xf_obj = create_xpcs_dataset(full_fname, qmap_manager=self.qmap_manager)
                if xf_obj is not None:
                    self.target.append(fn)
                    self.cache[full_fname] = xf_obj
            t1 = time.perf_counter()
            logger.info(f"Load {len(alist)}  files in {t1 - t0:.3f} seconds")
        else:
            logger.info("preload disabled or too many files added")
            self.target.extend(alist)
        self.timestamp = str(datetime.datetime.now())
        return

    def clear_target(self):
        self.target.clear()
        self.cache.clear()

    def remove_target(self, rlist):
        for x in rlist:
            if x in self.target:
                self.target.remove(x)
            self.cache.pop(os.path.normpath(os.path.join(self.path, x)), None)
        if not self.target:
            self.clear_target()
        self.timestamp = str(datetime.datetime.now())

    def reorder_target(self, row, direction="up"):
        size = len(self.target)
        assert 0 <= row < size, "check row value"
        if (direction == "up" and row == 0) or (
            direction == "down" and row == size - 1
        ):
            return -1

        item = self.target.pop(row)
        pos = row - 1 if direction == "up" else row + 1
        self.target.insert(pos, item)
        idx = self.target.index(pos)
        self.timestamp = str(datetime.datetime.now())
        return idx

    def search(self, val, filter_type="prefix"):
        assert filter_type in [
            "prefix",
            "substr",
        ], "filter_type must be prefix or substr"
        if filter_type == "prefix":
            selected = [x for x in self.source if x.startswith(val)]
        elif filter_type == "substr":
            filter_words = val.split()  # Split search query by whitespace
            selected = [x for x in self.source if all(t in x for t in filter_words)]
        self.source_search.replace(selected)

    def build(self, path=None, filter_list=(".hdf", ".h5"), sort_method="Filename"):
        self.path = path
        flist = [
            entry.name
            for entry in os.scandir(path)
            if entry.is_file()
            and entry.name.lower().endswith(filter_list)
            and not entry.name.startswith(".")
        ]
        if sort_method.startswith("Filename"):
            flist.sort()
        elif sort_method.startswith("Time"):
            flist.sort(
                key=lambda x: os.path.getmtime(os.path.normpath(os.path.join(path, x)))
            )
        elif sort_method.startswith("Index"):
            pass

        if sort_method.endswith("-reverse"):
            flist.reverse()
        self.source.replace(flist)
        return True


if __name__ == "__main__":
    # test1()
    fl = FileLocator(path="./data/files.txt")
