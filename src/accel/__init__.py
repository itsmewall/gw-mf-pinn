# src/accel/__init__.py
import sys, pathlib
from importlib import import_module

# garante que .../src esteja no sys.path para "import accel"
SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1])
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ativa cppimport
try:
    import cppimport.import_hook  # noqa: F401
    _CPPIMPORT_OK = True
except Exception:
    _CPPIMPORT_OK = False

def _load_ncc():
    # 1) tenta importar a extensão já compilada como submódulo do pacote
    try:
        return import_module(".ncc_fft", package=__name__)
    except Exception as e1:
        # 2) tenta compilar no ato via cppimport
        if _CPPIMPORT_OK:
            try:
                import cppimport
                return cppimport.imp("accel.ncc_fft")
            except Exception as e2:
                _last = f"cppimport failed: {type(e2).__name__}: {e2}"
        else:
            _last = "cppimport unavailable"
        # 3) fallback puro Python
        try:
            return import_module(".ncc_stub", package=__name__)
        except Exception as e3:
            raise ImportError(f"Failed to import accel.ncc_fft. Last={_last}. Fallback error={e3}") from e1

ncc_fft = _load_ncc()  # expõe objeto do módulo carregado
__all__ = ["ncc_fft"]
