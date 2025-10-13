# tenta compilar/ carregar os módulos C++; cai no stub se falhar
from importlib import import_module

def _try_import_cpp(module_path, fallback):
    try:
        import cppimport  # compila sob demanda
        return cppimport.imp(module_path)
    except Exception:
        return import_module(fallback)

# ncc_fft: cross-correlation normalizada com deslocamentos inteiros
ncc_fft = _try_import_cpp("src.accel.ncc_fft", "src.accel.ncc_stub")