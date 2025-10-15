#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.."; pwd)"
src="$root/src"
pkg="$src/gwmf"

mkdir -p "$pkg"/{accel,core,data,eval,models,physics,viz,stages/{mf1,mf2,pinn3,params4},cli}

# move blocos
mv "$src/accel"/*             "$pkg/accel/"        2>/dev/null || true
mv "$src/gwdata"/*            "$pkg/data/"         2>/dev/null || true
mv "$src/physics"/*           "$pkg/physics/"      2>/dev/null || true
mv "$src/models"/*            "$pkg/models/"       2>/dev/null || true
mv "$src/viz"/*               "$pkg/viz/"          2>/dev/null || true
mv "$src/sim"/*               "$pkg/physics/"      2>/dev/null || true

# eval e estágios
mv "$src/eval/dataset_builder.py" "$pkg/data/"      2>/dev/null || true
mv "$src/eval/baseline_ml.py"     "$pkg/stages/mf2/ml_baseline.py" 2>/dev/null || true
mv "$src/eval/mf_baseline.py"     "$pkg/stages/mf2/mf_baseline.py" 2>/dev/null || true

# stage2 treinadores
mkdir -p "$pkg/stages/mf2"
mv "$src/mf/"* "$pkg/stages/mf2/" 2>/dev/null || true
# pinn
mv "$src/mf_pinn/"* "$pkg/stages/pinn3/" 2>/dev/null || true

# cli
mv "$root/run.py"         "$pkg/cli/run.py"        2>/dev/null || true
mv "$root/run_gwosc.py"   "$pkg/cli/run_gwosc.py"  2>/dev/null || true
mv "$root/debug_win.py"   "$pkg/cli/debug_win.py"  2>/dev/null || true

# __init__ vazios
touch "$pkg/__init__.py"
for d in accel core data eval models physics viz stages stages/mf1 stages/mf2 stages/pinn3 stages/params4 cli; do
  touch "$pkg/$d/__init__.py"
done

# shims de compatibilidade
cat > "$src/eval/__init__.py" <<'PY'
# Compat layer
from gwmf.data.dataset_builder import main as dataset_builder_main
from gwmf.stages.mf2.ml_baseline import main as ml_main
from gwmf.stages.mf2.mf_baseline import main as mf_main
# nomes antigos
dataset_builder = type("X",(object,),{"main":dataset_builder_main})
baseline_ml     = type("Y",(object,),{"main":ml_main})
mf_baseline     = type("Z",(object,),{"main":mf_main})
PY

cat > "$src/gwdata/__init__.py" <<'PY'
# Compat layer
from gwmf.data.preprocess import *   # noqa
from gwmf.data import preprocess
PY

cat > "$src/mf/__init__.py" <<'PY'
# Compat layer
from gwmf.stages.mf2.sweep import *  # noqa
PY

echo "Reorg ok. Ajuste imports no seu editor para o pacote gwmf."