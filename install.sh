#!/usr/bin/env sh
# Установка Voicer на macOS/Linux: uv, Python 3.11 и библиотеки в папку .venv.
# Проверено только на Windows (install.bat); здесь — те же шаги без зафиксированных версий.
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Устанавливаю uv — менеджер Python: https://docs.astral.sh/uv/"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
fi

[ -x .venv/bin/python ] || uv venv .venv --python 3.11
PY=.venv/bin/python

extras=mcp
if [ "$(uname)" = Linux ] && command -v nvidia-smi >/dev/null 2>&1; then
  extras=mcp,gpu
fi
uv pip install --python "$PY" -e ".[$extras]"

# Шумоподавление DeepFilterNet: необязательно. deepfilternet 0.5.6 требует numpy<2, но работает и с numpy 2,
# поэтому ставится без зависимостей.
torch_index=""
[ "$(uname)" = Linux ] && torch_index="--index-url https://download.pytorch.org/whl/cpu"
if uv pip install --python "$PY" $torch_index torch torchaudio \
   && uv pip install --python "$PY" deepfilterlib==0.5.6 appdirs loguru packaging requests sympy \
   && uv pip install --python "$PY" --no-deps deepfilternet==0.5.6; then
  :
else
  echo "Шумоподавление не установилось: всё остальное работает."
fi

touch .venv/voicer-ok
echo "Готово. Запуск: ./start.sh"
