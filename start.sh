#!/usr/bin/env sh
# Voicer: веб-интерфейс http://127.0.0.1:8765. Первый запуск сам всё установит.
cd "$(dirname "$0")"
[ -f .venv/voicer-ok ] || ./install.sh || exit 1
exec .venv/bin/python -m voicer serve "$@"
