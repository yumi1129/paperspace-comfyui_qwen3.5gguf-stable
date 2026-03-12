Paperspace で作れた cp311 の wheel を使う

なので コンテナも Python 3.11 ベース

llama-cpp-python==0.3.16 は wheel でコンテナに同梱

llama-server も コンテナで事前ビルド

ComfyUI本体 / custom_nodes / モデルDL は ipynb 側

GitHub Actions は GHCR へ push
