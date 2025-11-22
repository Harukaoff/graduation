# Streamlit Cloud アップロードチェックリスト

## ✅ 必須ファイル

### アプリケーション
- [ ] `structural_analysis_app.py`
- [ ] `fem_lib.py`
- [ ] `draw_lib.py`

### 依存関係
- [ ] `requirements.txt`
- [ ] `packages.txt`

### 設定
- [ ] `.streamlit/config.toml`

### モデルとテンプレート
- [ ] `models/best.pt` （YOLOモデル）
- [ ] `templates/pin.png`
- [ ] `templates/roller.png`
- [ ] `templates/fixed.png`
- [ ] `templates/hinge.png`
- [ ] `templates/beam.png`
- [ ] `templates/load.png`
- [ ] `templates/UDL.png`
- [ ] `templates/momentL.png`
- [ ] `templates/momentR.png`

### ドキュメント（推奨）
- [ ] `README.md`
- [ ] `DEPLOY.md`
- [ ] `.gitignore`

## 📦 GitHubへのアップロード手順

### ステップ1: ファイルを確認
```bash
# 必須ファイルが存在するか確認
ls structural_analysis_app.py fem_lib.py draw_lib.py
ls requirements.txt packages.txt
ls .streamlit/config.toml
ls models/best.pt
ls templates/*.png
```

### ステップ2: Gitリポジトリを初期化（初回のみ）
```bash
git init
git branch -M main
```

### ステップ3: .gitignoreを確認
```bash
# .gitignoreが正しく設定されているか確認
cat .gitignore
```

### ステップ4: ファイルを追加
```bash
# 必須ファイルを追加
git add structural_analysis_app.py
git add fem_lib.py
git add draw_lib.py
git add requirements.txt
git add packages.txt
git add .streamlit/config.toml
git add README.md
git add DEPLOY.md
git add .gitignore

# モデルファイル（Git LFS推奨）
git lfs track "models/*.pt"
git add .gitattributes
git add models/best.pt

# テンプレート画像
git add templates/*.png
```

### ステップ5: コミット
```bash
git commit -m "Initial commit for Streamlit Cloud deployment"
```

### ステップ6: GitHubリポジトリに接続
```bash
# GitHubで新しいリポジトリを作成後
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### ステップ7: プッシュ
```bash
git push -u origin main
```

## 🚀 Streamlit Cloudでのデプロイ

1. [Streamlit Cloud](https://streamlit.io/cloud) にアクセス
2. GitHubアカウントでログイン
3. 「New app」をクリック
4. 以下を設定：
   - **Repository**: `YOUR_USERNAME/YOUR_REPO_NAME`
   - **Branch**: `main`
   - **Main file path**: `structural_analysis_app.py`
5. 「Deploy!」をクリック

## ⚠️ 注意事項

### モデルファイルのサイズ
- `best.pt`が100MB以上の場合、Git LFSを使用してください
- Git LFSの無料枠は1GBまで

### Git LFSの設定
```bash
# Git LFSをインストール
git lfs install

# モデルファイルをLFS管理下に
git lfs track "models/*.pt"
git add .gitattributes
git add models/best.pt
git commit -m "Add model with Git LFS"
git push
```

### モデルファイルが大きすぎる場合
外部ストレージ（Google Drive、Dropbox等）を使用し、アプリ起動時にダウンロードする方法もあります。
詳細は `DEPLOY.md` を参照してください。

## 🔍 デプロイ後の確認

- [ ] アプリが正常に起動する
- [ ] 画像をアップロードできる
- [ ] モデルが正しく読み込まれる
- [ ] テンプレート画像が表示される
- [ ] 解析が実行できる
- [ ] 結果が表示される

## 🐛 トラブルシューティング

### エラー: `ModuleNotFoundError: No module named 'cv2'`
→ `requirements.txt`と`packages.txt`を確認

### エラー: `FileNotFoundError: models/best.pt`
→ モデルファイルがGitHubにプッシュされているか確認

### エラー: メモリ不足
→ モデルを軽量化するか、有料プランを検討

詳細は `DEPLOY.md` を参照してください。

---

**開発者**: 森本遥香 (DA22340)
