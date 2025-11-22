# ロゴの設定方法

## 📋 手順

### 1. ロゴファイルを配置

ロゴ画像を以下の場所に保存してください：

```
assets/logo.png
```

### 2. ファイル形式

- **推奨形式**: PNG（透過背景）
- **推奨サイズ**: 
  - 幅: 500-1000px
  - 高さ: 500-1000px（正方形推奨）
- **ファイルサイズ**: 1MB以下

### 3. GitHubにアップロード

```bash
# assetsフォルダとロゴを追加
git add assets/logo.png
git add structural_analysis_app.py
git add README.md
git add .gitignore

# コミット
git commit -m "Add InstaStruct logo"

# プッシュ
git push origin main
```

### 4. ロゴの表示場所

ロゴは以下の場所に表示されます：

1. **メインページ**
   - タイトルの左側に表示（幅: 120px）

2. **サイドバー**
   - サイドバーの上部に表示（全幅）

3. **ブラウザタブ**
   - ページアイコンとして🏗️が表示

4. **README.md**
   - GitHubリポジトリのトップに表示

## 🎨 ロゴのカスタマイズ

### サイズ調整

メインページのロゴサイズを変更する場合：

```python
# structural_analysis_app.py の該当箇所
st.image(logo_path, width=120)  # ← この数値を変更
```

### 位置調整

タイトルとロゴの列幅を変更する場合：

```python
# structural_analysis_app.py の該当箇所
col_logo, col_title = st.columns([1, 4])  # ← この比率を変更
```

## 🔍 トラブルシューティング

### ロゴが表示されない

1. **ファイルパスを確認**
   ```bash
   # ファイルが存在するか確認
   ls assets/logo.png
   ```

2. **ファイル名を確認**
   - 正確に `logo.png` という名前か確認
   - 大文字小文字を確認

3. **Streamlit Cloudの場合**
   - GitHubにプッシュされているか確認
   - アプリを再起動

### ロゴが大きすぎる/小さすぎる

```python
# メインページ
st.image(logo_path, width=150)  # 数値を調整

# サイドバー
st.image(logo_path, use_container_width=True)  # 全幅表示
# または
st.image(logo_path, width=200)  # 固定幅
```

## 📝 現在の設定

### メインページ
- ロゴ幅: 120px
- タイトル: "InstaStruct"
- サブタイトル: "構造力学解析アプリ"

### サイドバー
- ロゴ: 全幅表示
- 区切り線あり

### ページ設定
- タイトル: "InstaStruct - 構造力学解析アプリ"
- アイコン: 🏗️
- メニュー: GitHubリンク、About情報

## 🎯 次のステップ

1. ロゴファイルを `assets/logo.png` に保存
2. GitHubにプッシュ
3. Streamlit Cloudで確認

---

**開発者**: 森本遥香 (DA22340)
