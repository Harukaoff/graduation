#!/bin/bash
# Streamlit Cloud用のファイルをGitHubにアップロードするスクリプト

echo "🚀 Streamlit Cloud デプロイ準備スクリプト"
echo "=========================================="

# 必須ファイルの確認
echo ""
echo "📋 必須ファイルを確認中..."

files=(
    "structural_analysis_app.py"
    "fem_lib.py"
    "draw_lib.py"
    "requirements.txt"
    "packages.txt"
    ".streamlit/config.toml"
    "models/best.pt"
)

missing_files=()
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (見つかりません)"
        missing_files+=("$file")
    fi
done

# テンプレート画像の確認
echo ""
echo "🖼️  テンプレート画像を確認中..."
templates=(
    "templates/pin.png"
    "templates/roller.png"
    "templates/fixed.png"
    "templates/hinge.png"
    "templates/beam.png"
    "templates/load.png"
    "templates/UDL.png"
    "templates/momentL.png"
    "templates/momentR.png"
)

for template in "${templates[@]}"; do
    if [ -f "$template" ]; then
        echo "✅ $template"
    else
        echo "❌ $template (見つかりません)"
        missing_files+=("$template")
    fi
done

# 不足ファイルがある場合は終了
if [ ${#missing_files[@]} -ne 0 ]; then
    echo ""
    echo "⚠️  以下のファイルが見つかりません："
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "必要なファイルを配置してから再実行してください。"
    exit 1
fi

echo ""
echo "✅ すべての必須ファイルが揃っています！"
echo ""

# Git初期化の確認
if [ ! -d ".git" ]; then
    echo "📦 Gitリポジトリを初期化中..."
    git init
    git branch -M main
    echo "✅ Gitリポジトリを初期化しました"
else
    echo "✅ Gitリポジトリは既に初期化されています"
fi

# .gitignoreの確認
if [ ! -f ".gitignore" ]; then
    echo "⚠️  .gitignoreが見つかりません"
    echo "   不要なファイルがコミットされる可能性があります"
fi

# Git LFSの確認
echo ""
echo "📦 Git LFSの設定を確認中..."
if command -v git-lfs &> /dev/null; then
    echo "✅ Git LFSがインストールされています"
    
    # モデルファイルのサイズを確認
    model_size=$(stat -f%z "models/best.pt" 2>/dev/null || stat -c%s "models/best.pt" 2>/dev/null)
    model_size_mb=$((model_size / 1024 / 1024))
    
    if [ $model_size_mb -gt 50 ]; then
        echo "⚠️  モデルファイルが大きいです (${model_size_mb}MB)"
        echo "   Git LFSの使用を推奨します"
        echo ""
        read -p "Git LFSを設定しますか？ (y/n): " use_lfs
        if [ "$use_lfs" = "y" ]; then
            git lfs install
            git lfs track "models/*.pt"
            git add .gitattributes
            echo "✅ Git LFSを設定しました"
        fi
    fi
else
    echo "⚠️  Git LFSがインストールされていません"
    echo "   大きなファイル（>100MB）をプッシュする場合は、Git LFSをインストールしてください"
    echo "   インストール: https://git-lfs.github.com/"
fi

# ファイルを追加
echo ""
echo "📝 ファイルをステージング中..."
git add structural_analysis_app.py
git add fem_lib.py
git add draw_lib.py
git add requirements.txt
git add packages.txt
git add .streamlit/config.toml
git add README.md
git add DEPLOY.md
git add .gitignore 2>/dev/null || true
git add models/best.pt
git add templates/*.png

echo "✅ ファイルをステージングしました"

# コミット
echo ""
read -p "コミットメッセージを入力 (デフォルト: 'Deploy to Streamlit Cloud'): " commit_msg
commit_msg=${commit_msg:-"Deploy to Streamlit Cloud"}

git commit -m "$commit_msg"
echo "✅ コミットしました"

# リモートリポジトリの確認
echo ""
if git remote | grep -q "origin"; then
    echo "✅ リモートリポジトリ 'origin' が設定されています"
    remote_url=$(git remote get-url origin)
    echo "   URL: $remote_url"
    echo ""
    read -p "このリポジトリにプッシュしますか？ (y/n): " do_push
else
    echo "⚠️  リモートリポジトリが設定されていません"
    echo ""
    echo "GitHubで新しいリポジトリを作成してから、以下のコマンドを実行してください："
    echo "  git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
    echo ""
    read -p "リモートリポジトリのURLを入力 (スキップする場合は空Enter): " remote_url
    
    if [ -n "$remote_url" ]; then
        git remote add origin "$remote_url"
        echo "✅ リモートリポジトリを設定しました"
        do_push="y"
    else
        do_push="n"
    fi
fi

# プッシュ
if [ "$do_push" = "y" ]; then
    echo ""
    echo "🚀 GitHubにプッシュ中..."
    git push -u origin main
    echo ""
    echo "✅ プッシュ完了！"
    echo ""
    echo "次のステップ："
    echo "1. https://streamlit.io/cloud にアクセス"
    echo "2. GitHubアカウントでログイン"
    echo "3. 'New app' をクリック"
    echo "4. リポジトリとブランチを選択"
    echo "5. Main file path: structural_analysis_app.py"
    echo "6. 'Deploy!' をクリック"
else
    echo ""
    echo "プッシュをスキップしました"
    echo ""
    echo "後でプッシュする場合："
    echo "  git push -u origin main"
fi

echo ""
echo "=========================================="
echo "✅ 完了！"
