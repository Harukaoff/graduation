@echo off
REM Streamlit Cloud用のファイルをGitHubにアップロードするスクリプト (Windows)

echo ========================================
echo 🚀 Streamlit Cloud デプロイ準備スクリプト
echo ========================================
echo.

echo 📋 必須ファイルを確認中...
echo.

set missing=0

if exist "structural_analysis_app.py" (echo ✅ structural_analysis_app.py) else (echo ❌ structural_analysis_app.py & set missing=1)
if exist "fem_lib.py" (echo ✅ fem_lib.py) else (echo ❌ fem_lib.py & set missing=1)
if exist "draw_lib.py" (echo ✅ draw_lib.py) else (echo ❌ draw_lib.py & set missing=1)
if exist "requirements.txt" (echo ✅ requirements.txt) else (echo ❌ requirements.txt & set missing=1)
if exist "packages.txt" (echo ✅ packages.txt) else (echo ❌ packages.txt & set missing=1)
if exist ".streamlit\config.toml" (echo ✅ .streamlit\config.toml) else (echo ❌ .streamlit\config.toml & set missing=1)
if exist "models\best.pt" (echo ✅ models\best.pt) else (echo ❌ models\best.pt & set missing=1)

echo.
echo 🖼️  テンプレート画像を確認中...
echo.

if exist "templates\pin.png" (echo ✅ templates\pin.png) else (echo ❌ templates\pin.png & set missing=1)
if exist "templates\roller.png" (echo ✅ templates\roller.png) else (echo ❌ templates\roller.png & set missing=1)
if exist "templates\fixed.png" (echo ✅ templates\fixed.png) else (echo ❌ templates\fixed.png & set missing=1)
if exist "templates\hinge.png" (echo ✅ templates\hinge.png) else (echo ❌ templates\hinge.png & set missing=1)
if exist "templates\beam.png" (echo ✅ templates\beam.png) else (echo ❌ templates\beam.png & set missing=1)
if exist "templates\load.png" (echo ✅ templates\load.png) else (echo ❌ templates\load.png & set missing=1)
if exist "templates\UDL.png" (echo ✅ templates\UDL.png) else (echo ❌ templates\UDL.png & set missing=1)
if exist "templates\momentL.png" (echo ✅ templates\momentL.png) else (echo ❌ templates\momentL.png & set missing=1)
if exist "templates\momentR.png" (echo ✅ templates\momentR.png) else (echo ❌ templates\momentR.png & set missing=1)

if %missing%==1 (
    echo.
    echo ⚠️  必要なファイルが見つかりません
    echo    必要なファイルを配置してから再実行してください。
    pause
    exit /b 1
)

echo.
echo ✅ すべての必須ファイルが揃っています！
echo.

REM Git初期化の確認
if not exist ".git" (
    echo 📦 Gitリポジトリを初期化中...
    git init
    git branch -M main
    echo ✅ Gitリポジトリを初期化しました
) else (
    echo ✅ Gitリポジトリは既に初期化されています
)

echo.
echo 📝 ファイルをステージング中...
git add structural_analysis_app.py
git add fem_lib.py
git add draw_lib.py
git add requirements.txt
git add packages.txt
git add .streamlit\config.toml
git add README.md
git add DEPLOY.md
git add .gitignore 2>nul
git add models\best.pt
git add templates\*.png

echo ✅ ファイルをステージングしました
echo.

set /p commit_msg="コミットメッセージを入力 (デフォルト: 'Deploy to Streamlit Cloud'): "
if "%commit_msg%"=="" set commit_msg=Deploy to Streamlit Cloud

git commit -m "%commit_msg%"
echo ✅ コミットしました
echo.

REM リモートリポジトリの確認
git remote | findstr "origin" >nul
if %errorlevel%==0 (
    echo ✅ リモートリポジトリ 'origin' が設定されています
    for /f "tokens=*" %%i in ('git remote get-url origin') do set remote_url=%%i
    echo    URL: %remote_url%
    echo.
    set /p do_push="このリポジトリにプッシュしますか？ (y/n): "
) else (
    echo ⚠️  リモートリポジトリが設定されていません
    echo.
    echo GitHubで新しいリポジトリを作成してから、以下のコマンドを実行してください：
    echo   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    echo.
    set /p remote_url="リモートリポジトリのURLを入力 (スキップする場合は空Enter): "
    
    if not "%remote_url%"=="" (
        git remote add origin "%remote_url%"
        echo ✅ リモートリポジトリを設定しました
        set do_push=y
    ) else (
        set do_push=n
    )
)

if /i "%do_push%"=="y" (
    echo.
    echo 🚀 GitHubにプッシュ中...
    git push -u origin main
    echo.
    echo ✅ プッシュ完了！
    echo.
    echo 次のステップ：
    echo 1. https://streamlit.io/cloud にアクセス
    echo 2. GitHubアカウントでログイン
    echo 3. 'New app' をクリック
    echo 4. リポジトリとブランチを選択
    echo 5. Main file path: structural_analysis_app.py
    echo 6. 'Deploy!' をクリック
) else (
    echo.
    echo プッシュをスキップしました
    echo.
    echo 後でプッシュする場合：
    echo   git push -u origin main
)

echo.
echo ========================================
echo ✅ 完了！
echo ========================================
pause
