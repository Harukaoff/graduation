"""
デプロイ前のファイル確認スクリプト
"""
import os
from pathlib import Path

def check_file(path, description):
    """ファイルの存在を確認"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        size_mb = size / (1024 * 1024)
        print(f"✅ {description}")
        print(f"   パス: {path}")
        print(f"   サイズ: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ {description}")
        print(f"   パス: {path} (見つかりません)")
        return False

def main():
    print("=" * 60)
    print("📋 Streamlit Cloud デプロイ前ファイルチェック")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # 必須ファイル
    print("【必須ファイル】")
    print()
    
    files = [
        ("structural_analysis_app.py", "メインアプリ"),
        ("fem_lib.py", "FEM解析ライブラリ"),
        ("draw_lib.py", "描画ライブラリ"),
        ("requirements.txt", "Pythonパッケージ"),
        ("packages.txt", "システムパッケージ"),
        (".streamlit/config.toml", "Streamlit設定"),
        ("models/best.pt", "YOLOモデル"),
    ]
    
    for file, desc in files:
        if not check_file(file, desc):
            all_ok = False
        print()
    
    # テンプレート画像
    print("【テンプレート画像】")
    print()
    
    templates = [
        "pin.png",
        "roller.png",
        "fixed.png",
        "hinge.png",
        "beam.png",
        "load.png",
        "UDL.png",
        "momentL.png",
        "momentR.png",
    ]
    
    for template in templates:
        path = f"templates/{template}"
        if not check_file(path, template):
            all_ok = False
        print()
    
    # ドキュメント（推奨）
    print("【ドキュメント（推奨）】")
    print()
    
    docs = [
        ("README.md", "プロジェクト説明"),
        ("DEPLOY.md", "デプロイガイド"),
        (".gitignore", "Git除外設定"),
    ]
    
    for file, desc in docs:
        check_file(file, desc)
        print()
    
    # 結果
    print("=" * 60)
    if all_ok:
        print("✅ すべての必須ファイルが揃っています！")
        print()
        print("次のステップ:")
        print("1. GitHubにプッシュ:")
        print("   - Windows: upload_to_github.bat を実行")
        print("   - Mac/Linux: ./upload_to_github.sh を実行")
        print()
        print("2. Streamlit Cloudでデプロイ:")
        print("   - https://streamlit.io/cloud にアクセス")
        print("   - リポジトリを選択")
        print("   - structural_analysis_app.py を指定")
    else:
        print("❌ 不足しているファイルがあります")
        print()
        print("必要なファイルを配置してから再度確認してください")
    print("=" * 60)

if __name__ == "__main__":
    main()
