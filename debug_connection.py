"""
節点接続のデバッグスクリプト
梁と支点の接続状況を確認
"""

import numpy as np

def test_connection_logic():
    """接続ロジックのテスト"""
    print("=" * 60)
    print("節点接続ロジックのテスト")
    print("=" * 60)
    
    # テストケース1: 支点と梁が近い場合
    print("\n【テストケース1】支点と梁端点が近い場合")
    support_node = np.array([100.0, 200.0])
    beam_endpoint = np.array([102.0, 198.0])
    threshold = 25.0
    
    distance = np.linalg.norm(beam_endpoint - support_node)
    print(f"支点座標: {support_node}")
    print(f"梁端点座標: {beam_endpoint}")
    print(f"距離: {distance:.2f}px")
    print(f"閾値: {threshold}px")
    
    if distance < threshold:
        print("✅ 接続成功: 梁端点を支点にスナップ")
        snapped = support_node
        print(f"スナップ後: {snapped}")
    else:
        print("❌ 接続失敗: 距離が閾値を超えています")
    
    # テストケース2: 支点と梁が遠い場合
    print("\n【テストケース2】支点と梁端点が遠い場合")
    support_node = np.array([100.0, 200.0])
    beam_endpoint = np.array([150.0, 250.0])
    
    distance = np.linalg.norm(beam_endpoint - support_node)
    print(f"支点座標: {support_node}")
    print(f"梁端点座標: {beam_endpoint}")
    print(f"距離: {distance:.2f}px")
    print(f"閾値: {threshold}px")
    
    if distance < threshold:
        print("✅ 接続成功: 梁端点を支点にスナップ")
    else:
        print("❌ 接続失敗: 新しい節点として登録")
        print(f"新規節点: {beam_endpoint}")
    
    # テストケース3: 複数の支点がある場合
    print("\n【テストケース3】複数の支点から最近傍を選択")
    support_nodes = [
        np.array([100.0, 200.0]),
        np.array([300.0, 200.0]),
        np.array([200.0, 100.0])
    ]
    beam_endpoint = np.array([105.0, 195.0])
    
    print(f"梁端点座標: {beam_endpoint}")
    print("支点リスト:")
    for i, node in enumerate(support_nodes):
        dist = np.linalg.norm(beam_endpoint - node)
        print(f"  支点{i}: {node} (距離: {dist:.2f}px)")
    
    distances = [np.linalg.norm(beam_endpoint - node) for node in support_nodes]
    nearest_idx = np.argmin(distances)
    nearest_dist = distances[nearest_idx]
    
    print(f"\n最近傍: 支点{nearest_idx} (距離: {nearest_dist:.2f}px)")
    
    if nearest_dist < threshold:
        print(f"✅ 接続成功: 支点{nearest_idx}にスナップ")
        print(f"スナップ後: {support_nodes[nearest_idx]}")
    else:
        print("❌ 接続失敗: すべての支点が遠すぎます")

def test_arrow_tip_detection():
    """矢じり先端検出のテスト"""
    print("\n" + "=" * 60)
    print("矢じり先端検出のテスト")
    print("=" * 60)
    
    # 荷重の四角形座標（例）
    test_cases = [
        {
            "name": "下向き荷重",
            "pts": np.array([[100, 50], [120, 50], [120, 100], [100, 100]]),
            "angle": 90,
            "expected": "y最大"
        },
        {
            "name": "右向き荷重",
            "pts": np.array([[50, 100], [100, 100], [100, 120], [50, 120]]),
            "angle": 180,
            "expected": "x最大"
        },
        {
            "name": "上向き荷重",
            "pts": np.array([[100, 100], [120, 100], [120, 50], [100, 50]]),
            "angle": 270,
            "expected": "y最小"
        },
        {
            "name": "左向き荷重",
            "pts": np.array([[100, 100], [50, 100], [50, 120], [100, 120]]),
            "angle": 0,
            "expected": "x最小"
        }
    ]
    
    for case in test_cases:
        print(f"\n【{case['name']}】")
        print(f"角度: {case['angle']}度")
        print(f"四角形座標: {case['pts'].tolist()}")
        
        angle = case['angle']
        pts = case['pts']
        
        if 45 <= angle < 135:  # 下向き
            idx = np.argmax(pts[:, 1])
            tip = pts[idx]
            direction = "下向き (y最大)"
        elif 135 <= angle < 225:  # 右向き
            idx = np.argmax(pts[:, 0])
            tip = pts[idx]
            direction = "右向き (x最大)"
        elif 225 <= angle < 315:  # 上向き
            idx = np.argmin(pts[:, 1])
            tip = pts[idx]
            direction = "上向き (y最小)"
        else:  # 左向き
            idx = np.argmin(pts[:, 0])
            tip = pts[idx]
            direction = "左向き (x最小)"
        
        print(f"検出方向: {direction}")
        print(f"矢じり先端: {tip}")
        print(f"期待値: {case['expected']}")
        
        if case['expected'] in direction:
            print("✅ 正しく検出")
        else:
            print("❌ 検出エラー")

def test_unique_nodes():
    """節点の一意性テスト"""
    print("\n" + "=" * 60)
    print("節点の一意性テスト")
    print("=" * 60)
    
    # 座標リスト（丸め誤差を含む）
    coords = [
        (100.0, 200.0),
        (100.01, 200.02),  # ほぼ同じ
        (300.0, 200.0),
        (100.0, 200.0),    # 完全に同じ
        (300.05, 200.03),  # ほぼ同じ
    ]
    
    print("\n元の座標リスト:")
    for i, coord in enumerate(coords):
        print(f"  {i}: {coord}")
    
    # 丸め処理
    rounded_coords = [tuple(np.round(coord, 2)) for coord in coords]
    
    print("\n丸め後の座標リスト:")
    for i, coord in enumerate(rounded_coords):
        print(f"  {i}: {coord}")
    
    # 一意な座標を抽出
    unique_nodes = {}
    node_counter = 0
    for coord in rounded_coords:
        if coord not in unique_nodes:
            unique_nodes[coord] = node_counter
            node_counter += 1
    
    print(f"\n一意な節点数: {len(unique_nodes)}")
    print("節点マッピング:")
    for coord, idx in sorted(unique_nodes.items(), key=lambda x: x[1]):
        print(f"  N{idx}: {coord}")

def main():
    """メインテスト実行"""
    print("\n🔍 構造力学解析アプリ - 接続ロジックテスト\n")
    
    test_connection_logic()
    test_arrow_tip_detection()
    test_unique_nodes()
    
    print("\n" + "=" * 60)
    print("✅ すべてのテストが完了しました")
    print("=" * 60)

if __name__ == "__main__":
    main()
