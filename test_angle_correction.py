"""
梁の角度補正テスト
15度刻みに角度を補正する処理の確認
"""

import numpy as np
import math

def round_angle_15deg(angle):
    """角度を15度刻みに丸める"""
    return round(angle / 15) * 15

def correct_beam_angle(node1, node2):
    """梁の角度を15度刻みに補正"""
    # 現在の角度を計算
    current_angle = math.degrees(math.atan2(node2[1] - node1[1], 
                                            node2[0] - node1[0]))
    if current_angle < 0:
        current_angle += 360
    
    # 15度刻みに丸める
    corrected_angle = round_angle_15deg(current_angle)
    
    # 梁の長さを保持
    beam_length = np.linalg.norm(node2 - node1)
    
    # 補正後の角度で端点2の新しい座標を計算
    angle_rad = math.radians(corrected_angle)
    new_node2_x = node1[0] + beam_length * math.cos(angle_rad)
    new_node2_y = node1[1] + beam_length * math.sin(angle_rad)
    new_node2 = np.array([new_node2_x, new_node2_y])
    
    return new_node2, current_angle, corrected_angle

def test_angle_correction():
    """角度補正のテスト"""
    print("=" * 60)
    print("梁の角度補正テスト（15度刻み）")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "ほぼ水平（3度）",
            "node1": np.array([100.0, 200.0]),
            "node2": np.array([300.0, 210.0]),
            "expected": 0.0
        },
        {
            "name": "ほぼ水平（-2度）",
            "node1": np.array([100.0, 200.0]),
            "node2": np.array([300.0, 193.0]),
            "expected": 0.0
        },
        {
            "name": "約45度（47度）",
            "node1": np.array([100.0, 200.0]),
            "node2": np.array([300.0, 415.0]),
            "expected": 45.0
        },
        {
            "name": "約90度（88度）",
            "node1": np.array([100.0, 200.0]),
            "node2": np.array([107.0, 400.0]),
            "expected": 90.0
        },
        {
            "name": "約180度（178度）",
            "node1": np.array([300.0, 200.0]),
            "node2": np.array([100.0, 193.0]),
            "expected": 180.0
        },
        {
            "name": "約270度（268度）",
            "node1": np.array([100.0, 200.0]),
            "node2": np.array([107.0, 0.0]),
            "expected": 270.0
        },
    ]
    
    for case in test_cases:
        print(f"\n【{case['name']}】")
        node1 = case["node1"]
        node2 = case["node2"]
        expected = case["expected"]
        
        # 元の角度を計算
        original_angle = math.degrees(math.atan2(node2[1] - node1[1], 
                                                 node2[0] - node1[0]))
        if original_angle < 0:
            original_angle += 360
        
        # 角度補正
        new_node2, current_angle, corrected_angle = correct_beam_angle(node1, node2)
        
        # 元の長さ
        original_length = np.linalg.norm(node2 - node1)
        # 補正後の長さ
        new_length = np.linalg.norm(new_node2 - node1)
        
        print(f"端点1: ({node1[0]:.1f}, {node1[1]:.1f})")
        print(f"端点2（元）: ({node2[0]:.1f}, {node2[1]:.1f})")
        print(f"端点2（補正後）: ({new_node2[0]:.1f}, {new_node2[1]:.1f})")
        print(f"角度（元）: {current_angle:.2f}°")
        print(f"角度（補正後）: {corrected_angle:.2f}°")
        print(f"期待値: {expected:.2f}°")
        print(f"長さ（元）: {original_length:.2f}px")
        print(f"長さ（補正後）: {new_length:.2f}px")
        print(f"座標移動: ({node2[0] - new_node2[0]:.2f}, {node2[1] - new_node2[1]:.2f})")
        
        # 検証
        if abs(corrected_angle - expected) < 0.1:
            print("✅ 角度補正: 正しい")
        else:
            print(f"❌ 角度補正: エラー（期待値: {expected}°, 実際: {corrected_angle}°）")
        
        if abs(original_length - new_length) < 0.1:
            print("✅ 長さ保持: 正しい")
        else:
            print(f"❌ 長さ保持: エラー（差: {abs(original_length - new_length):.2f}px）")

def test_15deg_increments():
    """15度刻みの確認"""
    print("\n" + "=" * 60)
    print("15度刻みの確認")
    print("=" * 60)
    
    print("\n0°～360°の範囲で15度刻みに丸める:")
    for angle in [0, 7, 8, 15, 22, 23, 30, 45, 82, 83, 90, 
                  135, 172, 173, 180, 225, 262, 263, 270, 315, 352, 353, 360]:
        rounded = round_angle_15deg(angle)
        print(f"{angle:3d}° → {rounded:3.0f}°")

def main():
    """メインテスト実行"""
    print("\n🔍 梁の角度補正テスト\n")
    
    test_angle_correction()
    test_15deg_increments()
    
    print("\n" + "=" * 60)
    print("✅ すべてのテストが完了しました")
    print("=" * 60)

if __name__ == "__main__":
    main()
