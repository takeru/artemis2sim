# Artemis II Mission Simulator

アルテミス2の月フライバイミッションを3Dで体験できるインタラクティブなWebシミュレーター。
JPL Horizonsの実軌道データを使用。

## デモ

ローカルで実行:
```bash
python3 -m http.server 8765
open http://localhost:8765
```

## データソース

- **軌道データ:** JPL Horizons API（宇宙機ID: -1024, 地球中心ICRF座標系, 10分/1分間隔）
- **地球テクスチャ:** NASA Blue Marble (visibleearth.nasa.gov)
- **月テクスチャ:** NASA SVS CGI Moon Kit (svs.gsfc.nasa.gov/4720)
- **太陽方向:** JPL Horizons（2026-04-02時点, ミッション中ほぼ一定のため固定値）
- **地球自転:** IAU 2000 ERA（Earth Rotation Angle）モデルで時刻同期
- **ミッションイベント:** NASA Artemis II Press Kit

## ファイル構成

```
index.html                  # シミュレーター本体（単一HTML, Three.js）
trajectory.json             # 軌道データ（gen_trajectory.pyで生成）
textures/
  earth.jpg                 # NASA Blue Marble (2.5MB)
  moon.jpg                  # NASA SVS LRO color map (136KB)
data/horizons/
  fetch.sh                  # Horizons APIからデータ取得
  spacecraft.txt            # 宇宙機ベクトル（10分間隔）
  spacecraft_fine.txt       # 宇宙機ベクトル（1分間隔, 近地点付近）
  moon.txt                  # 月ベクトル（10分間隔）
  moon_fine.txt             # 月ベクトル（1分間隔）
gen_trajectory.py           # Horizonsデータ → trajectory.json 変換
docs/
  glossary.md               # 用語集
```

## 打ち上げ延期時のデータ更新

Horizonsのエフェメリスは特定の打ち上げ日時に紐づいている。
打ち上げが延期された場合、以下の手順でデータを更新する。

### 1. Horizonsからデータ再取得

```bash
cd data/horizons
./fetch.sh 2026-04-03 2026-04-12   # 新しい開始日 終了日
```

引数なしで実行するとデフォルト（2026-04-02〜2026-04-10）を取得。

### 2. trajectory.json を再生成

```bash
python3 gen_trajectory.py                     # デフォルト: 2026-04-01 22:24 UTC
python3 gen_trajectory.py 2026-04-02T18:24    # 打ち上げ時刻を指定（UTC）
```

### 3. ブラウザリロード

trajectory.json が更新されればシミュレーターは自動的に新しいデータを読み込む。

## 軌道データの精度

| 区間 | データソース | 精度 |
|------|-------------|------|
| MET 0〜3.6h（打ち上げ〜ICPS分離付近） | 計算で補完（vis-viva楕円弧） | 概算 |
| MET 3.6h〜216.6h（ICPS分離後〜帰還） | JPL Horizons実データ | 高精度 |

補完区間はHorizonsにエフェメリスが存在しないため、軌道面と楕円軌道パラメータから合成している。
Horizonsデータとの接続点では位置・速度とも滑らかにブレンドしている。

## 技術メモ

- 座標系: ICRF（地球中心慣性系）。地球の自転軸はZ軸方向
- 3D空間の縮尺: 384,400 km（地球-月間距離）= 100単位。つまり1単位 = 3,844 km
- 地球・月サイズ: 実スケール比率（地球半径6,378km, 月半径1,737km）
- 宇宙船モデル: Three.jsプリミティブで構成した簡易Orion（CM + ESM + ソーラーパネル）。視認性のため実際の数万倍に拡大表示
- 通信断判定: 地球-宇宙船の見通し線が月の球体と交差するかで判定

## ライセンス

NASAの画像・データはパブリックドメイン。
