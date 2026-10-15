BOAT12: 3連単の風・展示モデル用「学習データ準備」第2段階

置き場所:
trifecta_live_dataset.py -> boat12/ (一番上)
trifecta-live-dataset.yml -> boat12/.github/workflows/

GitHub Actions -> Trifecta Live Dataset -> Run workflow
完了後、GitHubで trifecta_live_report.json を確認。

これは締切前の直前データと3連単確定結果を照合する処理です。
まだ予測確率を変更しません。学習・較正・精度検証は次段階です。
不正な時刻、データ欠損、未確認の結果を含むレースは除外します。
注意: 取得元の時刻が締切前でも、実際のユーザーへの配信時刻を完全には保証しません。
