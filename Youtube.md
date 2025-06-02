# 📊 YouTube 廣告指標參考資料（2024–2025）

本文件整理了 YouTube 廣告在 2024–2025 年的實際數據範圍，可用於模擬分析或廣告策略規劃。

---

## 🎯 CPM（每千次曝光成本）

- **全球平均 CPM**：約 **$3.61 美元**
- **高收入國家 CPM 範圍**：
  - 美國：約 **$36.03 美元**
  - 澳洲：約 **$39.83 美元**

> 💡 CPM（Cost Per Mille）受廣告類型、目標市場、平台定位影響很大。

---

## 📈 CTR（Click-Through Rate，點擊率）

- **整體平均 CTR**：約 **0.65%**（每 1,000 次曝光約有 6.5 次點擊）

### 各產業平均 CTR：
| 產業類型 | CTR |
|----------|-----|
| 🎮 遊戲     | 0.90% |
| 🌍 旅遊     | 0.78% |
| 🛒 零售     | 0.84% |
| 🎓 教育     | 0.56% |
| 🎬 娛樂     | 0.35% |
| 🍔 餐飲     | 0.04% |

> 💡 CTR 會隨廣告內容、投放策略、受眾精準度不同而變化。

---

## 🔄 CR（Conversion Rate，轉換率）

- **目前缺乏公開穩定的 YouTube CR 數據**
- 建議模擬設定在 **5%–10%** 範圍內，並配合產品類型做 A/B 測試校正。

---

## 🔧 模擬參數建議（適用於 Python 模型）

```python
# CPM：每千次曝光成本（單位：USD）
CPM = 3.5  # 建議值，可依地區市場調整

# CTR：點擊率範圍
CTR_RANGE = (0.006, 0.009)  # 0.6% 到 0.9%

# CR：轉換率範圍
CR_RANGE = (0.05, 0.10)  # 5% 到 10%

## 📚 參考資料

1. [YouTube Ads Benchmarks (2025) – Store Growers](https://www.storegrowers.com/youtube-ads-benchmarks/)
2. [YouTube Ad Benchmarks by Industry (2024–2025) – Mega Digital](https://megadigital.ai/en/blog/youtube-ad-benchmarks/)
3. [YouTube CPM Rates by Country – RatherGoodX](https://rathergoodx.com/youtube-cpm-rates/)
