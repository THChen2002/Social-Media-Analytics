import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import os
from itertools import combinations

# ========== Constants ==========
DATA_PATH = "data/social media influencers - youtube.csv"

# Beta 分布參數，用來模擬每位網紅的觸擊率（reach probability）
ALPHA = 2    # 成功次數的先驗參數
BETA = 10    # 失敗次數的先驗參數

# 模擬抽樣的次數（越多越穩定）
N_TRIALS = 1000

# 兩位網紅合作時，觀眾重疊比例（0 表示完全不重疊，1 表示完全重疊）
OVERLAP_RATIO = 0.25

# 廣告轉換後每單位可帶來的收益（單位：自訂，如 USD）
REVENUE_PER_CONVERSION = 5

# 每千次曝光的廣告成本（常見單位：USD）
CPM = 10

# 點擊率 CTR、轉換率 CR 的模擬區間（使用 uniform 隨機分布）
CTR_RANGE = (0.02, 0.05)
CR_RANGE = (0.05, 0.15)

# 要納入分析的前 N 名網紅
TOP_N = 10

# ========== Data Preparation ==========

# 將訂閱數字串（例如 "1.2M", "3.4K"）轉換為數值型態
def parse_subscriber_count(value):
    value = str(value).strip().upper()
    if 'K' in value:
        return float(value.replace('K', '')) * 1e3
    elif 'M' in value:
        return float(value.replace('M', '')) * 1e6
    elif 'B' in value:
        return float(value.replace('B', '')) * 1e9
    else:
        try:
            return float(value.replace(',', ''))
        except ValueError:
            return None

# 載入 CSV 並篩選出有效訂閱數的資料
def load_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    # 將欄位名稱轉為小寫並去除空格
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['subscribers'] = df['subscribers'].apply(parse_subscriber_count)
    df = df[df['subscribers'].notnull() & (df['subscribers'] > 0)]
    return df



# ========== Simulation Models ==========

# 使用 Beta 分布模擬單一網紅的觸擊率，並計算其實際觸及人數
def simulate_reach_distribution(followers, alpha, beta, n_trials):
    with pm.Model() as model:
        reach_prob = pm.Beta('reach_prob', alpha=alpha, beta=beta)
        trace = pm.sample(draws=n_trials, chains=4, tune=1000, target_accept=0.9,
                          progressbar=False, random_seed=42)
    reach_samples = trace.posterior['reach_prob'].stack(draws=("chain", "draw")).values
    return followers * reach_samples

# 模擬單一網紅的觸及人數
def simulate_single_influencer(followers, alpha, beta, n_trials):
    return simulate_reach_distribution(followers, alpha, beta, n_trials)

# 模擬兩位網紅合作的觸及人數，考慮觀眾重疊（overlap）
def simulate_dual_influencer(f1, f2, overlap_ratio, alpha, beta, n_trials):
    r1 = simulate_reach_distribution(f1, alpha, beta, n_trials)
    r2 = simulate_reach_distribution(f2, alpha, beta, n_trials)
    combined = r1 + r2 - overlap_ratio * np.minimum(r1, r2)
    return combined


# ========== Metrics Calculation ==========

# 根據觸及樣本，模擬點擊率與轉換率後計算收益、成本與 ROI
def compute_ad_metrics(reach_array, ctr_range, cr_range, revenue_per_conversion, cpm):
    ctr = np.random.uniform(*ctr_range, size=len(reach_array))
    cr = np.random.uniform(*cr_range, size=len(reach_array))
    clicks = reach_array * ctr
    conversions = clicks * cr
    revenue = conversions * revenue_per_conversion
    cost = (reach_array / 1000) * cpm
    roi = (revenue - cost) / cost
    return pd.DataFrame({
        'Reach': reach_array,
        'Revenue': revenue,
        'Cost': cost,
        'ROI': roi
    })

def save_roi_histogram(df, title, path):
    if df['ROI'].std() > 1e-6:
        df['ROI'].hist(bins=40)
        plt.title(title)
        plt.xlabel("ROI")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"ROI 圖已儲存：{path}")
    else:
        print(f"{title}的ROI幾乎無變化，未繪製圖表")

def main():
    os.makedirs("results/single", exist_ok=True)
    os.makedirs("results/dual", exist_ok=True)

    df = load_clean_data(DATA_PATH)
    top_df = df.sort_values(by="subscribers", ascending=False).head(TOP_N)

    # 單人模擬
    single_summary = []
    for row in top_df.itertuples(index=False):
        name = row.channel_name
        subs = row.subscribers
        print(f"\nSimulating SINGLE Influencer: {name} ({int(subs)} subscribers)")
        reach = simulate_single_influencer(subs, ALPHA, BETA, N_TRIALS)
        metrics = compute_ad_metrics(reach, CTR_RANGE, CR_RANGE, REVENUE_PER_CONVERSION, CPM)

        single_summary.append({
            'Influencer': name,
            'Mean ROI': metrics['ROI'].mean(),
            'Median ROI': metrics['ROI'].median(),
            'Std ROI': metrics['ROI'].std()
        })

        # 儲存圖
        filename = f"results/single/ROI_single_{name.replace(' ', '_')}.png"
        save_roi_histogram(metrics, f"ROI Distribution: {name}", filename)

    pd.DataFrame(single_summary).to_csv("results/single_influencer_summary.csv", index=False)

    # 雙人模擬
    pair_summary = []
    for a, b in combinations(top_df.itertuples(index=False), 2):
        name1 = a.channel_name
        name2 = b.channel_name
        f1 = a.subscribers
        f2 = b.subscribers
        print(f"\nSimulating DUAL: {name1} + {name2}")
        reach = simulate_dual_influencer(f1, f2, OVERLAP_RATIO, ALPHA, BETA, N_TRIALS)
        metrics = compute_ad_metrics(reach, CTR_RANGE, CR_RANGE, REVENUE_PER_CONVERSION, CPM)

        pair_summary.append({
            'Influencer 1': name1,
            'Influencer 2': name2,
            'Mean ROI': metrics['ROI'].mean(),
            'Median ROI': metrics['ROI'].median(),
            'Std ROI': metrics['ROI'].std()
        })

        filename = f"results/dual/ROI_dual_{name1.replace(' ', '_')}_{name2.replace(' ', '_')}.png"
        save_roi_histogram(metrics, f"{name1} + {name2}", filename)

    pd.DataFrame(pair_summary).to_csv("results/dual_influencer_summary.csv", index=False)


if __name__ == "__main__":
    main()
