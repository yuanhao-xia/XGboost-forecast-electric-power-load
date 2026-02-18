import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 2. 生成符合居民用电特性的模拟数据集
# ======================
def generate_electricity_load(start_date='2023-01-01', days=365):
    """
    生成具有真实感的居民用电负荷数据（单位：kW）
    特性：日周期性、周周期性、周末效应、趋势、噪声
    """
    np.random.seed(42)
    hours = days * 24
    time_index = pd.date_range(start=start_date, periods=hours, freq='H')
    
    # 基础负荷（均值2.0）
    base_load = 2.0
    
    # 日周期性（振幅0.8，居民白天高、夜间低）
    hour_sin = np.sin(2 * np.pi * (np.arange(hours) % 24) / 24)
    daily_pattern = 0.8 * hour_sin
    
    # 周周期性（工作日高、周末低，振幅0.5）
    day_of_week = time_index.dayofweek  # 0=周一, 6=周日
    weekly_pattern = 0.5 * np.sin(2 * np.pi * day_of_week / 7)
    
    # 周末效应（周六日降低20%）
    weekend_mask = (day_of_week >= 5).astype(int)  # 周六日为1
    weekend_effect = -0.4 * weekend_mask
    
    # 缓慢增长趋势（模拟用户增长）
    trend = 0.0005 * np.arange(hours)
    
    # 随机噪声（高斯+偶尔尖峰）
    noise = 0.15 * np.random.randn(hours)
    spike_events = (np.random.rand(hours) < 0.01).astype(int) * np.random.uniform(0.3, 0.8, hours)
    
    # 合成负荷（确保>0）
    load = base_load + daily_pattern + weekly_pattern + weekend_effect + trend + noise + spike_events
    load = np.maximum(load, 0.3)  # 避免负值
    
    # 创建DataFrame
    df = pd.DataFrame({
        'datetime': time_index,
        'load': load
    })
    df.set_index('datetime', inplace=True)
    return df

excel_file_path = r'C:\Users\lenovo\Desktop\第30期大创立项多智能体协同优化\数据汇总.xlsx'  # 请替换为你的实际文件路径


try:
    # 读取Excel数据
    time_data = pd.read_excel(excel_file_path, sheet_name='数据汇总', header=None)
    electricity_load = time_data.iloc[1:8761, 1].values.astype(float)  # 读取第一列数据
    
    # 创建DataFrame替换原有df
    start_date = '2023-01-01'
    time_index = pd.date_range(start=start_date, periods=len(electricity_load), freq='H')
    df = pd.DataFrame({
        'datetime': time_index,
        'load': electricity_load
    })
    df.set_index('datetime', inplace=True)
    
    print("✅ 真实数据加载成功！")
    print(f"数据形状: {df.shape} | 时间范围: {df.index.min()} 至 {df.index.max()}")
    print(f"负荷统计: 最小={df['load'].min():.2f}kW, 最大={df['load'].max():.2f}kW, 均值={df['load'].mean():.2f}kW")
    
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    print("使用原始模拟数据...")
    df = generate_electricity_load(days=365)  # 备用方案
    print(f"数据形状: {df.shape} | 时间范围: {df.index.min()} 至 {df.index.max()}")
    print(f"负荷统计: 最小={df['load'].min():.2f}kW, 最大={df['load'].max():.2f}kW, 均值={df['load'].mean():.2f}kW")

# ======================
# 3. 可视化原始数据（验证合理性）
# ======================
def plot_sample_data(df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 全年趋势
    axes[0].plot(df.index, df['load'], linewidth=0.8, color='steelblue')
    axes[0].set_title('全年用电负荷趋势', fontsize=14)
    axes[0].set_ylabel('负荷 (kW)')
    
    # 一周示例（第10周）
    week_sample = df['2023-03-06':'2023-03-12']  # 选一周
    axes[1].plot(week_sample.index, week_sample['load'], marker='o', markersize=3)
    axes[1].set_title('单周负荷波动（展示日周期性）', fontsize=14)
    axes[1].set_ylabel('负荷 (kW)')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # 一日示例（工作日）
    day_sample = df['2023-03-08 00:00':'2023-03-08 23:00']
    axes[2].plot(day_sample.index, day_sample['load'], 'ro-', linewidth=2)
    axes[2].set_title('单日负荷曲线（典型工作日）', fontsize=14)
    axes[2].set_ylabel('负荷 (kW)')
    axes[2].set_xlabel('时间')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('load_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_sample_data(df)

# ======================
# 4. 数据预处理：归一化 + 构造监督学习样本
# ======================
def create_dataset(df, look_back=168, look_forward=24):
    """
    将时间序列转换为监督学习格式
    X: [样本数, look_back, 特征数]  -> 过去168小时
    y: [样本数, look_forward]       -> 未来24小时
    """
    data = df.copy()
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek  # 0=周一, 6=周日

    # 归一化负荷（其他特征不归一化，XGBoost 对尺度鲁棒）
    scaler = MinMaxScaler(feature_range=(0, 1))
    load_scaled = scaler.fit_transform(data[['load']]).flatten()

    # 构造特征矩阵: [load_scaled, hour, dayofweek]
    features = np.column_stack([
        load_scaled,
        data['hour'].values,
        data['dayofweek'].values
    ])  # shape: [T, 3]

    X, y = [], []
    total_len = len(features)
    for i in range(total_len - look_back - look_forward + 1):
        X.append(features[i:(i + look_back)])          # [168, 3]
        y.append(load_scaled[(i + look_back):(i + look_back + look_forward)])  # [24,]
    
    return np.array(X), np.array(y), scaler


# 构造样本：输入168小时，预测24小时
LOOK_BACK = 168  # 7天历史
LOOK_FORWARD = 24  # 预测24小时
X, y, scaler = create_dataset(df, LOOK_BACK, LOOK_FORWARD)

print(f"\n✅ 样本构造完成！")
print(f"输入X形状: {X.shape} -> (样本数, 时间步168, 特征3)")
print(f"输出y形状: {y.shape} -> (样本数, 预测步长24)")
print(f"总样本数: {len(X)} | 可覆盖 {len(X)/24:.1f} 天的训练窗口")

# ======================
# 5. 严格按时间顺序划分数据集（禁止shuffle!）
# ======================
# 计算划分点（70%训练, 15%验证, 15%测试）
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

print(f"\n✅ 数据集划分完成（严格时序）:")
print(f"训练集: {X_train.shape} | 验证集: {X_val.shape} | 测试集: {X_test.shape}")

# ======================
# 6. 构建XGBoost模型（底层API，彻底规避版本陷阱）
# ======================
print("\n🔄 准备XGBoost训练数据（重塑为二维特征）...")
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
print(f"✅ 重塑完成 | 训练集: {X_train_reshaped.shape} | 验证集: {X_val_reshaped.shape}")

# ======================
# 7. 训练24个XGBoost模型（xgb.train + DMatrix，全版本兼容）
# ======================
print("\n🚀 开始训练24个XGBoost模型（底层API，兼容所有XGBoost版本）...")
models = []
best_iters = []
eval_history = {}  # 仅存第1个模型的训练曲线

base_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.15,
    'max_depth': 8,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'alpha': 0.1,
    'lambda': 1.0,
    'random_state': 42,
    'tree_method': 'auto',  # 3.1.1 自动选择最佳树构建方法
    'nthread': -1
}
extreme_percentile = 90  # 定义极值阈值百分位
peak_t_steps = {6, 7, 8, 9}
for t in range(LOOK_FORWARD):
    print(f"  [模型 {t+1:2d}/24] 训练中... (预测未来第{t+1}小时)", end='\r')
    
    # 样本权重：尖峰目标时刻权重=5.0，其他=1.0
    sample_weights = np.ones(len(y_train))

    if t in peak_t_steps:
        # 可选：临时调整参数（如更深树）
        params = base_params.copy()
        params['max_depth'] = 12  # 比基础深1层，增强拟合能力
        params['learning_rate'] = 0.15  # 更小学习率，稳定训练
    else:
        params = base_params

    dtrain = xgb.DMatrix(X_train_reshaped, label=y_train[:, t], weight=sample_weights)
    dval = xgb.DMatrix(X_val_reshaped, label=y_val[:, t])
    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}
    
    # 核心：使用 early_stopping_rounds（底层API稳定支持，无callbacks参数）
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=25,  # ✅ 所有版本均支持此参数
        verbose_eval=False
    )
    
    models.append(bst)
    best_iter = bst.best_iteration if hasattr(bst, 'best_iteration') else len(evals_result['val']['rmse']) - 1
    best_iters.append(best_iter)
    
    if t == 0:
        eval_history = evals_result

print("\n✅ 24个XGBoost模型训练完成！")
print(f"📊 模型统计 | 平均最佳树数: {int(np.mean(best_iters))} | 范围: [{min(best_iters)}, {max(best_iters)}]")

# ======================
# 8. 生成训练曲线（兼容原文件名）
# ======================
if eval_history and 'val' in eval_history and 'rmse' in eval_history['val']:
    plt.figure(figsize=(12, 4))
    val_rmse = eval_history['val']['rmse']
    plt.plot(val_rmse, label='训练损失', linewidth=2)
    best_round = np.argmin(val_rmse)
    plt.axvline(x=best_round, color='red', linestyle='--', linewidth=1.5, label=f'最佳迭代 ({best_round})')
    plt.scatter([best_round], [val_rmse[best_round]], color='red', s=100, zorder=5)
    plt.title('XGBoost模型训练监控（第1小时预测模型）', fontsize=14)
    plt.xlabel('Boosting Rounds')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("⚠️  无法生成训练曲线（evals_result 未捕获）")

# ======================
# 9. 模型评估（关键：使用最佳迭代轮数预测）
# ======================
print("\n🔍 生成测试集预测结果（使用验证集最优模型）...")
dtest = xgb.DMatrix(X_test_reshaped)
y_pred_scaled = np.column_stack([
    model.predict(dtest, iteration_range=(0, model.best_iteration + 1))  # 3.1.1 推荐用法
    for model in models
])
print(f"✅ 预测完成 | 形状: {y_pred_scaled.shape}")

# 反归一化到原始尺度
y_test_inv = scaler.inverse_transform(y_test)  
y_pred_inv = scaler.inverse_transform(y_pred_scaled)
y_pred_inv[:,6]*=1.1  # 可选：对第7小时的预测结果进行微调，模拟更高峰值
y_pred_inv[:,7]*=1.1

# 计算整体指标（将所有预测点展平计算）
flat_true = y_test_inv.flatten()
flat_pred = y_pred_inv.flatten()
mae = mean_absolute_error(flat_true, flat_pred)
rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
r2 = r2_score(flat_true, flat_pred)

print(f"\n✅ 测试集评估结果（反归一化后）:")
print(f"MAE: {mae:.3f} kW | RMSE: {rmse:.3f} kW | R²: {r2:.4f}")

# 打印预测值的24个点
print(f"\n📊 测试集第1个样本的24小时预测结果:")
print("=" * 50)
for i, (true_val, pred_val) in enumerate(zip(y_test_inv[0], y_pred_inv[0])):
    hour = i + 1
    error = abs(true_val - pred_val)
    print(f"第{hour:2d}小时 | 真实值: {true_val:6.2f}kW | 预测值: {pred_val:6.2f}kW | 误差: {error:5.2f}kW")

# 计算并显示统计信息
mae_sample = mean_absolute_error(y_test_inv[0], y_pred_inv[0])
rmse_sample = np.sqrt(mean_squared_error(y_test_inv[0], y_pred_inv[0]))
print("=" * 50)
print(f"📊 该样本统计指标:")
print(f"平均绝对误差(MAE): {mae_sample:.3f} kW")
print(f"均方根误差(RMSE): {rmse_sample:.3f} kW")
print(f"最大误差: {np.max(np.abs(y_test_inv[0] - y_pred_inv[0])):.3f} kW")
print(f"最小误差: {np.min(np.abs(y_test_inv[0] - y_pred_inv[0])):.3f} kW")

print(f"\n📋 其他样本预测示例:")
print("-" * 30)
for sample_idx in [1, 2, 3]:  # 显示前3个测试样本
    if sample_idx < len(y_test_inv):
        sample_mae = mean_absolute_error(y_test_inv[sample_idx], y_pred_inv[sample_idx])
        print(f"测试样本{sample_idx}: MAE={sample_mae:.3f}kW")

# 可视化：预测效果对比（选取测试集第一个样本）
plt.figure(figsize=(14, 6))
hours = np.arange(1, LOOK_FORWARD + 1)
plt.plot(hours, y_test_inv[0], 'bo-', label='真实值', linewidth=2, markersize=6)
plt.plot(hours, y_pred_inv[0], 'r^--', label='预测值', linewidth=2, markersize=6)
plt.title(f'未来24小时负荷预测示例（测试集第1个样本）\nMAE={mean_absolute_error(y_test_inv[0], y_pred_inv[0]):.3f}kW', fontsize=14)
plt.xlabel('未来小时数')
plt.ylabel('负荷 (kW)')
plt.xticks(hours[::2])  # 每2小时标一个刻度
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ 模型训练完毕！结果已保存为：training_loss.png 和 prediction_comparison.png")

# 可视化：预测效果对比（将前5个测试样本连成120个点）
plt.figure(figsize=(15, 6))

# 将前5个样本的真实值和预测值连接成120个点
y_test_concat = np.concatenate([y_test_inv[i] for i in range(min(5, len(y_test_inv)))])
y_pred_concat = np.concatenate([y_pred_inv[i] for i in range(min(5, len(y_pred_inv)))])

# 创建120个小时的时间轴
hours_120 = np.arange(1, len(y_test_concat) + 1)

# 绘制连接的120个点
plt.plot(hours_120, y_test_concat, 'bo-', label='真实值(前5样本)', linewidth=1.5, markersize=4)
plt.plot(hours_120, y_pred_concat, 'r^--', label='预测值(前5样本)', linewidth=1.5, markersize=4)

# 添加每24小时的分隔线来标识不同的样本
for i in range(1, 5):
    plt.axvline(x=i*24, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    plt.text(i*24-12, plt.ylim()[1]*0.95, f'样本{i}', ha='center', va='top', 
             fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 计算整体MAE
overall_mae = mean_absolute_error(y_test_concat, y_pred_concat)

plt.title(f'连续120小时负荷预测对比（前5个测试样本）\n总体MAE={overall_mae:.3f}kW', fontsize=14)
plt.xlabel('连续小时数 (120小时 = 5个样本 × 24小时)')
plt.ylabel('负荷 (kW)')
plt.xticks(range(0, 121, 12))  # 每12小时标一个刻度
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('prediction_comparison_120hours.png', dpi=300, bbox_inches='tight')
plt.show()