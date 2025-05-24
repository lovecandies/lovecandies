import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp as pl
import seaborn as sns

# 常量定义
ROOMS = 75  # 直播间数量
MAKEUP_ARTISTS = 7  # 化妆师数量
BROADCAST_HOURS = 6.5  # 每位主播直播时长
MAKEUP_MINUTES = 30  # 化妆时长
MIN_MAKEUP_RATIO = 0.6  # 最低化妆率要求
MAX_WAIT_HOURS = 5  # 化妆后最多等待时间
ARTIST_SALARY = 9000  # 化妆师月薪
STREAMER_PROFIT = 4000  # 每个主播月收益


# 将时间转换为时间段索引 (每30分钟一个时段)
def time_to_slot(hour, minute):
    return (hour - 7) * 2 + (minute // 30)


# 将时间段索引转回时间字符串
def slot_to_time(slot):
    hour = 7 + slot // 2
    minute = (slot % 2) * 30
    if hour >= 24:
        hour -= 24
    return f"{hour:02d}:{minute:02d}"


# 计算时间段的结束时间 - 修正计算结束时间
def slot_to_end_time(slot):
    """计算时间段的结束时间（即下一个时段的开始时间）"""
    next_slot = slot + 1
    hour = 7 + next_slot // 2
    minute = (next_slot % 2) * 30
    if hour >= 24:
        hour -= 24
    return f"{hour:02d}:{minute:02d}"


def optimize_schedule():
    """使用整数线性规划优化排班"""
    # 创建优化问题
    prob = pl.LpProblem("MaximizeStreamers", pl.LpMaximize)

    # 定义时间段 (0-41对应7:00-4:00的半小时时段)
    total_slots = 42
    broadcast_slots = int(BROADCAST_HOURS * 2)  # 6.5小时直播 = 13个半小时时段

    print(f"每位主播直播时长: {BROADCAST_HOURS}小时 ({broadcast_slots}个半小时时段)")

    # 有效化妆时段 (7:00-12:00, 13:00-17:00)
    valid_makeup_slots = list(range(0, 10)) + list(range(12, 20))

    # 定义变量
    # X[i,k,s]: 化妆师k在时段i为主播化妆，该主播在时段s开始直播
    X = {}
    for i in valid_makeup_slots:
        for k in range(MAKEUP_ARTISTS):
            earliest_start = max(2, i + 1)  # 至少8:00开始播，且在化妆后
            latest_start = min(i + 2 * MAX_WAIT_HOURS, total_slots - broadcast_slots)  # 必须在化妆后5小时内播

            for s in range(earliest_start, latest_start + 1):
                X[(i, k, s)] = pl.LpVariable(f"X_{i}_{k}_{s}", cat=pl.LpBinary)

    # Y[s]: 一位不需要化妆的主播在时段s开始直播
    Y = {}
    for s in range(2, total_slots - broadcast_slots + 1):
        Y[s] = pl.LpVariable(f"Y_{s}", lowBound=0, cat=pl.LpInteger)

    # 目标函数：最大化主播总数
    prob += pl.lpSum([X[(i, k, s)] for i in valid_makeup_slots
                      for k in range(MAKEUP_ARTISTS)
                      for s in range(max(2, i + 1),
                                     min(i + 2 * MAX_WAIT_HOURS, total_slots - broadcast_slots) + 1)
                      if (i, k, s) in X]) + \
            pl.lpSum([Y[s] for s in range(2, total_slots - broadcast_slots + 1)])

    # 约束1: 每个化妆师在每个时段最多化妆一位主播
    for i in valid_makeup_slots:
        for k in range(MAKEUP_ARTISTS):
            valid_starts = [s for s in range(max(2, i + 1),
                                             min(i + 2 * MAX_WAIT_HOURS, total_slots - broadcast_slots) + 1)
                            if (i, k, s) in X]
            if valid_starts:
                prob += pl.lpSum([X[(i, k, s)] for s in valid_starts]) <= 1

    # 约束2: 每个化妆师每天最多工作18个半小时时段
    for k in range(MAKEUP_ARTISTS):
        valid_pairs = [(i, s) for i in valid_makeup_slots
                       for s in range(max(2, i + 1),
                                      min(i + 2 * MAX_WAIT_HOURS, total_slots - broadcast_slots) + 1)
                       if (i, k, s) in X]
        if valid_pairs:
            prob += pl.lpSum([X[(i, k, s)] for i, s in valid_pairs]) <= 18

    # 约束3: 任何时间点的直播间使用数量不超过总数(75间)
    for t in range(total_slots):
        # 计算在时间段t正在直播的主播数量
        broadcasting_at_t = []

        # 化妆主播
        for i in valid_makeup_slots:
            for k in range(MAKEUP_ARTISTS):
                for s in range(max(2, i + 1),
                               min(i + 2 * MAX_WAIT_HOURS, total_slots - broadcast_slots) + 1):
                    if (i, k, s) in X and s <= t < s + broadcast_slots:
                        broadcasting_at_t.append(X[(i, k, s)])

        # 非化妆主播
        for s in range(2, total_slots - broadcast_slots + 1):
            if s <= t < s + broadcast_slots:
                broadcasting_at_t.append(Y[s])

        if broadcasting_at_t:
            prob += pl.lpSum(broadcasting_at_t) <= ROOMS

    # 约束4: 化妆主播比例不低于60%
    makeup_streamers = pl.lpSum([X[(i, k, s)] for i in valid_makeup_slots
                                 for k in range(MAKEUP_ARTISTS)
                                 for s in range(max(2, i + 1),
                                                min(i + 2 * MAX_WAIT_HOURS, total_slots - broadcast_slots) + 1)
                                 if (i, k, s) in X])

    non_makeup_streamers = pl.lpSum([Y[s] for s in range(2, total_slots - broadcast_slots + 1)])

    prob += makeup_streamers >= MIN_MAKEUP_RATIO * (makeup_streamers + non_makeup_streamers)

    # 求解模型
    print("开始求解最优排班...")
    prob.solve(pl.PULP_CBC_CMD(msg=True, timeLimit=600))

    print(f"求解状态: {pl.LpStatus[prob.status]}")

    # 解析结果
    schedule = []
    streamer_id = 1

    # 处理化妆主播
    for i in valid_makeup_slots:
        for k in range(MAKEUP_ARTISTS):
            for s in range(max(2, i + 1), min(i + 2 * MAX_WAIT_HOURS, total_slots - broadcast_slots) + 1):
                if (i, k, s) in X and pl.value(X[(i, k, s)]) == 1:
                    # 修正: 计算准确的结束时间
                    end_slot = s + broadcast_slots - 1

                    schedule.append({
                        "主播ID": streamer_id,
                        "需要化妆": "是",
                        "化妆师": k + 1,
                        "化妆时间": slot_to_time(i),
                        "直播开始": slot_to_time(s),
                        "直播结束": slot_to_end_time(end_slot),  # 使用新函数计算结束时间
                        "直播开始时段": s,
                        "化妆时段": i
                    })
                    streamer_id += 1

    # 处理非化妆主播
    for s in range(2, total_slots - broadcast_slots + 1):
        if s in Y and pl.value(Y[s]) > 0:
            # 修正: 计算准确的结束时间
            end_slot = s + broadcast_slots - 1

            for _ in range(int(pl.value(Y[s]))):
                schedule.append({
                    "主播ID": streamer_id,
                    "需要化妆": "否",
                    "化妆师": "-",
                    "化妆时间": "-",
                    "直播开始": slot_to_time(s),
                    "直播结束": slot_to_end_time(end_slot),  # 使用新函数计算结束时间
                    "直播开始时段": s,
                    "化妆时段": -1
                })
                streamer_id += 1

    # 创建排班表DataFrame
    schedule_df = pd.DataFrame(schedule)

    return schedule_df, prob.objective.value()


def create_makeup_artist_schedule(schedule_df):
    """创建化妆师排班表"""
    # 创建每个化妆师的工作安排表
    artist_schedule = {}

    # 时间段标签
    time_slots = [slot_to_time(i) for i in range(42)]

    # 初始化每个化妆师的工作状态为空闲
    for artist_id in range(1, MAKEUP_ARTISTS + 1):
        artist_schedule[artist_id] = ["空闲"] * 42

    # 填充化妆师工作安排
    makeup_streamers = schedule_df[schedule_df["需要化妆"] == "是"]
    for _, row in makeup_streamers.iterrows():
        artist_id = row["化妆师"]
        makeup_slot = row["化妆时段"]

        # 只处理有效的化妆师ID和时段
        if artist_id != "-" and isinstance(makeup_slot, (int, np.integer)) and 0 <= makeup_slot < 42:
            streamer_id = row["主播ID"]
            artist_schedule[artist_id][makeup_slot] = f"主播{streamer_id}化妆"

    # 转换为DataFrame
    artist_df = pd.DataFrame(artist_schedule, index=time_slots)

    # 转置DataFrame使化妆师成为行，时间成为列
    artist_df = artist_df.transpose()

    return artist_df


def visualize_makeup_artist_schedule(artist_schedule_df):
    """可视化化妆师排班表 (替代热力图)"""
    plt.figure(figsize=(18, 10))

    # 设置颜色
    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    # 获取有效的工作时段 (7:00-12:00, 13:00-17:00)
    valid_slots = list(range(0, 10)) + list(range(12, 20))

    # 标记午休时间
    lunch_start = 10
    lunch_end = 12

    # 为每个化妆师绘制工作时段
    for artist_id in range(1, MAKEUP_ARTISTS + 1):
        artist_slots = []
        streamer_ids = []

        y_pos = MAKEUP_ARTISTS - artist_id + 1  # 反转Y轴顺序

        # 找出该化妆师的工作时段
        for i, value in enumerate(artist_schedule_df.loc[artist_id]):
            if value != "空闲" and i in valid_slots:
                artist_slots.append(i)
                # 提取主播ID
                streamer_id = ''.join(filter(str.isdigit, value))
                streamer_ids.append(streamer_id)

        # 绘制工作时段
        for i, slot in enumerate(artist_slots):
            plt.plot([slot, slot + 1], [y_pos, y_pos], linewidth=8,
                     color=colors[artist_id % len(colors)], solid_capstyle='butt')

            # 显示主播ID
            plt.text(slot + 0.5, y_pos + 0.15, streamer_ids[i],
                     horizontalalignment='center', fontsize=8)

    # 添加午休时间标记
    plt.axvspan(lunch_start, lunch_end, alpha=0.2, color='gray', label='Lunch Break')

    # 设置X轴标签 (每半小时一个)
    time_labels = [slot_to_time(i) for i in range(0, 42, 2)]
    plt.xticks(range(0, 42, 2), time_labels, rotation=45)

    # 设置Y轴标签
    plt.yticks(range(1, MAKEUP_ARTISTS + 1),
               [f'Makeup Artist {i}' for i in range(MAKEUP_ARTISTS, 0, -1)])

    # 设置图表标题和标签
    plt.title('Makeup Artists Working Schedule', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.xlim(0, 42)
    plt.ylim(0.5, MAKEUP_ARTISTS + 0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 保存图表
    plt.tight_layout()
    plt.savefig('makeup_artist_schedule_gantt.png', dpi=300)

    return


def create_visualization(schedule_df):
    """创建排班可视化图表 (英文版避免中文显示问题)"""
    # 计算统计数据
    total = len(schedule_df)
    makeup = len(schedule_df[schedule_df["需要化妆"] == "是"])
    non_makeup = total - makeup
    makeup_rate = makeup / total
    profit = STREAMER_PROFIT * total - MAKEUP_ARTISTS * ARTIST_SALARY

    # 使用纯英文显示
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # 图表1: 直播间使用情况
    plt.figure(figsize=(15, 8))

    # 计算每个时段的直播间使用数量
    room_usage = np.zeros(42)
    broadcast_slots = int(BROADCAST_HOURS * 2)

    for _, row in schedule_df.iterrows():
        start_slot = row["直播开始时段"]
        for t in range(start_slot, start_slot + broadcast_slots):
            if t < 42:
                room_usage[t] += 1

    # 绘制使用情况
    time_labels = [slot_to_time(i) for i in range(0, 42, 2)]
    x_ticks = list(range(0, 42, 2))

    colors = []
    for usage in room_usage:
        if usage > ROOMS * 0.9:
            colors.append('#e63946')  # 红色 (高负载)
        elif usage > ROOMS * 0.7:
            colors.append('#457b9d')  # 蓝色 (中负载)
        else:
            colors.append('#a8dadc')  # 浅色 (低负载)

    # 绘制柱状图
    bars = plt.bar(range(42), room_usage, color=colors, width=0.8)

    # ===== 修正：确保所有柱子都有数值标签 =====
    # 创建两组标签，交错显示以避免拥挤
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:  # 只为有值的柱子添加标签
            # 为了避免标签重叠，偶数和奇数索引的标签位置稍有不同
            if i % 2 == 0:
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{int(height)}',
                         ha='center', va='bottom',
                         fontsize=8, fontweight='bold' if height > ROOMS * 0.7 else 'normal',
                         color='black')
            else:
                # 奇数索引的标签位置略微上移
                plt.text(bar.get_x() + bar.get_width() / 2., height + 1.5,
                         f'{int(height)}',
                         ha='center', va='bottom',
                         fontsize=8, fontweight='bold' if height > ROOMS * 0.7 else 'normal',
                         color='black')

    # 标记最大容量75 - 在图表上方添加水平线和标签
    plt.axhline(y=ROOMS, color='red', linestyle='--', linewidth=2,
                label=f'Room Limit ({ROOMS})')

    # 在多个位置重复显示75标记，确保可见
    for x in [5, 15, 25, 35]:
        plt.text(x, ROOMS + 2, f'{ROOMS}',
                 color='red', fontweight='bold', fontsize=10)

    plt.title('Broadcasting Room Usage Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Number of Rooms in Use', fontsize=14)
    plt.xticks(x_ticks, time_labels, rotation=45)
    plt.ylim(0, ROOMS * 1.2)  # 扩大Y轴范围，确保能看到所有标签
    plt.legend()

    # 保存第一张图
    plt.tight_layout()
    plt.savefig('room_usage.png', dpi=300)
    plt.savefig('room_usage.pdf')  # 同时保存PDF版本，更高质量

    # 创建并保存化妆师排班表CSV
    artist_schedule_df = create_makeup_artist_schedule(schedule_df)
    artist_schedule_df.to_csv('化妆师排班表.csv', encoding='utf-8-sig')

    # 可视化化妆师排班表 (替代热力图)
    visualize_makeup_artist_schedule(artist_schedule_df)

    # 另一种显示方式：绘制简化版的柱状图，只显示重要时间点
    plt.figure(figsize=(12, 6))
    # 只选择重要时间点(每小时)
    important_slots = list(range(0, 42, 2))
    important_labels = [slot_to_time(i) for i in important_slots]
    important_values = [room_usage[i] for i in important_slots]

    # 绘制简化版柱状图
    bars2 = plt.bar(range(len(important_slots)), important_values, color='steelblue')

    # 确保为每个柱子添加标签
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{int(height)}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    plt.axhline(y=ROOMS, color='red', linestyle='--', linewidth=2)
    plt.text(len(important_slots) - 3, ROOMS + 2, f'Max: {ROOMS}', color='red', fontweight='bold')

    plt.title('Room Usage (Hourly View)', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Rooms in Use', fontsize=14)
    plt.xticks(range(len(important_slots)), important_labels, rotation=45)
    plt.ylim(0, ROOMS * 1.2)

    plt.tight_layout()
    plt.savefig('room_usage_hourly.png', dpi=300)

    # 图表3: 总结信息
    plt.figure(figsize=(10, 6))

    # 创建基本信息表格
    data = [
        ["Total Streamers", total],
        ["Makeup Streamers", makeup],
        ["Non-Makeup Streamers", non_makeup],
        ["Makeup Rate", f"{makeup_rate:.2%}"],
        ["Maximum Room Usage", f"{max(room_usage)} ({max(room_usage) / ROOMS:.2%})"],
        ["Total Profit", f"{profit} CNY/month"]
    ]

    # 隐藏坐标轴
    plt.axis('off')

    # 创建表格
    table = plt.table(cellText=data, colLabels=["Metric", "Value"],
                      loc='center', cellLoc='center', colColours=['#f1faee', '#f1faee'])

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2)

    plt.title('Optimization Results Summary', fontsize=18, pad=20)

    # 保存结果摘要
    plt.tight_layout()
    plt.savefig('optimization_summary.png', dpi=300)

    print("已生成可视化图表和排班表：")
    print("1. room_usage.png/pdf - 直播间使用情况 (所有时段)")
    print("2. room_usage_hourly.png - 直播间使用情况 (每小时视图)")
    print("3. makeup_artist_schedule_gantt.png - 化妆师工作安排甘特图")
    print("4. optimization_summary.png - 优化结果摘要")
    print("5. 化妆师排班表.csv - 化妆师详细排班表")
    print("6. 共享直播间排班表.csv - 主播排班表")

    # 返回统计数据
    return {
        "总主播数": total,
        "化妆主播": makeup,
        "不化妆主播": non_makeup,
        "化妆率": makeup_rate,
        "总收益": profit,
        "最大直播间使用数": max(room_usage)
    }


def analyze_schedule(schedule_df):
    """分析排班表"""
    # 1. 检查化妆师工作量分布
    if "需要化妆" in schedule_df.columns and "化妆师" in schedule_df.columns:
        makeup_df = schedule_df[schedule_df["需要化妆"] == "是"]
        artist_workload = makeup_df["化妆师"].value_counts().sort_index()

        print("\n化妆师工作量分布:")
        for artist, count in artist_workload.items():
            if artist != "-":
                print(f"化妆师{artist}: {count}位主播")

        # 检查是否存在化妆师超出工作时间限制
        if max(artist_workload.values) > 18:
            print(f"警告: 有化妆师超出工作时间限制! 最大工作量: {max(artist_workload.values)}次化妆")

    # 2. 分析各时段开始直播的主播数量
    start_slots = schedule_df["直播开始时段"].value_counts().sort_index()

    print("\n各时段开始直播的主播数量:")
    for slot, count in start_slots.items():
        print(f"{slot_to_time(slot)}: {count}位主播开始直播")

    # 3. 分析化妆后等待时间
    if "需要化妆" in schedule_df.columns and "化妆时段" in schedule_df.columns and "直播开始时段" in schedule_df.columns:
        makeup_df = schedule_df[schedule_df["需要化妆"] == "是"].copy()
        makeup_df["等待时段"] = makeup_df["直播开始时段"] - makeup_df["化妆时段"] - 1
        makeup_df["等待时间"] = makeup_df["等待时段"] / 2  # 将时段转为小时

        avg_wait = makeup_df["等待时间"].mean()
        max_wait = makeup_df["等待时间"].max()

        print(f"\n化妆主播平均等待时间: {avg_wait:.2f}小时")
        print(f"化妆主播最长等待时间: {max_wait:.2f}小时")

        # 等待时间分布
        wait_dist = makeup_df["等待时间"].value_counts().sort_index()

        print("\n等待时间分布:")
        for hours, count in wait_dist.items():
            print(f"等待{hours:.1f}小时: {count}位主播")

    # 4. 验证直播时长
    print("\n验证直播时长:")
    for _, row in schedule_df.head(10).iterrows():  # 增加到10个验证样本
        start_time = row["直播开始"]
        end_time = row["直播结束"]
        # 解析时间字符串
        start_h, start_m = map(int, start_time.split(':'))
        end_h, end_m = map(int, end_time.split(':'))

        # 处理跨天的情况
        if end_h < start_h or (end_h == start_h and end_m < start_m):
            end_h += 24

        duration_minutes = (end_h - start_h) * 60 + (end_m - start_m)
        duration_hours = duration_minutes / 60

        print(f"主播{row['主播ID']}: {start_time}-{end_time} ({duration_hours:.1f}小时)")


# 执行优化并输出结果
if __name__ == "__main__":
    # 1. 运行优化算法
    print("开始运行直播间排班优化算法...")
    schedule_df, objective_value = optimize_schedule()

    # 2. 输出排班表
    print("\n生成排班表完成!")
    print(f"最优解值: {objective_value}")
    print(f"排班表前10行:")
    print(schedule_df[["主播ID", "需要化妆", "化妆师", "化妆时间", "直播开始", "直播结束"]].head(10))

    # 3. 保存排班表到CSV
    schedule_df.to_csv('共享直播间排班表.csv', index=False, encoding='utf-8-sig')
    print("\n完整排班表已保存到'共享直播间排班表.csv'")

    # 4. 创建可视化图表
    print("\n生成排班表可视化...")
    stats = create_visualization(schedule_df)

    # 5. 输出统计信息
    print("\n优化结果统计:")
    print(f"总主播数: {stats['总主播数']}人")
    print(f"化妆主播: {stats['化妆主播']}人")
    print(f"不化妆主播: {stats['不化妆主播']}人")
    print(f"化妆率: {stats['化妆率']:.2%}")
    print(f"总收益: {stats['总收益']}元/月")
    print(f"最大直播间使用数: {stats['最大直播间使用数']}间 ({stats['最大直播间使用数'] / ROOMS:.2%})")

    # 6. 分析排班表
    print("\n排班表详细分析:")
    analyze_schedule(schedule_df)

    # 7. 输出结论
    print("\n最终结论:")
    print(f"在满足所有约束条件的情况下，最大可安排主播数量为: {stats['总主播数']}人")
    print(f"最大总收益为: {stats['总收益']}元/月")
    print("排班表已保存到'共享直播间排班表.csv'")
    print("化妆师排班表已保存到'化妆师排班表.csv'")
    print("可视化图表已保存为多个文件")
