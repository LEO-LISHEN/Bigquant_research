# factor_lib

`factor_lib` 用于把分散在研究 Notebook 中的因子、预处理、评价和策略代码，沉淀为可发现、可复用、可检查、可追溯的研究组件。

当前主要数据和运行平台是 BigQuant。本项目所说的“生产化”是研究工程生产化，不代表已经满足实盘系统在稳定性、监控、容灾、权限和资金安全方面的全部要求。

## 1. 当前架构原则

1. 因子函数只负责计算因子值。
2. 因子和公共预处理使用数据源无关的语义字段。
3. 平台表名、源字段和查询接口只存在于对应平台的数据适配器。
4. `FACTOR` 只声明需要哪些语义字段，不声明字段属于哪个适配器、数据域或数据表。
5. loader 根据各适配器的 `ADAPTER_SPEC` 自动完成字段路由。
6. 不同主键粒度的数据保存在 `FactorDataBundle` 的不同数据域中，不进行无意义广播。
7. 评价和策略公开入口接收因子名称及因子参数，不要求用户提前传入 `factor_data`。
8. 评价和策略的编排代码可以依赖当前研究平台；因子公式和公共预处理不能依赖平台。
9. 可以通过参数表达的同类因子只保留一个参数化脚本。
10. `target_dates`、数据准备日期和 `as_of_date` 必须严格区分。
11. 无效值、预热不足和无法计算的结果保留 `NaN`，不得为了表面完整统一填为 0。
12. 本地检查通过不等于 BigQuant 真实查询或回测已经通过。

## 2. 大类目录

```text
factor_lib/
├── README.md
├── LLM_CONTEXT.md
├── Factor Repository/
├── common/
│   ├── preprocess/
│   ├── data_adapters/
│   ├── factor_evaluation/
│   └── strategies/
└── factor_hub/
```

| 目录 | 当前职责 |
|---|---|
| `Factor Repository/` | 存放参数化因子脚本以及模块内的 `FACTOR` 元数据。 |
| `common/preprocess/` | 存放不读取数据源的通用去极值、标准化和中性化函数。 |
| `common/data_adapters/` | 存放不同平台的数据映射、适配器、loader 和分粒度数据容器。 |
| `common/factor_evaluation/` | 当前存放自动取数、计算因子、构造标签、计算指标和绘图的评价入口。 |
| `common/strategies/` | 当前存放依赖 BigTrader 执行的策略回测入口。 |
| `factor_hub/` | 动态发现、列表、搜索、说明和调用因子。 |

`factor_evaluation` 和 `strategies` 当前确实包含 BigQuant 平台编排代码。以后如调整目录，可以把平台运行器移到单独的平台目录，但不应仅为了追求形式上的“common”而阻碍当前可用性。

## 3. 总体调用链

```text
评价或策略公开入口
        ↓
factor_hub 动态读取 FACTOR
        ↓
loader 合并默认参数与本次 factor_params
        ↓
解析动态 data_window 和条件字段
        ↓
读取各适配器的 ADAPTER_SPEC 字段目录
        ↓
daily / financial / market_daily 等适配器拉取原始数据
        ↓
FactorDataBundle 按原始主键粒度保存各数据域
        ↓
get_factor 按因子名称调用计算函数
        ↓
因子仅在 target_dates 上返回标准因子面板
        ↓
评价指标、组合构建或 BigTrader 执行
```

策略和评价函数负责决定需要哪些日期；loader 只根据收到的连续区间或日期列表取数，不负责生成调仓计划、评价频率或信号日。

## 4. 语义标准字段

因子脚本使用稳定的业务语义名称，例如：

```text
date
instrument
close
turn
pb
total_market_cap
industry
quarterly_net_profit_yoy
market_close
```

语义字段必须满足：

- 同一业务含义使用同一个名称；
- 名称不携带平台表名；
- 含义足够精确，例如价格复权口径在整个窗口内必须一致；
- 财务字段说明真实可得时点；
- 市场共享字段不伪装成逐股票字段。

`FACTOR` 中不写以下信息：

- `frequency`；
- `data_domain`；
- BigQuant 表名；
- BigQuant 源字段名；
- SQL；
- 查询优先级。

这些信息由平台适配器负责。

## 5. 数据适配器与 ADAPTER_SPEC

每个平台可以按频率、数据域或表结构继续拆分多个适配器。BigQuant 当前包括日频股票、点时财务和日频市场指数适配器。

每个适配器公开一个 `ADAPTER_SPEC`：

```python
ADAPTER_SPEC = {
    "name": "daily",
    "output_group": "security_daily",
    "key_columns": ("date", "instrument"),
    "supported_fields": tuple(FIELD_MAPPING),
    "context_parameters": (),
}
```

字段含义：

- `name`：适配器注册名称；
- `output_group`：适配器输出进入哪个数据域；
- `key_columns`：该输出的唯一主键；
- `supported_fields`：适配器能够提供的语义字段；
- `context_parameters`：取数时还需要从因子参数中取得的上下文，例如 `market_index`。

loader 会把所有适配器的 `supported_fields` 组成字段目录。一个语义字段在同一平台只能由一个适配器声明，否则无法自动路由。

适配器允许执行：

- 按连续日期区间或离散日期列表查询；
- 按证券或指数范围过滤；
- 把平台字段重命名为语义字段；
- 按平台规则完成必要的点时对齐；
- 验证主键、重复记录和返回字段。

适配器不得执行：

- 因子公式；
- 因子专属缺失值填补；
- 因子去极值、标准化和中性化；
- 标签构造；
- 市值分组和选股；
- 回测或订单逻辑。

日期选择必须二选一：

```python
start_date="2022-01-01",
end_date="2024-12-31",
```

或：

```python
dates=["2022-01-04", "2022-02-08"],
```

不能同时使用两种方式。

## 6. FactorDataBundle

`FactorDataBundle` 用于保存不同主键粒度的原始数据面板。

当前典型数据域：

```text
security_daily: date + instrument
market_daily:   date + market_index
```

这样可以避免把同一个市场指数值复制到当天每一只股票行上。

常用接口：

```python
bundle.domain_names
bundle.get_domain("market_daily")
bundle.get_security_daily()
bundle.row_counts()
bundle.missing_dates("market_daily", dates)
bundle.select_dates(dates)
bundle.with_domain("security_daily", panel)
```

容器不查询数据、不计算因子，也不跨粒度自动合并。

## 7. loader

BigQuant loader 的核心入口是：

```python
load_factor_raw_data(
    factor_name,
    start_date=None,
    end_date=None,
    dates=None,
    factor_params=None,
    instruments=None,
    adapter_overrides=None,
    show_progress=False,
)
```

loader 的工作顺序：

1. 动态取得因子的 `FACTOR`；
2. 合并 `parameters` 默认值与调用方传入的 `factor_params`；
3. 根据条件字段的 `required_when` 决定本次真实所需字段；
4. 调用 `data_window.resolver` 解析本次预热窗口；
5. 根据 `ADAPTER_SPEC` 字段目录路由字段；
6. 从同一份已解析参数中取得适配器上下文；
7. 调用适配器；
8. 只合并相同主键粒度的面板；
9. 返回 `FactorDataBundle`。

loader 不决定研究频率、调仓频率、目标日期、信号日期或执行日期。

## 8. 因子函数接口

推荐接口：

```python
def calc_factor_name(
    data,
    target_dates=None,
    as_of_date=None,
    show_progress=False,
    progress_every=20,
    **factor_params,
):
    ...
```

参数说明：

- `data`：语义字段组成的股票面板，可以包含目标日以前的预热数据；
- `target_dates`：实际需要输出因子值的截面日期；
- `as_of_date`：本次计算允许使用信息的全局截止日；
- `factor_params`：因子窗口、最小观测数、中性化开关等参数；
- `show_progress`：是否显示进度；
- `progress_every`：有循环时的刷新间隔。

`target_dates` 和数据中的所有日期不是同一概念。数据可以包含预热日期，但结果只能包含目标日期。

需要市场指数等共享数据的因子，可以额外声明内部参数：

```python
domain_data=None
```

调用者不需要手工传入。`get_factor()` 发现输入是 `FactorDataBundle` 且函数声明了 `domain_data` 后，会自动传入完整容器，同时把 `security_daily` DataFrame 作为 `data`。

## 9. 因子输出

所有因子统一返回：

```text
date | instrument | factor_name
```

要求：

- 因子值列名与 `FACTOR["name"]` 一致；
- 一行对应一个目标日期和一只证券；
- `date + instrument` 不允许重复；
- 不输出预热日期；
- 无效值和无法计算值保留 `NaN`；
- 不返回标签、分组、收益率、持仓或图表；
- 按 `date`、`instrument` 稳定排序。

## 10. FACTOR 元数据规范

每个因子脚本必须在模块顶层声明 `FACTOR`。

### 10.1 核心必填字段

| 字段 | 含义 |
|---|---|
| `name` | 因子唯一名称，也必须是输出因子列名。 |
| `func` | 因子计算函数对象。 |
| `category` | 因子类别。 |
| `direction` | 经验方向：`1` 通常表示高值较优，`-1` 通常表示低值较优。 |
| `description` | 简要经济含义和处理流程。 |
| `formula` | 明确公式、变量和处理顺序。 |
| `input_schema` | 必需字段与条件字段的语义说明。 |
| `parameters` | 全部公开参数的默认值、合法值和影响。 |
| `data_window` | 固定或动态预热需求。 |
| `output_schema` | 输出列、类型和解释。 |
| `usage_notes` | 适用方式和限制。 |
| `pit_notes` | 可得时点和潜在前视风险。 |
| `status` | 当前研究状态。 |
| `version` | 因子定义版本。 |

可选扩展字段：

```text
best_practice
references
research_findings
tags
deprecation_notes
change_log
```

其他因子不需要为了某个扩展字段强制补空值。

### 10.2 input_schema

```python
"input_schema": {
    "required": {
        "date": {
            "dtype": "datetime64[ns] 或可解析日期",
            "meaning": "观测日期及目标因子截面日期。",
        },
        "instrument": {
            "dtype": "string",
            "meaning": "证券唯一标识。",
        },
        "close": {
            "dtype": "float",
            "meaning": "与因子定义一致且窗口内口径统一的复权收盘价。",
        },
    },
    "conditional": {
        "industry": {
            "dtype": "string",
            "meaning": "目标日可得的行业分类。",
            "required_when": {"neutralize_industry": True},
        },
    },
}
```

字段规范中不记录适配器或数据域。

### 10.3 parameters

每个函数公开参数必须登记：

```python
"n_months": {
    "default": 6,
    "accepted_values": "正整数。",
    "effect": "决定回看月份数。",
    "changes_data_requirements": True,
}
```

`parameters` 的默认值必须与真实函数签名一致。

`data` 和内部使用的 `domain_data` 不登记为普通因子参数。`target_dates`、`as_of_date`、`show_progress` 和 `progress_every` 需要登记，因为它们属于公开统一接口，但评价和策略调用时由外层控制。

### 10.4 data_window

固定窗口直接声明：

```python
"data_window": {
    "lookback_trading_days": 0,
    "requires_target_date_data": True,
    "minimum_history_observations": 0,
    "preheating_required": False,
    "insufficient_window_behavior": "无法计算时保留 NaN。",
}
```

参数影响窗口时使用 resolver：

```python
"data_window": {
    "resolver": _resolve_factor_data_window,
    "default": {
        "lookback_trading_days": 126,
        "requires_target_date_data": True,
        "minimum_history_observations": 126,
        "preheating_required": True,
        "insufficient_window_behavior": "历史不足时保留 NaN。",
    },
    "resolver_notes": "窗口由 n_months 和 trading_days_per_month 决定。",
}
```

策略和评价必须使用 loader 解析出的同一份 `resolved_factor_params` 完成预热和最终计算，防止参数错位。

### 10.5 direction

`direction` 是因子的经验研究信息，不自动决定所有策略的选股顺序。

当前市值分组策略中的 `factor_quantile_range` 明确定义为：

```text
原始因子值从小到大的分位区间
```

例如 `(0.0, 0.1)` 选择每个市值组内原始因子值最低的约 10%，`(0.9, 1.0)` 选择最高的约 10%。调用者必须根据研究目标显式指定区间，策略不会根据 `direction` 自动翻转。

## 11. 因子中心

因子中心不维护人工巨型登记表。

```python
discover_factors()
list_factors()
search_factors(keyword)
describe_factor(name)
get_factor(name, data, **params)
```

新增因子只需把脚本放入 `Factor Repository` 的适当类别目录并声明 `FACTOR`。Notebook 已经导入旧版本时应重启内核。

## 12. 公共预处理

公共预处理函数必须：

- 不读取平台数据；
- 不出现因子名称或策略逻辑；
- 保留输入索引；
- 默认不填充缺失值；
- 明确处理单日截面还是完整面板；
- 被因子嵌套调用时保持静默。

当前典型能力：

- `winsorize_mad`：MAD 去极值；
- `zscore`：可选择 `ddof` 的截面标准化；
- `neutralize_ols`：一般 OLS 残差；
- `neutralize_size_industry`：市值及可选行业中性化。

因子应优先调用语义最贴近需求的公共函数。例如 BP 使用 `neutralize_size_industry(..., standardize_residual=False)` 取得残差，再按原定义执行 MAD 和 Z-score。

## 13. 因子评价

公开基础评价入口接收：

```python
calculate_factor_basic_metrics(
    start_date,
    end_date,
    frequency,
    factor_name,
    factor_params=None,
    instruments=None,
    min_obs=30,
    plot=True,
    plot_title=None,
    figsize=(14, 5),
    show_progress=False,
    progress_every=20,
)
```

它内部负责：

1. 生成评价截面；
2. 解析因子窗口；
3. 调用 loader 和因子中心；
4. 构造完整结束的未来收益标签；
5. 计算 IC、RankIC、ICIR、RankICIR、因子收益和平均 t 值；
6. 可选绘制 IC/RankIC 时序图。

未来收益是研究标签，不是可成交策略收益。研究结束日前尚未完整结束的标签必须剔除。

相关性入口为：

```python
calculate_factor_correlation(
    start_date,
    end_date,
    frequency,
    factor_names,
    factor_params_by_name=None,
    instruments=None,
    method="spearman",
    min_obs=30,
    plot=True,
    plot_title=None,
    figsize=(10, 8),
    annotate=True,
    show_progress=False,
    progress_every=20,
)
```

它接收因子名称列表和每个因子的独立参数，分别解析字段和预热窗口，对齐相同日期与证券后，先计算逐日截面相关系数，再汇总和可选绘图。

评价入口当前依赖 BigQuant 数据适配器获取交易日历、因子原始数据和标签价格，这是当前平台编排职责，不影响因子函数本身的数据源独立性。

## 14. 市值分组回测

公开入口：

```python
run_market_cap_group_backtest(
    start_date,
    end_date,
    rebalance_interval,
    universe,
    factor_name,
    market_cap_group_count=15,
    selected_market_cap_groups=None,
    factor_quantile_range=(0.0, 0.1),
    factor_params=None,
    order_price_field_buy="open",
    order_price_field_sell="open",
    initial_cash=1_000_000,
    benchmark="000300.SH",
    trading_costs=None,
    slippage_value=None,
    volume_limit=0.025,
    weight_tolerance=1e-4,
    show_progress=False,
    progress_every=20,
)
```

股票池支持：

- 全部 A 股；
- 一个或多个指数的历史成分股；
- 固定自定义证券列表。

核心流程：

1. 生成交易日调仓计划；
2. 执行日前一交易日作为信号日；
3. 根据因子参数解析每个信号日的预热窗口；
4. 分别预存因子数据、信号日选股状态和执行日交易限制；
5. 在信号日调用因子函数，仅计算该截面；
6. 按总市值从小到大划分近似等数量组；
7. 在每个指定组内按原始因子值分位区间选股；
8. 合并后全局等权；
9. 订单在下一交易日由 BigTrader 撮合；
10. 保留信号、调仓、执行、订单和成交审计表。

交易限制包括历史 ST/风险警示、停牌、成交量、执行价格、涨停买入和跌停卖出限制。无法卖出的旧持仓继续占用资金，无法买入的目标保留现金，不使用未来信息寻找替补股票。

策略依赖 BigQuant 原生 BigTrader 执行，这是策略运行器的正常平台依赖。由于回调闭包持有本次计划、数据和审计容器，必须设置：

```python
m_cached=False
```

避免复用上一次回测结果。

## 15. 点时性

必须区分：

```text
报告期
公告或真实可得日
原始数据日期
目标因子日
信号日
订单提交日
执行日
标签结束日
研究结束日
```

默认安全约定：

- 因子只使用目标日及以前真实可得的信息；
- 财务数据按公告或平台真实可得时间进入；
- 行业、指数成分、风险状态和停牌状态使用历史值；
- T 日收盘信息形成的信号不按 T 日收盘价成交；
- 未来收益只用于已完整结束的历史标签；
- 预热不足保留 `NaN`；
- 当前股票池和当前行业不能回填到历史。

## 16. 进度与缓存

长时间入口默认静默，支持：

```python
show_progress=False
progress_every=20
```

有循环时使用 `\r` 单行刷新，显示阶段、完成度、当前日期或任务、耗时和预计剩余时间。嵌套调用只由最外层显示进度，结束或异常退出时补换行。

修改 `.py` 后，BigQuant Notebook 可能仍保留旧模块：

1. 保存文件；
2. 重启内核；
3. 从头运行；
4. 再比较结果。

`sys.dont_write_bytecode = True` 只能阻止生成新的 `__pycache__`，不能清除已经导入的模块。

## 17. 迁移新因子

1. 从旧 Notebook 找到真正的裸因子公式；
2. 分离 SQL、标签、评价、图表和回测；
3. 搜索是否存在可以继续参数化的同类因子；
4. 识别可复用预处理；
5. 确定唯一语义字段名称；
6. 在正确平台适配器中补充字段映射；
7. 编写因子函数和完整 `FACTOR`；
8. 核对动态预热窗口；
9. 用相同日期和股票池对照旧结果；
10. 再进行统一评价和含成本回测；
11. 更新版本和 Git 记录。

不要因时间窗口不同复制脚本。例如 `return_1m`、`return_3m` 和 `return_6m` 应统一为 `return_nm` 并通过参数控制。

## 18. 验证边界

建议明确交付达到的验证等级：

| 等级 | 含义 |
|---|---|
| L1 | 仅静态阅读。 |
| L2 | 语法、AST 和导入检查。 |
| L3 | 构造数据或本地纯函数测试。 |
| L4 | BigQuant 真实数据查询和字段验证。 |
| L5 | BigTrader 回测、回调、订单和成交审计验证。 |
| L6 | 模拟盘生命周期、状态和订单验证。 |
| L7 | 小规模实盘验证。 |

不得把 L2 或 L3 表述为已经在 BigQuant 真实运行。

## 19. Git 与协作

- GitHub 保存可复现代码和有意义的变更记录；
- 不提交数据、凭据或无意义缓存；
- 提交前检查 `git status` 和暂存区统计；
- `git add -A` 会同时暂存删除文件；
- 修改因子、适配器或策略后，在 Notebook 中重启内核；
- 新的 LLM 或协作者应先阅读本文件和 `LLM_CONTEXT.md`。
