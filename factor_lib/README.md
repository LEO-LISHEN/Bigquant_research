# factor_lib

`factor_lib` 是一个面向量化研究生产化的可复用组件库，用于沉淀因子、公共预处理、因子评价、策略模板和数据源适配逻辑。

当前主要运行环境是 BigQuant，但核心设计目标不是绑定 BigQuant，而是让同一个因子公式和研究逻辑能够在更换数据源后继续复用。该项目所说的“生产化”，是指代码可管理、可发现、可复用、可测试和可追溯，不代表已经满足实盘交易系统的全部生产要求。

## 1. 核心原则

整个项目遵循以下原则：

1. 因子、数据、评价、策略和执行分层管理。
2. 核心计算使用统一的语义标准字段，不直接依赖数据供应商字段。
3. 因子函数只计算因子值，不负责查询数据、构造标签、绘图、选股或回测。
4. 数据适配器只负责拉取、筛选、映射和对齐原始数据，不负责因子特有的数据处理。
5. 面向用户的评价函数和策略函数直接接收因子名称，不要求用户提前准备 `factor_data`。
6. 能通过参数表达的同类因子只保留一个参数化脚本。
7. 所有研究必须遵守点时性，禁止使用信号时点尚不可获得的信息。
8. 长时间任务默认静默，可选显示单行刷新的进度；嵌套调用只由最外层显示进度。
9. 本地语法检查通过不等于 BigQuant 平台运行通过，平台相关功能必须在 BigQuant 中验证。

## 2. 目录结构

目录仅按职责划分。`Factor Repository` 内部可以继续按因子类别扩展，但本 README 不固定具体类别和因子文件。

```text
factor_lib/
├── README.md
├── Factor Repository/
├── common/
│   ├── preprocess/
│   ├── factor_evaluation/
│   ├── strategies/
│   └── data_adapters/
└── factor_hub/
```

各大类职责如下：

| 目录 | 职责 |
|---|---|
| `Factor Repository/` | 存放因子计算脚本及其 `FACTOR` 元数据；原则上一个参数化因子族一个脚本。 |
| `common/preprocess/` | 存放可跨因子复用的纯预处理函数，例如去极值、标准化和中性化。 |
| `common/factor_evaluation/` | 存放因子基础指标、相关性分析及配套绘图等统一评价入口。 |
| `common/strategies/` | 存放可复用的选股、组合构建和回测策略模板。 |
| `common/data_adapters/` | 存放不同数据源、数据域和频率的字段映射与原始数据加载逻辑。 |
| `factor_hub/` | 动态发现、查询、说明和调用因子，避免维护人工巨型登记表。 |

## 3. 总体数据流

推荐的调用链如下：

```text
研究、评价或策略公开入口
        ↓
因子中心读取 FACTOR 元数据
        ↓
解析因子参数和本次数据窗口
        ↓
loader 统筹所需数据域
        ↓
具体平台适配器拉取并映射原始字段
        ↓
生成使用语义标准字段的 DataFrame
        ↓
因子函数仅在 target_dates 上计算因子值
        ↓
评价、组合构建或平台执行
```

必须区分两类平台依赖：

- **数据源适配器**：处理 BigQuant、其他数据库或本地文件的取数与字段映射。
- **执行平台适配器或运行器**：处理 BigTrader 等回测/交易引擎的模块、回调、订单和账户接口。

选股规则、组合构建和因子逻辑应尽量保持平台无关。`dai`、SQL、BigQuant 表名、`M.bigtrader`、`context` 和平台回调等内容属于适配或执行层，不应进入因子计算函数。

## 4. 语义标准字段

因子函数声明的是跨数据源的语义字段，例如：

```text
date
instrument
open
close
volume
turn
pb
total_market_cap
industry
```

这些名称表达字段的业务含义，不表达某个平台的真实表名。平台适配器负责把实际字段映射为标准字段。

新增字段前必须先检查：

1. 现有适配器是否已经存在同义字段；
2. 是否能够沿用现有标准名称；
3. 字段属于哪个数据域和频率；
4. 是否满足点时性；
5. 是否需要新的适配器，而不是把所有数据源逻辑塞进一个脚本。

数据适配器可以执行：

- 按日期区间或日期列表拉取数据；
- 按证券范围过滤；
- 将平台字段重命名为标准字段；
- 对不同数据域进行必要的点时对齐；
- 合并多个适配器返回的数据面板；
- 检查键列、重复记录和必要字段。

数据适配器不得执行：

- 因子公式计算；
- 因子专用缺失值填充；
- 去极值、标准化或中性化；
- 标签构造；
- 股票筛选、分组和组合权重计算；
- 回测交易逻辑。

## 5. 因子脚本规范

### 5.1 一个参数化因子族一个脚本

仅时间窗口、衰减参数、最小观测数或处理开关不同的因子，应视为同一个因子族。

例如：

```text
return_1m、return_3m、return_6m
        ↓
return_nm.py + n_months 参数
```

不要因为参数不同复制多个高度相似的脚本。推荐实例可以写入 `FACTOR["best_practice"]`，但不能代替参数化设计。

只有在公式、数据语义或处理流程发生本质变化时，才应新建因子脚本。

### 5.2 统一因子函数接口

推荐接口为：

```python
def calc_factor(
    data,
    target_dates=None,
    as_of_date=None,
    show_progress=False,
    progress_every=20,
    **factor_params,
):
    ...
```

参数含义：

- `data`：已经由数据适配器准备好的原始数据，可以包含目标日前的预热数据。
- `target_dates`：实际需要输出因子值的截面日期。它与数据准备日期必须分开。
- `as_of_date`：本次计算允许使用信息的全局截止日，不能代替 `target_dates`。
- `show_progress`：是否显示计算进度，默认 `False`。
- `progress_every`：循环任务每处理多少个单位刷新一次进度。
- `factor_params`：该因子自身的窗口、衰减、处理开关等参数。

当 `target_dates=None` 时，函数可以按自身文档约定计算输入数据中所有可计算日期；策略和正式评价入口必须显式传入目标日期。

### 5.3 统一输出

所有因子函数统一返回长表：

```text
date | instrument | factor_name
```

要求：

- `date` 是目标因子截面日期；
- `instrument` 是证券唯一标识；
- 因子值列名与 `FACTOR["name"]` 一致；
- 一行对应一个日期和一只证券；
- 输出只包含目标日期，不应混入仅用于预热的日期；
- 历史不足、输入无效或无法计算时保留 `NaN`，不得为了“完整”而统一填为 `0`；
- 结果应按 `date`、`instrument` 稳定排序；
- 不返回标签、分组、持仓、收益率或图表。

### 5.4 因子内部允许与禁止的内容

因子脚本允许：

- 校验标准输入字段；
- 校验因子参数；
- 按 `as_of_date` 截断数据；
- 调用公共预处理函数；
- 根据历史窗口计算目标截面因子；
- 返回标准结果；
- 声明 `FACTOR` 元数据。

因子脚本禁止：

- 导入 `dai` 并查询数据；
- 出现 BigQuant 表名或 SQL；
- 构造未来收益标签；
- 计算 IC、RankIC 或回测绩效；
- 进行股票池选择和交易限制过滤；
- 提交订单；
- 自动保存文件或展示图表。

## 6. `FACTOR` 元数据规范

每个因子脚本必须在模块顶层提供一个名为 `FACTOR` 的字典。因子中心、loader、评价函数和策略函数都以它为统一契约。

### 6.1 核心必填字段

| 字段 | 含义 |
|---|---|
| `name` | 因子中心使用的唯一名称，也应与输出因子值列名一致。 |
| `func` | 实际因子计算函数对象。 |
| `category` | 因子类别，用于查询和组织。 |
| `direction` | 经验方向：`1` 表示值越大通常越优，`-1` 表示值越小通常越优。 |
| `description` | 简明说明因子的经济含义与主要处理流程。 |
| `formula` | 公式、变量定义和关键计算顺序。 |
| `input_schema` | 标准输入字段及其类型、频率、含义和条件依赖。 |
| `parameters` | 因子参数的默认值、合法取值、效果及是否改变数据需求。 |
| `data_window` | 固定或动态预热窗口、目标日需求、最少历史观测和不足时行为。 |
| `output_schema` | 输出字段、类型和含义。 |
| `usage_notes` | 调用方式、参数对应关系和使用限制。 |
| `pit_notes` | 点时性要求、可用时点和潜在前视风险。 |

### 6.2 输入字段规范

`input_schema` 至少包含：

```python
"input_schema": {
    "required": {
        "date": {
            "dtype": "datetime64[ns] 或可解析日期",
            "frequency": "daily",
            "meaning": "观测日期及目标因子截面日期",
        },
        "instrument": {
            "dtype": "string",
            "frequency": "daily",
            "meaning": "证券唯一标识",
        },
    },
    "conditional": {
        "industry": {
            "dtype": "string",
            "frequency": "daily",
            "meaning": "目标日可得的历史行业分类",
            "required_when": "neutralize_industry=True",
        },
    },
}
```

每个字段都应说明：

- 标准字段名；
- 数据类型；
- 数据频率或所属数据域；
- 业务含义；
- 条件字段在什么参数条件下必需。

不得在这里写死 BigQuant 表名。具体表和真实字段属于平台适配器。

### 6.3 参数规范

每个公开因子参数至少说明：

```python
"parameters": {
    "n_months": {
        "default": 6,
        "accepted_values": "正整数",
        "effect": "决定因子回看月份数",
        "changes_data_requirements": True,
    },
}
```

其中：

- `default`：默认值；
- `accepted_values`：合法范围或可选值；
- `effect`：对公式或结果的影响；
- `changes_data_requirements`：是否会改变所需字段或预热窗口。

策略和评价函数必须用同一份已经解析的 `factor_params` 同时完成数据窗口解析和最终因子计算，避免“按一个窗口取数、按另一个窗口计算”。

### 6.4 固定与动态数据窗口

固定窗口可以直接写在 `data_window["default"]` 中。

参数会改变窗口时，必须提供解析函数：

```python
def _resolve_example_data_window(resolved_params):
    n_months = resolved_params.get("n_months", 6)
    trading_days_per_month = resolved_params.get(
        "trading_days_per_month",
        21,
    )
    lookback_days = n_months * trading_days_per_month

    return {
        "lookback_trading_days": lookback_days,
        "requires_target_date_data": True,
        "minimum_history_observations": lookback_days,
        "preheating_required": True,
        "insufficient_window_behavior": "历史不足时输出 NaN",
    }
```

对应元数据：

```python
"data_window": {
    "resolver": _resolve_example_data_window,
    "default": {
        "lookback_trading_days": 126,
        "requires_target_date_data": True,
        "minimum_history_observations": 126,
        "preheating_required": True,
        "insufficient_window_behavior": "历史不足时输出 NaN",
    },
    "resolver_notes": (
        "loader 和因子函数必须使用同一份 resolved_factor_params。"
    ),
}
```

`data_window` 至少明确：

- `lookback_trading_days`：目标日前需要回看的交易日数量；
- `requires_target_date_data`：公式是否需要目标日数据；
- `minimum_history_observations`：单只证券最少有效历史观测数；
- `preheating_required`：是否必须准备预热数据；
- `insufficient_window_behavior`：预热不足时如何处理；
- 动态窗口的 `resolver` 和解析说明。

### 6.5 可扩展字段

不同因子可以按需要增加以下字段，其他因子无需被迫同步增加：

- `best_practice`
- `references`
- `research_findings`
- `tags`
- `status`
- `version`
- `deprecation_notes`
- `change_log`

核心字段必须统一，可扩展字段按实际研究需要增加。

### 6.6 完整骨架

```python
FACTOR = {
    "name": "example_factor_nm",
    "func": calc_example_factor_nm,
    "category": "example",
    "direction": 1,
    "description": "因子经济含义和主要处理流程。",
    "formula": "清晰写明公式、变量和计算顺序。",
    "input_schema": {
        "required": {
            "date": {
                "dtype": "datetime64[ns] 或可解析日期",
                "frequency": "daily",
                "meaning": "观测日期",
            },
            "instrument": {
                "dtype": "string",
                "frequency": "daily",
                "meaning": "证券唯一标识",
            },
        },
        "conditional": {},
    },
    "parameters": {},
    "data_window": {
        "resolver": _resolve_example_data_window,
        "default": {
            "lookback_trading_days": 0,
            "requires_target_date_data": True,
            "minimum_history_observations": 1,
            "preheating_required": False,
            "insufficient_window_behavior": "无法计算时输出 NaN",
        },
        "resolver_notes": "说明窗口与参数之间的关系。",
    },
    "output_schema": {
        "date": {
            "dtype": "datetime64[ns]",
            "meaning": "目标因子截面日期",
        },
        "instrument": {
            "dtype": "string",
            "meaning": "证券唯一标识",
        },
        "example_factor_nm": {
            "dtype": "float64",
            "meaning": "因子值含义",
        },
    },
    "usage_notes": [],
    "pit_notes": [],
    "best_practice": {},
    "status": "research",
    "version": "1.0.0",
}
```

## 7. 公共预处理函数规范

公共预处理函数应遵循：

1. 一个可复用操作一个脚本；
2. 不读取数据源；
3. 不包含特定因子名称和策略逻辑；
4. 默认不擅自填充缺失值；
5. 尽量保持输入索引，避免样本错位；
6. 明确函数处理的是单日截面还是完整面板；
7. 暴露 `show_progress=False`，有循环时同时暴露 `progress_every=20`；
8. 被因子函数嵌套调用时保持静默。

底层通用函数和上层便捷函数可以同时存在。例如：

- 底层 OLS 残差函数负责一般线性中性化；
- 上层市值行业中性化函数负责构造常用控制变量，再调用底层函数。

因子优先调用语义最贴近需求的公共函数；只有出现新的通用能力时才扩展 `preprocess`，不要在多个因子脚本中复制相同实现。

## 8. 因子中心规范

因子中心通过扫描因子脚本中的 `FACTOR` 自动发现因子，不维护人工巨型 `FACTOR_REGISTRY`。

建议公开能力：

- `discover_factors()`：扫描并建立当前因子索引；
- `list_factors()`：列出可用因子的摘要；
- `search_factors()`：按名称、类别、描述或标签搜索；
- `describe_factor()`：查看某个因子的完整元数据；
- `get_factor()`：按因子名称调用真实计算函数。

典型用法：

```python
from factor_lib.factor_hub.list_factors import list_factors
from factor_lib.factor_hub.search_factors import search_factors
from factor_lib.factor_hub.describe_factor import describe_factor

display(list_factors())
display(search_factors("momentum"))
display(describe_factor("factor_name"))
```

新增因子时，只需：

1. 将脚本放入 `Factor Repository` 的合适类别目录；
2. 提供符合规范的计算函数和 `FACTOR`；
3. 重新启动 Notebook 内核或刷新因子发现缓存。

不需要在中心登记表中重复手工登记。

## 9. 因子评价规范

面向研究者的公开评价函数应直接接收：

- 起始日期；
- 结束日期；
- 评价频率；
- 因子名称或因子名称列表；
- 因子参数；
- 可选证券范围；
- 指标、绘图和进度参数。

公开入口不应要求用户提前传入 `factor_data` 或 `label_data`。评价函数内部负责：

1. 生成评价截面；
2. 读取 `FACTOR`；
3. 解析因子参数和预热窗口；
4. 调用 loader 拉取原始数据；
5. 调用因子函数；
6. 构造点时一致的完整未来标签；
7. 计算指标；
8. 按参数决定是否绘图。

内部私有纯函数为了便于单元测试，可以接收已经准备好的 DataFrame；“不接收 `factor_data`”的要求针对面向用户的公开入口。

### 9.1 基础指标

```python
from factor_lib.common.factor_evaluation.calculate_factor_basic_metrics import (
    calculate_factor_basic_metrics,
)

result = calculate_factor_basic_metrics(
    start_date="2022-01-01",
    end_date="2024-12-31",
    frequency=20,
    factor_name="factor_name",
    factor_params={},
    instruments=None,
    min_obs=30,
    plot=True,
    show_progress=True,
)

display(result["summary"])
```

基础评价中的未来收益只是研究标签，不等同于可成交的组合回测收益。结束日前无法完整结束的未来标签必须剔除。

### 9.2 因子相关性

```python
from factor_lib.common.factor_evaluation.calculate_factor_correlation import (
    calculate_factor_correlation,
)

result = calculate_factor_correlation(
    start_date="2022-01-01",
    end_date="2024-12-31",
    frequency=20,
    factor_names=[
        "factor_a",
        "factor_b",
    ],
    factor_params_by_name={
        "factor_a": {},
        "factor_b": {},
    },
    method="spearman",
    min_obs=30,
    plot=True,
    show_progress=True,
)

display(result["correlation_matrix"])
```

相关性应优先计算同一日期、同一股票上的截面相关系数，再对有效日期汇总；不能把整个面板直接混在一起计算而忽略日期结构。

## 10. 策略函数规范

面向用户的策略公开入口应接收：

- 回测起止日期；
- 调仓频率或触发规则；
- 选股范围；
- 因子名称；
- 因子参数；
- 组合构建参数；
- 交易价格；
- 初始资金；
- 基准；
- 交易成本、税费、滑点和成交量限制；
- 进度参数。

不应要求用户传入已经计算好的 `factor_data`。

推荐流程：

1. 根据回测区间和交易日历生成调仓执行日；
2. 明确信号日与执行日；
3. 根据 `FACTOR` 和因子参数解析预热窗口；
4. 分别准备因子数据日期和交易限制数据日期；
5. 通过 loader 一次性或分块加载原始数据；
6. 在信号日调用因子函数，只计算所需截面；
7. 根据 `FACTOR["direction"]`、选股规则和组合规则生成目标持仓；
8. 在执行日检查 ST、停牌、涨跌停、成交量等限制；
9. 通过平台执行器提交订单；
10. 返回平台绩效对象和可选审计数据。

默认的安全时序是：

```text
T 日收盘后获得完整 T 日数据并产生信号
                    ↓
T+1 交易日按设定价格执行
```

如果使用其他时序，必须明确该时点真实可获得的数据，并证明不存在使用未来信息或使用尚未形成的价格成交。

市值分组回测示例：

```python
from factor_lib.common.strategies.market_cap_group_backtest import (
    run_market_cap_group_backtest,
)

result = run_market_cap_group_backtest(
    start_date="2022-01-01",
    end_date="2024-12-31",
    rebalance_interval=20,
    universe={"type": "all_a"},
    factor_name="factor_name",
    factor_params={},
    market_cap_group_count=15,
    selected_market_cap_groups=[1, 2, 3],
    factor_quantile_range=(0.0, 0.1),
    order_price_field_buy="open",
    order_price_field_sell="open",
    initial_cash=1_000_000,
    benchmark="000300.SH",
    trading_costs={
        "buy_cost": 0.0003,
        "sell_cost": 0.0003,
        "min_cost": 5.0,
        "tax_ratio": 0.0005,
    },
    slippage_value=0.001,
    volume_limit=0.025,
    show_progress=True,
)
```

公开调用默认只展示 BigQuant 回测图表。信号、调仓、执行、订单和成交审计表应通过返回结果显式查看，避免 Notebook 自动刷出大量中间内容。

## 11. 点时性与防前视

所有因子、评价和策略必须遵守：

1. 目标日因子只能使用目标日及以前真实可得的数据；
2. `as_of_date` 之后的数据必须截断；
3. 财务数据按公告日或真实可用日对齐，不能仅按报告期结束日回填历史；
4. 行业、指数成分、风险警示和停牌状态必须使用历史状态；
5. 未来收益只能作为标签，不能进入因子输入；
6. 中性化和分组使用的截面股票池必须明确；
7. 使用 T 日收盘数据生成的信号不能按 T 日收盘价成交；
8. 评价结束日之后才完整形成的标签必须剔除；
9. 预热不足时保留 `NaN`，不能使用未来数据补齐；
10. 任何缓存都不能让上一次运行的闭包、信号或参数污染本次回测。

## 12. 进度、输出与 Notebook 缓存

公共因子、预处理、评价和策略函数统一支持：

```python
show_progress=False
progress_every=20
```

规范：

- 默认静默；
- 使用 `\r` 在终端单行刷新；
- 显示阶段、完成度、当前日期或任务、已耗时和预计剩余时间；
- 嵌套调用时只有最外层函数显示进度；
- 完成或异常退出时补换行；
- 不在循环中刷出大量日志；
- 不自动展示大型 DataFrame；
- 不自动写入研究结果文件。

BigQuant Notebook 会缓存已经导入的 Python 模块。修改因子库、适配器、评价函数或策略后，最可靠的做法是重启内核后重新运行。

```python
sys.dont_write_bytecode = True
```

只能阻止生成新的 `__pycache__` 字节码文件，不能清除内存中已经导入的旧模块。

对于当前依赖 Notebook 闭包状态的 BigTrader 回测，必须关闭模块结果缓存，例如使用：

```python
m_cached=False
```

否则相同模块参数可能命中旧缓存，导致本次回测没有执行新的回调或错误展示上一次结果。

## 13. 从旧 Notebook 迁移因子

迁移流程：

1. 找到旧 Notebook 中真正的裸因子公式；
2. 分离 SQL、数据清洗、标签、评价、图表和回测代码；
3. 搜索因子库，确认是否已存在可参数化的同类因子；
4. 检查预处理步骤能否复用现有公共函数；
5. 若是通用操作，将其独立放入 `common/preprocess`；
6. 为全部原始输入确定唯一的语义标准字段；
7. 检查对应数据适配器是否已有映射；
8. 编写因子计算函数；
9. 完整填写 `FACTOR`；
10. 用相同日期、相同股票池对照旧结果；
11. 通过统一基础指标和 IC/RankIC 时序验证；
12. 再用带交易成本和交易限制的策略回测验证；
13. 记录版本和有意义的 Git 变更。

迁移时必须尽量保留原研究的：

- 原始公式；
- 数据口径；
- 处理顺序；
- 截面范围；
- 标准差自由度等统计细节；
- 因子方向；
- 有效样本要求。

如果主动改变这些内容，应视为新版本或新定义，并在元数据和版本记录中说明。

## 14. 版本、状态与废弃

建议因子使用清晰状态：

```text
draft → research → validated → deprecated
```

- `draft`：正在迁移或开发；
- `research`：可以用于研究，但尚未完成充分验证；
- `validated`：已通过定义对照、统一评价和指定策略验证；
- `deprecated`：保留兼容性，但不建议新研究继续使用。

下列变化应更新版本或变更记录：

- 公式变化；
- 输入字段含义变化；
- 预处理顺序变化；
- 点时口径变化；
- 默认参数变化；
- 因子方向变化；
- 输出语义变化。

仅修改文字说明或不影响数值结果的代码整理，可以只记录普通提交。

GitHub 用于保存可复现代码和有意义的变更记录。原始大数据、临时缓存和无必要的运行产物不应提交。研究结论属于文档或元数据扩展字段，不能写进因子计算流程并改变其职责。

## 15. 新增或修改组件的检查清单

### 新增因子

- [ ] 已搜索现有因子，确认不是同类参数实例；
- [ ] 相同公式的窗口差异已经参数化；
- [ ] 函数只接收标准字段；
- [ ] 没有 `dai`、SQL、平台表名和策略代码；
- [ ] `data` 与 `target_dates` 分离；
- [ ] `as_of_date` 截止逻辑明确；
- [ ] 输出仅含 `date`、`instrument` 和因子列；
- [ ] 预热不足输出 `NaN`；
- [ ] `FACTOR` 核心字段完整；
- [ ] 动态参数能够正确解析数据窗口；
- [ ] 已检查适配器字段映射；
- [ ] 已与旧定义或参考实现对照。

### 新增预处理

- [ ] 操作可被多个因子复用；
- [ ] 没有数据源和因子专属逻辑；
- [ ] 索引与缺失值行为明确；
- [ ] 截面或时序处理范围明确；
- [ ] 没有重复实现现有函数。

### 新增评价函数

- [ ] 公开入口接收因子名称而非 `factor_data`；
- [ ] 自动解析参数和预热窗口；
- [ ] 标签完整且无前视；
- [ ] 绘图由参数控制；
- [ ] 返回结构包含必要结果与审计数据。

### 新增策略

- [ ] 公开入口接收因子名称和因子参数；
- [ ] 信号日与执行日明确；
- [ ] 使用同一份因子参数取数和计算；
- [ ] 股票池使用历史状态；
- [ ] 考虑 ST、停牌、涨跌停和成交量限制；
- [ ] 考虑佣金、税费和滑点；
- [ ] 订单失败和未成交有审计记录；
- [ ] 平台执行代码与通用组合逻辑边界清晰。

## 16. 面向 LLM 的强制约定

LLM 在修改本项目时必须：

1. 先阅读本 README；
2. 先搜索是否已有因子、字段映射和公共预处理；
3. 优先扩展参数，不复制同类因子脚本；
4. 根据 `FACTOR` 自动解析字段和数据窗口；
5. 保证取数与计算使用同一份已解析因子参数；
6. 保留公式、数据口径和处理顺序；
7. 明确本地验证与 BigQuant 平台验证的边界；
8. 对公式或时序变化进行版本说明；
9. 修改库文件后提醒重启 Notebook 内核；
10. 为长任务提供默认静默的单行进度。

LLM 不得：

1. 在因子函数中加入 SQL、数据表名、标签、图表、选股或订单；
2. 让公开评价或策略接口要求用户手工传入 `factor_data`；
3. 把不同窗口的同一公式拆成多个因子脚本；
4. 在多个因子中复制同一预处理实现；
5. 为了消除缺失值而统一填 `0`；
6. 使用未来财务公告、未来成分股或未来交易状态；
7. 把语法检查通过表述为 BigQuant 回测或交易已经验证成功；
8. 未说明原因就改变公式、因子方向、默认参数或交易时序。

---

本 README 是 `factor_lib` 的架构契约。新增组件时，应优先保持该契约稳定；确需改变公共接口或职责边界时，应先更新本文档，再同步修改相关代码和调用示例。
