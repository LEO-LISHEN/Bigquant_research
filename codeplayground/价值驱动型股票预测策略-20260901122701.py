# 本代码由可视化策略环境自动生成 2023年7月16日 23:13
# 本代码单元只能在可视化模式下编辑。您也可以拷贝代码，粘贴到新建的代码单元或者策略，然后修改。
 
# 显式导入 BigQuant 相关 SDK 模块
from bigdatasource.api import DataSource
from bigdata.api.datareader import D
from biglearning.api import M
from biglearning.api import tools as T
from biglearning.module2.common.data import Outputs
 
import pandas as pd
import numpy as np
import math
import warnings
import datetime
 
from zipline.finance.commission import PerOrder
from zipline.api import get_open_orders
from zipline.api import symbol
 
from bigtrader.sdk import *
from bigtrader.utils.my_collections import NumPyDeque
from bigtrader.constant import OrderType
from bigtrader.constant import Direction

# <aistudiograph>

# @param(id="m3", name="initialize")
# 交易引擎：初始化函数，只执行一次
def m3_initialize_bigquant_run(context):    
    import math

    from bigtrader.finance.commission import PerOrder
    from biglearning.api import tools as T
    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    # 设置买入的股票数量，这里买入预测股票列表排名靠前的5只
    stock_count = 5
    # 每只的股票的权重，如下的权重分配会使得靠前的股票分配多一点的资金，[0.339160, 0.213986, 0.169580, ..]
    context.stock_weights = T.norm([1 / math.log(i + 2) for i in range(0, stock_count)])
    # 设置每只股票占用的最大资金比例
    context.max_cash_per_instrument = 0.2
    context.options['hold_days'] = 5

# @param(id="m3", name="before_trading_start")
# 交易引擎：每个单位时间开盘前调用一次。
def m3_before_trading_start_bigquant_run(context, data):
    # 盘前处理，订阅行情等
    pass

# @param(id="m3", name="handle_tick")
# 交易引擎：tick数据处理函数，每个tick执行一次
def m3_handle_tick_bigquant_run(context, tick):
    pass

# @param(id="m3", name="handle_data")
def m3_handle_data_bigquant_run(context, data):
    # 按日期过滤得到今日的预测数据
    ranker_prediction = context.data[context.data.date == data.current_dt.strftime('%Y-%m-%d')]
    # 按照position排序
    ranker_prediction.sort_values(["date", "position"], inplace=True)
    ranker_prediction.reset_index(drop=True, inplace=True)

    # 1. 资金分配
    # 平均持仓时间是hold_days，每日都将买入股票，每日预期使用 1/hold_days 的资金
    # 实际操作中，会存在一定的买入误差，所以在前hold_days天，等量使用资金；之后，尽量使用剩余资金（这里设置最多用等量的1.5倍）
    is_staging = context.trading_day_index < context.options['hold_days'] # 是否在建仓期间（前 hold_days 天）
    cash_avg = context.portfolio.portfolio_value / context.options['hold_days']
    cash_for_buy = min(context.portfolio.cash, (1 if is_staging else 1.5) * cash_avg)
    cash_for_sell = cash_avg - (context.portfolio.cash - cash_for_buy)
    positions = {e: p.amount * p.last_sale_price
                for e, p in context.portfolio.positions.items()}

    # 2. 生成卖出订单：hold_days天之后才开始卖出；对持仓的股票，按机器学习算法预测的排序末位淘汰
    if not is_staging and cash_for_sell > 0:
        equities = {e: e for e, p in context.portfolio.positions.items()}
        instruments = list(reversed(list(ranker_prediction.instrument[ranker_prediction.instrument.apply(
                lambda x: x in equities)])))
        for instrument in instruments:
            context.order_target(instrument, 0)
            cash_for_sell -= positions[instrument]
            if cash_for_sell <= 0:
                break

    # 3. 生成买入订单：按机器学习算法预测的排序，买入前面的stock_count只股票
    buy_cash_weights = context.stock_weights
    buy_instruments = list(ranker_prediction.instrument[:len(buy_cash_weights)])
    max_cash_per_instrument = context.portfolio.portfolio_value * context.max_cash_per_instrument
    for i, instrument in enumerate(buy_instruments):
        cash = cash_for_buy * buy_cash_weights[i]
        if cash > max_cash_per_instrument - positions.get(instrument, 0):
            # 确保股票持仓量不会超过每次股票最大的占用资金量
            cash = max_cash_per_instrument - positions.get(instrument, 0)
        if cash > 0:
            context.order_value(instrument, cash)

# @param(id="m3", name="handle_trade")
# 交易引擎：成交回报处理函数，每个成交发生时执行一次
def m3_handle_trade_bigquant_run(context, trade):
    pass

# @param(id="m3", name="handle_order")
# 交易引擎：委托回报处理函数，每个委托变化时执行一次
def m3_handle_order_bigquant_run(context, order):
    pass

# @param(id="m3", name="after_trading")
# 交易引擎：盘后处理函数，每日盘后执行一次
def m3_after_trading_bigquant_run(context, data):
    pass


# @module(position="-244,-8", comment='通过SQL调用数据、因子和表达式等构建策略逻辑', comment_collapsed=False)
m1 = M.input_features_dai.v6(
    sql="""-- 使用DAI SQL获取数据，构建因子等，如下是一个例子作为参考
-- DAI SQL 语法: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-sql%E5%85%A5%E9%97%A8%E6%95%99%E7%A8%8B

SELECT
    position,
    -- 日期，这是每个股票每天的数据
    date,
    -- 股票代码，代表每一支股票
    instrument
FROM user_factor_019c406a1637c46418f4f5e1fc0b817b
JOIN cn_stock_factors
USING (date, instrument)
WHERE
    -- 剔除ST股票
    st_status = 0
    -- 非停牌股
    AND suspended = 0
    -- 不属于北交所
    AND list_sector < 4
"""
)

# @module(position="-100,146", comment='抽取数据，设置数据开始时间和结束时间，并绑定模拟交易', comment_collapsed=False)
m2 = M.extract_data_dai.v7(
    sql=m1.data,
    start_date='',
    start_date_bound_to_trading_date=True,
    end_date='',
    end_date_bound_to_trading_date=True,
    before_start_days=10,
    debug=False
)

# @module(position="43,301", comment='交易，日线，设置初始化函数和K线处理函数，以及初始资金、基准等', comment_collapsed=False)
m3 = M.bigtrader.v7(
    data=m2.data,
    start_date='',
    end_date='',
    initialize=m3_initialize_bigquant_run,
    before_trading_start=m3_before_trading_start_bigquant_run,
    handle_tick=m3_handle_tick_bigquant_run,
    handle_data=m3_handle_data_bigquant_run,
    handle_trade=m3_handle_trade_bigquant_run,
    handle_order=m3_handle_order_bigquant_run,
    after_trading=m3_after_trading_bigquant_run,
    capital_base=1000000,
    frequency='daily',
    product_type='股票',
    before_start_days=0,
    volume_limit=1,
    order_price_field_buy='open',
    order_price_field_sell='close',
    benchmark='000300.SH',
    plot_charts=True,
    disable_cache=True,
    debug=False,
    backtest_only=False
)
# </aistudiograph>