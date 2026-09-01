from bigmodule import M

# <aistudiograph>

# @param(id="m8", name="initialize")
# 交易引擎：初始化函数, 只执行一次
def m8_initialize_bigquant_run(context):
    import math
    import numpy as np

    from bigtrader.finance.commission import PerOrder

    # 系统已经设置了默认的交易手续费和滑点, 要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    # 预测数据, 通过 options 传入进来, 使用 read_df 函数, 加载到内存 (DataFrame)
    context.stock_count = 2

    # 每只股票的权重平均分配
    context.stock_weights = 1/context.stock_count

    context.options['hold_days'] = 0


# @param(id="m8", name="before_trading_start")
# 交易引擎：每个单位时间开盘前调用一次。
def m8_before_trading_start_bigquant_run(context, data):
    # 盘前处理，订阅行情等
    pass

# @param(id="m8", name="handle_tick")
# 交易引擎：tick数据处理函数，每个tick执行一次
def m8_handle_tick_bigquant_run(context, tick):
    pass

# @param(id="m8", name="handle_data")
# 回测引擎：每日数据处理函数, 每天执行一次
def m8_handle_data_bigquant_run(context, data):
    today = data.current_dt.strftime('%Y-%m-%d')
    equities = {e: p for e, p in context.portfolio.positions.items() if p.amount>0}
    stock_now = len(equities); #获取当前持仓股票数量
    stock_count = context.stock_count
    
    # 按日期过滤得到今日的预测数据
    df_today = context.data[
        context.data.date == data.current_dt.strftime("%Y-%m-%d")
    ]
    df_today.set_index('instrument')
    
    
    now_stock = []
    sell_stock = []

    #df_today['instrument'] = df_today['instrument'].str.replace('A$', '', regex=True)
    buy_list = df_today['instrument']

    
    # 1. 资金分配


    positions = {e: p.amount * p.last_sale_price
                 for e, p in context.portfolio.positions.items()}
    
    
    if 1==1 :    
        if len(equities) > 0:
            for i in equities.keys():
                last_sale_date = equities[i].last_sale_date	# 上次交易日期
                delta_days = data.current_dt - last_sale_date  
                hold_days = delta_days.days # 持仓天数
                if hold_days >= context.options['hold_days'] and i not in buy_list :
                    context.order_target(context.symbol(i), 0)
                    sell_stock.append(i)
                    stock_now = stock_now -1
                 
    # 3. 生成买入订单
    #buy_num = stock_count - stock_now
    buy_num=min(len(buy_list),stock_count)
    if buy_num>0 and len(buy_list)>0 :
        cash_for_buy=context.portfolio.portfolio_value/(buy_num*2)
        # 不再买入已经轮仓卖出和移动止损的股票,以防止出现空头持仓
        buy_instruments = [i for i in buy_list if i not in now_stock][:buy_num]
        for i, instrument in enumerate(buy_instruments):
            cash_for_buy=min(cash_for_buy,context.portfolio.cash)
            #仓位不足10%，不进行买入动作
            if cash_for_buy<context.portfolio.portfolio_value*0.1 :
                break
            context.order_value(instrument, cash_for_buy)
            stock_now = stock_now + 1


# @param(id="m8", name="handle_trade")
# 交易引擎：成交回报处理函数，每个成交发生时执行一次
def m8_handle_trade_bigquant_run(context, trade):
    pass

# @param(id="m8", name="handle_order")
# 交易引擎：委托回报处理函数，每个委托变化时执行一次
def m8_handle_order_bigquant_run(context, order):
    pass

# @param(id="m8", name="after_trading")
# 交易引擎：盘后处理函数，每日盘后执行一次
def m8_after_trading_bigquant_run(context, data):
    pass

# @module(position="-369,-649", comment="""""", comment_collapsed=True)
m9 = M.input_features_dai.v23(
    mode="""SQL""",
    expr="""""",
    expr_filters="""""",
    expr_tables="""cn_stock_factors""",
    extra_fields="""date, instrument""",
    order_by="""date, instrument""",
    expr_drop_na=True,
    sql="""select date, instrument
from 
user_data_yuhuai16321 

order by date desc,sort asc""",
    extract_data=False,
    m_name="""m9"""
)

# @module(position="-364,-491", comment="""""", comment_collapsed=True)
m7 = M.extract_data_dai.v15(
    sql=m9.data,
    start_date="""2023-01-01""",
    start_date_bound_to_trading_date=True,
    end_date="""2024-04-30""",
    end_date_bound_to_trading_date=True,
    before_start_days=0,
    debug=False,
    m_name="""m7"""
)

# @module(position="-450,-262", comment="""""", comment_collapsed=True)
m8 = M.bigtrader.v17(
    data=m7.data,
    start_date="""""",
    end_date="""""",
    initialize=m8_initialize_bigquant_run,
    before_trading_start=m8_before_trading_start_bigquant_run,
    handle_tick=m8_handle_tick_bigquant_run,
    handle_data=m8_handle_data_bigquant_run,
    handle_trade=m8_handle_trade_bigquant_run,
    handle_order=m8_handle_order_bigquant_run,
    after_trading=m8_after_trading_bigquant_run,
    capital_base=1000000,
    frequency="""daily""",
    product_type="""股票""",
    rebalance_period_type="""交易日""",
    rebalance_period_days="""5""",
    rebalance_period_roll_forward=True,
    backtest_engine_mode="""标准模式""",
    before_start_days=0,
    volume_limit=1,
    order_price_field_buy="""open""",
    order_price_field_sell="""close""",
    benchmark="""沪深300指数""",
    plot_charts=True,
    debug=False,
    backtest_only=False,
    m_name="""m8"""
)
# </aistudiograph>