from bigquant import bigtrader, dai

def initialize(context: bigtrader.IContext):
    context.set_commission(bigtrader.PerOrder(buy_cost=0.0003, sell_cost=0.0008, min_cost=5))


    import dai 
    sql = "select * from user_factor_sfgsxysnxfhs where date >= '2010-01-01' "
    df = dai.query(sql).df() 


    # 每日选取市值最小的前 N 只股票
    stock_count = 5
    df = df.groupby('date').head(stock_count)

    # 设置每日调仓
    df = bigtrader.TradingDaysRebalance(1, context=context).select_rebalance_data(df)

    context.data = df


performance = bigtrader.run(
    market=bigtrader.Market.CN_STOCK,
    frequency=bigtrader.Frequency.DAILY,
    start_date="2021-02-01",
    end_date="2026-05-07",
    capital_base=300000,
    initialize=initialize,
    handle_data=bigtrader.HandleDataLib.handle_data_weight_based,
    order_price_field_buy='open',
    order_price_field_sell='open',
)


import dai 
sql = "select * from user_factor_sfgsxysnxfhs where date >= '2010-01-01' order by date"
df = dai.query(sql).df() 
df