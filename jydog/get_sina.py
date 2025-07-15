import akshare as ak
import pandas as pd
from datetime import datetime
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

################################################################################
# 获取股票代码列表

def get_stock_list():
    """
    获取A股实时行情数据，并提取股票代码和名称。
    """
    stock_list = ak.stock_zh_a_spot()
    stock_list['交易所'] = stock_list['代码'].str[:2]
    stock_list['代码'] = stock_list['代码']
    return stock_list

################################################################################
# 获取历史数据

def get_historical_data(code, output_file, start_date, end_date):
    """
    根据股票代码获取历史行情数据，并将其追加到指定的CSV文件中。
    """
    try:
        his_data = ak.stock_zh_a_daily(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
        if his_data.empty:
            logging.warning(f"No data available for {code}")
            return
        his_data['code'] = code
        his_data.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
        logging.info(f"Data for {code} has been written to {output_file}")
    except Exception as e:
        logging.error(f"Error fetching data for {code}: {e}")

################################################################################
# 主函数

def main():
    """
    主函数，负责将股票代码列表拆分成多个块，并处理所有块。
    """
    num_chunks = 5  # 默认分块数量
    today = datetime.today().strftime('%Y%m%d')
    today2 = datetime.today().strftime('%Y%m%d%H')
    output_file = f'data/sinadata_{today2}.csv'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 创建一个空的CSV文件，写入表头
    header = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'outstanding_share', 'turnover', 'code']
    pd.DataFrame(columns=header).to_csv(output_file, index=False, encoding='utf-8')

    # 获取股票列表
    stock_list = get_stock_list()
    all_codes = stock_list[['代码', '名称']]

    # 设置日期范围
    start_date = "19910403"
    end_date = today

    # 将 all_codes 拆分成多个任务
    chunk_size = len(all_codes) // num_chunks
    chunks = [all_codes[i:i + chunk_size] for i in range(0, len(all_codes), chunk_size)]

    # 处理所有块
    for chunk_index, chunk in enumerate(chunks, start=1):
        logging.info(f"Processing chunk {chunk_index}/{num_chunks}")
        for code in chunk['代码'].tolist():
            get_historical_data(code, output_file, start_date, end_date)

    logging.info(f"All data has been fetched and written to {output_file}.")

################################################################################
# 入口点

if __name__ == "__main__":
    main()
