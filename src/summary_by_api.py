from utils.utils import MySQLAgent
from src.queries.queries import NOV_NEWS_TEST, NOV_50_NEWS

from llm_api.config.llm_config import LLMConfig
from llm_api.summary_bot import SummaryBot
from llm_api.prompts.prompts import SUMMARY_PROMPT
from utils.utils import get_db_data

def genearte_summary_by_api():

    summary_bot = SummaryBot(config=LLMConfig())

    df_news = get_db_data(query=NOV_NEWS_TEST)

    df_news['content_summary'] = ""

    for idx, row in df_news.iterrows():

        content_summary = summary_bot.summarize(row['content'], prompt=SUMMARY_PROMPT)

        df_news.at[idx, 'content_summary'] = content_summary
        

    sql_agent = MySQLAgent()
    table_name = 'news_summary'
    sql_agent.write_table(df_news, table_name=table_name, if_exists='append', index=None, data_type=None)
    
    print(f"Successfully wrote to '{table_name}' table with {len(df_news)} rows.")

