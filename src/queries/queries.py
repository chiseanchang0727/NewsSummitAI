NOV_NEWS_TEST="""
    SELECT id, title, section, content, news_date
    FROM news_triplets
    WHERE MONTH(news_date) = 11 
    AND YEAR(news_date) = 2024
    AND id NOT IN (SELECT id FROM news_summary)
    LIMIT 3
"""

NOV_50_NEWS="""
    SELECT id, title, section, content, news_date
    FROM news_triplets
    WHERE MONTH(news_date) = 11 AND YEAR(news_date) = 2024
    LIMIT 50
"""