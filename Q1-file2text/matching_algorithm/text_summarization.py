#-*- encoding:utf-8 -*-
import re
from matching_algorithm.textrank4zh import TextRank4Keyword
def text_summarization(text):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, lower=True, window=2)
    results = []
    for words in tr4w.words_all_filters:
        summar = ''.join(words)
        summar = re.sub(r'[0-9a-zA-Z]{10,}', '', summar)
        results.append(summar)
    return ''.join(results)

if __name__ == '__main__':
    text = "林钰婷  社会化媒体运营推广  1993.02 北京海淀 138-0013 -8000  zxj2015@gmail.comweibo.com /qingcv  linked.in/qingcvfacebook.com /qingcv  twitter.com /qingcv    教育背景 汉语言文学  2012-2016  北京师范大学              东城警察》电视专题栏目及负责联系各大媒体对新闻事件的集中报道；     风飞雨工作室  带领 4人小组组建了风飞雨工作室，主要致力于品牌建设，以及市场营销搭建工作； 网站常态运营活动规划和推进执行，制定相关运营策略和指标；    实习经验    工作经验  推广组实习生  2015.04 -2016.06  宣传科实习生  2014.04 -2014.09  产品运营负责人  2016.07-至今   奖项荣誉 奖学金  2012-2015学年  连续 3个学年获国家一等奖学金；   活动获奖 2015.09    校企面对面广告策划比赛公益类作品最佳创意奖  团体奖项 2015.09   移动杯校园营销大赛团体第一名    自我评价 微博第一批种子用户，活跃于各种论坛，天涯，知乎，豆瓣，扒皮小能手一枚；  文笔流畅，多次在校内刊物上发表文章，熟练软文写作；  善于沟通、具有良好的团队合作能力和人际交往能力。  软件技能  photoshop   90% ppt  80% ppt  75% 职业技能 团队 协作 创新 精神 沟通 能力 组织 能力"
    print(text_summarization(text))