from transformers import BertTokenizer
from matching_algorithm.model.bert_avg import BERT_BiLSTM_Attention
import torch
model_path = "./ner/RoBERTa_zh"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_nn = BERT_BiLSTM_Attention(model_path, 768, 0).to(device)
model_nn.load_state_dict(torch.load('./matching_algorithm/weights/best_model_1.pt'), strict=False)
tokenizer = BertTokenizer.from_pretrained(model_path)
def compute_match_score(text1, text2):
    cutlen = len(text1) if len(text1) > len(text2) else len(text2)
    encoders = tokenizer.encode_plus(str(text1), max_length=cutlen, truncation=True,
                                        padding='max_length')
    encoders1 = tokenizer.encode_plus(str(text2), max_length=cutlen, truncation=True,
                                        padding='max_length')
    input_ids1 = [encoders['input_ids']]
    input_ids2 = [encoders1['input_ids']]
    attention_mask1 = [encoders['attention_mask']]
    attention_mask2 = [encoders1['attention_mask']]
    token_type_ids1 = [encoders['token_type_ids']]
    token_type_ids2 = [encoders1['token_type_ids']]

    input_ids1 = torch.tensor(input_ids1).int().to(device)
    input_ids2 = torch.tensor(input_ids2).int().to(device)
    token_type_ids1 = torch.tensor(token_type_ids1).int().to(device)
    token_type_ids2 = torch.tensor(token_type_ids2).int().to(device)
    attention_mask1 = torch.tensor(attention_mask1).int().to(device)
    attention_mask2 = torch.tensor(attention_mask2).int().to(device)
    with torch.no_grad():
        output = model_nn(input_ids1, input_ids2,token_type_ids1,token_type_ids2, attention_mask1, attention_mask2)
    match_score = torch.nn.functional.softmax(output, dim=1)[0][1].item()
    return match_score
if __name__ == '__main__':
    text1 = '林钰婷社会化媒体运营推广北京海淀教育背景汉语言文学北京师范大学东城警察电视专题栏目负责联系媒体新闻事件集中报道风飞雨工作室带领人小组组建风飞雨工作室致力于品牌建设市场营销搭建工作网站常态运营活动规划推进执行制定相关运营策略指标实习经验工作经验推广实习生宣传科实习生产品运营负责人奖项荣誉奖学金学年学年获国家奖学金活动获奖校企广告策划比赛公益作品创意奖团体奖项移动校园营销大赛团体评价博种子用户活跃论坛天涯知乎豆瓣扒皮能手文笔流畅刊物发表文章软文写作善于沟通具有团队合作能力人际交往能力软件技能职业技能团队协作创新精神沟通能力组织能力'
    text2 = '林钰婷社会化媒体运营推广北京海淀教育背景汉语言文学北京师范大学东城警察电视专题栏目负责联系媒体新闻事件集中报道风飞雨工作室带领人小组组建风飞雨工作室致力于品牌建设市场营销搭建工作网站常态运营活动规划推进执行制定相关运营策略指标实习经验工作经验推广实习生宣传科实习生产品运营负责人奖项荣誉奖学金学年学年获国家奖学金活动获奖校企广告策划比赛公益作品创意奖团体奖项移动校园营销大赛团体评价博种子用户活跃论坛天涯知乎豆瓣扒皮能手文笔流畅刊物发表文章软文写作善于沟通具有团队合作能力人际交往能力软件技能职业技能团队协作创新精神沟通能力组织能力'
    match_score = compute_match_score(text1, text2)
    print(match_score)