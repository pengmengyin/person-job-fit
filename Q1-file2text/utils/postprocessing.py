import re


def post_processing(result, sentence):
    result_dic = {}
    edu_sch_list = []
    name = ''
    age = ''
    edu = ''
    sch = ''
    birth_day = ''
    education_list = ['无', '小学', '初中', '高中', '中专', '大专', '本科', '学士', '硕士', '博士']
    month = 0
    flag = -1
    temp_list = []
    for i in result:
        if len(i) != 0:
            for j in i:
                temp_list.append(j)
    result = temp_list
    for item in result:
        nums = re.findall(r'\d+', item[0])
        if 'YEARSOFWORK' == item[1]:
            date = date_validation(sentence, item[0])
            nums = re.findall(r'\d+', date)
            num = ''.join(nums)
            if len(num) == 8:
                years = int(num[4:]) - int(num[:4]) + 1
                months = 12 * years
            elif len(num) == 10:
                months = (int(num[5:9]) * 12 + int(num[9])) - (int(num[:4]) * 12 + int(num[4]))
            elif len(num) == 12:
                months = (int(num[6:10]) * 12 + int(num[10])) - (int(num[:4]) * 12 + int(num[4]))
            elif len(num) == 9:
                years = int(num[4:8]) - int(num[:4]) + 1
                months = 12 * years
                if months < 0 or months > 100:
                    years = int(num[5:]) - int(num[:4]) + 1
                    months = 12 * years
            elif len(num) == 4:
                months = 2023 * 12 - int(num) * 12
            elif len(num) == 5:
                months = 2023 * 12 + 4 - int(num[0:4]) * 12 - int(num[4])
            elif len(num) == 6:
                months = 2023 * 12 + 4 - int(num[0:4]) * 12 - int(num[4:])
            else:
                months = 0
            if 0 < months < 120:
                month += months
        elif 'SCHOOL' == item[1]:
            edu_sch_list.append(item)
        elif 'EDUCATION' == item[1]:
            edu_sch_list.append(item)
        elif 'AGE' == item[1]:
            num = ''.join(nums)
            if len(num) >= 4:
                age = 2023 - int(num[:4]) + 1
            else:
                age = num
        elif 'NAME' == item[1]:
            if not item[0].isdigit():
                name = item[0]
        elif 'BIRTHDAY' == item[1]:
            num = ''.join(nums)
            if len(num) > 3:
                birth_day = 2023 - int(num[:4]) + 1
    for item in edu_sch_list:
        if item[1] == 'EDUCATION':
            data = item[0]
            if data == '本':
                data = '本科'
            if data[-3:] == '研究生':
                data = data[:-3]
            index = education_list.index(data)
            if index > flag:
                flag = index
    pre_count = 0
    end_count = 0
    edu_flag = 0
    edu_flag_1 = 0
    count = 0
    if edu_sch_list:
        for item in edu_sch_list:
            if item[0] == education_list[flag]:
                edu_flag = 1
                edu_flag_1 = count

            if edu_flag == 0:
                if item[1] == 'EDUCATION':
                    pre_count += 1
                else:
                    pre_count = 0
            else:
                if item[1] == 'EDUCATION':
                    end_count += 1
                else:
                    break
            count += 1
        edu = education_list[flag]
        if pre_count <= end_count:
            sch = edu_sch_list[edu_flag_1 - pre_count - 1][0]
        else:
            sch = edu_sch_list[end_count - edu_flag_1][0]
    result_dic['Name'] = name
    result_dic['SCHOOL'] = sch
    if edu == '学士':
        result_dic['Education'] = '本科'
    elif edu == '':
        result_dic['Education'] = '无'
    else:
        result_dic['Education'] = edu
    if (month % 12) > 0:
        result_dic['YEARSOFWORK'] = month // 12 + 1
    else:
        result_dic['YEARSOFWORK'] = month / 12
    if birth_day != '':
        result_dic['Age'] = birth_day
    elif age != '':
        result_dic['Age'] = age
    else:
        result_dic['Age'] = match_age(sentence)
    return result_dic


def match_age(sentence):
    pattern = r'(?<!年)(?<!月)[\u4e00-\u9fa5]+(\d{2})[\u4e00-\u9fa5]+(?!\d)'
    matches = re.findall(pattern, sentence)
    age = ''
    if matches:
        for match in matches:
            age = match
            break
    else:
        print('没有匹配的数字。')
    return age


def date_validation(sentence, date):
    pos = sentence.find(date)  # 查找子字符串 '哈哈' 的位置
    valid = ''
    while pos != -1:
        pos = pos - 1
        if sentence[pos].isdigit():
            valid = sentence[pos] + valid
        else:
            pos = -1
    return valid + date
