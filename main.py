import heapq
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT
import json
import re
from gensim.models import KeyedVectors
import opencc

# 指定詞向量檔案的路徑
w2v_model_file = "w2v_CNA_ASBC_300d.vec"  # 替換為實際的 w2v 模型檔案路徑
# 載入詞向量模型
w2v_model = KeyedVectors.load_word2vec_format(
    w2v_model_file, encoding='utf-8', unicode_errors='ignore')


Tag = ["海大", "機車", "拉麵"]  # 輸入使用者興趣
Type = {'籃球': 0, '男女': 1, '閒聊': 2, '車': 3, '八卦': 4, '政治': 5,
        '房價': 6, '日本行': 7, '女優': 8, '韓星': 9, '省錢': 10,
        '英雄': 11, '婚姻': 12, '軍事': 13, '電影': 14, '籃球': 15,
        '網購': 16, '性愛': 17, '股票': 18, '科技': 19, '女人': 20, '棒球': 21}  # 看板

recommend = []
# 將興趣與看板進行語意相似度比較，對於每一個興趣選出一個最有興趣的看板
for Interest in Tag:
    largest = 0
    for t in Type:
        similarity_score = w2v_model.similarity(Interest, t)
        print(Interest, "跟看板", t, "相關分數: ", similarity_score)  # 目前興趣跟看版的相關度
        if (largest < similarity_score):
            # print("目前最高分為", largest, "已更新成", similarity_score)  # 選出最有關連的看版
            largest = similarity_score
            board = t  # 儲存關聯最高的版面

    print(Interest + "最有興趣的看板是", board)
    recommend.append(Type[board])  # 有興趣的是哪些看板

for i in recommend:
    print("利用三個興趣計算出的興趣看板", i)


# 讀入某一段時間內的資料集
club_list = [
    {
        'club_name': "Baseball",
        'start': 5000,
        'end': 13000
    },
    {
        'club_name': "Boy-Girl",
        'start': 4000,
        'end': 6000
    },
    {
        'club_name': "C_Chat",
        'start': 2000,
        'end': 17700
    },
    {
        'club_name': "car",
        'start': 2000,
        'end': 5000
    },
    {
        'club_name': "Gossiping",
        'start': 4500,
        'end': 39000
    },
    {
        'club_name': "HatePolitics",
        'start': 400,
        'end': 4000
    },
    {
        'club_name': "home-sale",
        'start': 2500,
        'end': 5000
    },
    {
        'club_name': "Japan_Travel",
        'start': 6000,
        'end': 7500
    },
    {
        'club_name': "japanavgirls",
        'start': 100,
        'end': 1000
    },
    {
        'club_name': "KoreaStar",
        'start': 1000,
        'end': 2200
    },
    {
        'club_name': "Lifeismoney",
        'start': 2900,
        'end': 3900
    },
    {
        'club_name': "LOL",
        'start': 7000,
        'end': 15000
    },
    {
        'club_name': "marriage",
        'start': 1800,
        'end': 2800
    },
    {
        'club_name': "Military",
        'start': 700,
        'end': 2000
    },
    {
        'club_name': "movie",
        'start': 7000,
        'end': 9500
    },
    {
        'club_name': "NBA",
        'start': 4000,
        'end': 6500
    },
    {
        'club_name': "PC_Shopping",
        'start': 1000,
        'end': 4000
    },
    {
        'club_name': "sex",
        'start': 3000,
        'end': 4000
    },
    {
        'club_name': "Stock",
        'start': 4000,
        'end': 6100
    },
    {
        'club_name': "Tech_Job",
        'start': 3000,
        'end': 4000
    },
    {
        'club_name': "WomenTalk",
        'start': 3000,
        'end': 7500
    },
    {
        'club_name': "Baseball",
        'start': 5100,
        'end': 12900
    }
]

documents = []


def get_file(start, club):  # 一次讀取100篇進行處理
    # 將程式碼放在與資料集相同的資料夾，並修改為實際檔案路徑
    with open('C:/Recommendation System/'+str(club) + '/' + str(club) + ' ' + str(start) + '-' + str(start+100) + '.json') as file:
        data = json.load(file)
        return data


# 寫正則表示，去除文章標題中 []
def regular(text):
    dataset = []
    patterns = r"\[[^\]]*\]|^Re:\s*"  # 正則表達式模式，表示要匹配 "[閒聊]"
    for i in text:
        i["title"] = re.sub(patterns, "", i["title"])


article = []
temp = []
for i in recommend:  # 讀取使用者有興趣看板
    times = 0
    temp = []
    for j in range(3):
        data = get_file(club_list[i]["start"] + times,
                        club_list[i]["club_name"])
        regular(data)
        temp = data + temp
        times = times + 100

    article = article + temp  # 集合所有已清洗的文章
    print(len(article))
    # print(data[0]["title"])  # 顯示原始文章
    # regular(data)
    #print("清洗完後 :" + data[0]["title"])


def tokenize_zh(text):
    sentence = jieba.lcut(text)
    return sentence


result_dict = {}
matrix = []
vectorizer = CountVectorizer(tokenizer=tokenize_zh)
kw_model = KeyBERT()
time = 0  # 記錄第幾篇文章

for i in article:  # 讀取每一篇文章
    str1 = ''
    str2 = ''
    max1 = 0
    max2 = 0
    score1 = 0
    score2 = 0
    keywords = kw_model.extract_keywords(
        i["title"], vectorizer=vectorizer)  # 從文章取關鍵字
    print("文章標題")
    print(i["title"])

    for j in keywords:
        if j[1] > max1:  # 選出前二高的相關詞
            max2 = max1
            max1 = j[1]
            str2 = str1
            str1 = j[0]
        elif j[1] > max2:
            max2 = j[1]
            str2 = j[0]

    print("關鍵詞 : ", str1, str2)

    if str1 in w2v_model.key_to_index:  # 先確認關鍵詞有在字典
        # 各個文章關鍵詞比對興趣
        for Interest in Tag:
            score1 = score1 + w2v_model.similarity(str1, Interest)

    if str2 in w2v_model.key_to_index:
        for Interest in Tag:
            score2 = score2 + w2v_model.similarity(str2, Interest)

    result_dict[time] = score1 + score2
    # print("分數", score1 + score2)
    if str1 in w2v_model.key_to_index:  # 先確認關鍵詞有在字典
        for Interest in Tag:  # 各興趣跟關鍵詞比對
            score1 = score1 + w2v_model.similarity(str1, Interest)
    # print("分數", score1) # print("第", time, "文章結束")
    result_dict[time] = score1

    time = time+1

recommend_acticle = heapq.nlargest(15, result_dict.items(), key=lambda x: x[1])


i = 0
for key, value in recommend_acticle:
    i = i+1
    print("第", i, "篇推薦的文章：" + article[key]["title"])
    print(article[key]["link"])
    print("\n")
