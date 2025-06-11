import liwc
from collections import Counter
from nltk import word_tokenize
from transformers import BertTokenizer
import re
from datetime import datetime
from urllib.parse import urlparse
import numpy as np
np.bool = np.bool_
import pandas as pd
from transformers import pipeline
import textstat
import string


def clean_tweet(tweet_text):
    tweet_text = remove_links(tweet_text)
    tweet_text = remove_users(tweet_text)
    tweet_text = tweet_text.lower()
    tweet_text = re.sub('[' + string.punctuation + ']+', ' ', tweet_text)
    tweet_text = re.sub('[^a-zA-Z0-9\s]', '', tweet_text)
    tweet_text = re.sub('\s+', ' ', tweet_text)
    return tweet_text.strip()


def remove_links(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'bit.ly/\S+', '', tweet)
    return tweet


def remove_users(tweet):
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    return tweet


def get_accounts_ids(accounts_file, tweets_file):
    accounts_df = pd.read_csv(accounts_file)
    print("the number of accounts: ", len(accounts_df))
    accounts_ids = accounts_df["userid"].tolist()
    gold_accouts = []
    tweets_df = pd.read_csv(tweets_file)
    for i, id in enumerate(accounts_ids):
        tweets = tweets_df[tweets_df['userid'] == id]
        tweets = tweets[tweets["tweet_language"] == "en"]
        if len(tweets) >= 20:
            gold_accouts.append(id)
    return gold_accouts


def get_tweets_count(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file, tweets_file)
    tweets_df = pd.read_csv(tweets_file)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        print(i)
        tweets = tweets_df[tweets_df['userid'] == id]
        tweets_count = len(tweets)
        account_age = get_account_age(tweets)
        df.loc[i, "userid"] = str(id)
        df.loc[i, "tweets_count"] = str(round(tweets_count/account_age, 2))
    output_file = feature_dir + "tweets_count_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def get_account_age(tweets):
    tweets = tweets.sort_values(by='tweet_time', ascending=False)
    last_tweet_date = datetime.strptime(str(tweets["tweet_time"].tolist()[0]), "%Y-%m-%d %H:%M")
    creation_date = datetime.strptime(str(tweets["account_creation_date"].tolist()[0]), "%Y-%m-%d")
    account_age = last_tweet_date - creation_date
    account_age = account_age.days
    return account_age


def get_replies_count(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file, tweets_file)
    tweets_df = pd.read_csv(tweets_file)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        tweets = tweets_df[tweets_df['userid'] == id]
        replies = tweets[pd.notna(tweets['in_reply_to_tweetid'])]
        tweets_count = len(tweets)
        replies_count = len(replies)
        df.loc[i, "userid"] = str(id)
        df.loc[i, "replies_count"] = str(round(replies_count/tweets_count, 2))
    output_file = feature_dir + "replies_count_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def get_retweets_count(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file, tweets_file)
    tweets_df = pd.read_csv(tweets_file)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        tweets = tweets_df[tweets_df['userid'] == id]
        retweets = tweets[tweets['is_retweet'] == True]
        tweets_count = len(tweets)
        retweets_count = len(retweets)
        df.loc[i, "userid"] = str(id)
        df.loc[i, "retweets_count"] = str(round(retweets_count/tweets_count, 2))
    output_file = feature_dir + "retweets_count_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def get_URLs_count(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids_v2(accounts_file,tweets_file)
    tweets_df = pd.read_csv(tweets_file)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        tweets = tweets_df[tweets_df['userid'] == id]
        tweets_count = len(tweets)
        URLs_count = URLs_count_in_tweets(tweets)
        df.loc[i, "userid"] = str(id)
        df.loc[i, "URLs_count"] = str(round(URLs_count/tweets_count, 2))
    output_file = feature_dir + "URLs_count_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def URLs_count_in_tweets(tweets):
    URLs_list = tweets["urls"].tolist()
    URLs_count = 0
    for URLs in URLs_list:
        if str(URLs) == "nan" or str(URLs) == "[]":
            URLs_count += 0
        else:
            URLs = URLs.strip('[]')
            elements = URLs.split(',')
            URLs = [element.strip() for element in elements]
            URLs_count += len(URLs)
    return URLs_count


def get_mentions_count(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file, tweets_file)
    tweets_df = pd.read_csv(tweets_file)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        tweets = tweets_df[tweets_df['userid'] == id]
        tweets_count = len(tweets)
        mentions_count = mentions_count_in_tweets(tweets)
        df.loc[i, "userid"] = str(id)
        df.loc[i, "mentions_count"] = str(round(mentions_count/tweets_count, 2))
    output_file = feature_dir + "mentions_count_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def mentions_count_in_tweets(tweets):
    mentions_list = tweets["user_mentions"].tolist()
    mentions_count = 0
    for mentions in mentions_list:
        if str(mentions) == "nan" or str(mentions) == "[]":
            mentions_count += 0
        else:
            mentions = mentions.strip('[]')
            elements = mentions.split(',')
            mentions = [element.strip() for element in elements]
            mentions_count += len(mentions)
    return mentions_count


def get_tags_count(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file, tweets_file)
    tweets_df = pd.read_csv(tweets_file)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        tweets = tweets_df[tweets_df['userid'] == id]
        tweets_count = len(tweets)
        tags_count = tags_count_in_tweets(tweets)
        df.loc[i, "userid"] = str(id)
        df.loc[i, "tags_count"] = str(round(tags_count/tweets_count, 2))
    output_file = feature_dir + "tags_count_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def tags_count_in_tweets(tweets):
    tags_list = tweets["hashtags"].tolist()
    tags_count = 0
    for tags in tags_list:
        if str(tags) == "nan" or str(tags) == "[]":
            tags_count += 0
        else:
            tags = tags.strip('[]')
            elements = tags.split(',')
            tags = [element.strip() for element in elements]
            tags_count += len(tags)
    return tags_count


def get_followers_to_followees(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file, tweets_file)
    users_df = pd.read_csv(accounts_file)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        user = users_df[users_df['userid'] == id]
        follower_count = user["follower_count"].tolist()[0]
        following_count = user["following_count"].tolist()[0]
        df.loc[i, "userid"] = str(id)
        df.loc[i, "follower_following"] = str(round(follower_count/following_count, 2))
    output_file = feature_dir + "follower_following_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def get_tweet_day(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file, tweets_file)
    tweets_df = pd.read_csv(tweets_file)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        print(i)
        tweets = tweets_df[tweets_df['userid'] == id]
        tweets_count = len(tweets)
        mon, tue, wed, thu, fri, sat, sun = tweets_in_days(tweets)
        df.loc[i, "userid"] = str(id)
        df.loc[i, "mon"] = str(round(mon/tweets_count, 2))
        df.loc[i, "tue"] = str(round(tue/tweets_count, 2))
        df.loc[i, "wed"] = str(round(wed/tweets_count, 2))
        df.loc[i, "thu"] = str(round(thu/tweets_count, 2))
        df.loc[i, "fri"] = str(round(fri/tweets_count, 2))
        df.loc[i, "sat"] = str(round(sat/tweets_count, 2))
        df.loc[i, "sun"] = str(round(sun/tweets_count, 2))
    output_file = feature_dir + "tweet_day_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def get_day(date):
    datetime_obj = datetime.strptime(date, "%Y-%m-%d %H:%M")
    day_of_week = datetime_obj.strftime("%A")
    return day_of_week.lower()[:3]


def tweets_in_days(tweets):
    times = tweets["tweet_time"]
    day_counts = {'mon': 0, 'tue': 0, 'wed': 0, 'thu': 0, 'fri': 0, 'sat': 0, 'sun': 0}
    for time in times:
        day = get_day(time)
        day_counts[day] += 1
    return day_counts['mon'], day_counts['tue'], day_counts['wed'], day_counts['thu'], day_counts['fri'], day_counts['sat'],day_counts['sun']


def get_clients(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file, tweets_file)
    tweets_df = pd.read_csv(tweets_file)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        tweets = tweets_df[tweets_df['userid'] == id]
        tweets_count = len(tweets)
        web_client, web_app, android, iphone, deck = clients_in_tweets(tweets)
        sum = web_client+ web_app+ android+ iphone+ deck
        df.loc[i, "userid"] = str(id)
        df.loc[i, "web_client"] = str(round(web_client/sum, 2))
        df.loc[i, "web_app"] = str(round(web_app/sum, 2))
        df.loc[i, "android"] = str(round(android/sum, 2))
        df.loc[i, "iphone"] = str(round(iphone/sum, 2))
        df.loc[i, "deck"] = str(round(deck/sum, 2))
    output_file = feature_dir + "clients_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def clients_in_tweets(tweets):
    clients = tweets["tweet_client_name"]
    web_client = web_app = android = iphone = deck = 0
    for client in clients:
        client = str(client).lower()
        if "deck" in client:
            deck += 1
        elif "iphone" in client:
            iphone += 1
        elif "android" in client:
            android += 1
        elif "web app" in client:
            web_app += 1
        elif "web client" in client:
            web_client += 1
    return web_client, web_app, android, iphone, deck


def get_tweet_length(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file)
    tweets_df = pd.read_csv(tweets_file)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        tweets = tweets_df[tweets_df['userid'] == id]
        tweets_count = len(tweets)
        texts = tweets["tweet_text"]
        words_count = 0
        for text in texts:
            words_count += len(str(text).split())
        df.loc[i, "userid"] = str(id)
        df.loc[i, "length"] = str(round(words_count/tweets_count, 2))
    output_file = feature_dir + "tweet_length_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def get_emotion_analysis(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file, tweets_file)
    tweets_df = pd.read_csv(tweets_file)
    emotion_analyser = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                                return_all_scores=True, truncation=True)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        tweets = tweets_df[tweets_df['userid'] == id]
        tweets = tweets[tweets["tweet_language"] == "en"]
        e1, e2, e3, e4, e5, e6, e7 = tweets_emotions(tweets, emotion_analyser)
        print(e1, e2, e3, e4, e5, e6, e7)
        df.loc[i, "userid"] = str(id)
        df.loc[i, "e1"] = str(round(e1 , 2))
        df.loc[i, "e2"] = str(round(e2 , 2))
        df.loc[i, "e3"] = str(round(e3 , 2))
        df.loc[i, "e4"] = str(round(e4 , 2))
        df.loc[i, "e5"] = str(round(e5 , 2))
        df.loc[i, "e6"] = str(round(e6 , 2))
        df.loc[i, "e7"] = str(round(e7 , 2))
    output_file = feature_dir + "emotions_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def tweets_emotions(tweets, emotion_analyser):
    texts = tweets["tweet_text"]
    emotions = {
        'anger': 0,
        'disgust': 0,
        'fear': 0,
        'joy': 0,
        'neutral': 0,
        'sadness': 0,
        'surprise': 0,
    }
    i = 0
    for text in texts:
        es = emotion_analyser(text)[0]
        for e in es:
            emotions[e['label']] += e['score']
        i += 1
    return emotions['anger']/i, emotions['disgust']/i, emotions['fear']/i, emotions['joy']/i, emotions['neutral']/i, emotions[
        'sadness']/i, emotions['surprise']/i


def get_sentiment_analysis(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file,tweets_file)
    tweets_df = pd.read_csv(tweets_file)
    sentiment_analyser = pipeline("sentiment-analysis")
    df = pd.DataFrame()
    output_file = feature_dir + "sentiments_" + accounts_file.split("/")[1]
    for i, id in enumerate(accounts_ids):
        tweets = tweets_df[tweets_df['userid'] == id]
        tweets = tweets[tweets["tweet_language"] == "en"]
        p, n = tweets_sentiments(tweets, sentiment_analyser)
        df.loc[0, "userid"] = str(id)
        df.loc[0, "p"] = str(round(p, 2))
        df.loc[0, "n"] = str(round(n, 2))
        df.to_csv(output_file, mode='a', header= False, index=False)


def tweets_sentiments(tweets, sentiment_analyser):
    texts = tweets["tweet_text"]
    i = 0
    sentiments = {
        'POSITIVE': 0,
        'NEGATIVE': 0
    }
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for text in texts:
        tokens = tokenizer.tokenize(text)
        text = clean_tweet(text)
        s = sentiment_analyser(text)[0]
        sentiments[s['label']] += 1
        i += 1
    return sentiments['POSITIVE']/i, sentiments['NEGATIVE']/i


def calculate_readability(tweet):
    readability_score = textstat.flesch_reading_ease(tweet)
    return readability_score


def get_readability(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file, tweets_file)
    tweets_df = pd.read_csv(tweets_file)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        tweets = tweets_df[tweets_df['userid'] == id]
        tweets = tweets[tweets["tweet_language"] == "en"]
        texts = tweets["tweet_text"]
        readability_score = 0
        counter = 0
        for text in texts:
            text = clean_tweet(text)
            score = calculate_readability(text)
            if score != 0:
                readability_score += score
                counter += 1
        df.loc[i, "userid"] = str(id)
        df.loc[i, "readability"] = str(round(readability_score/counter, 2))
    output_file = feature_dir + "readability_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def get_linguistic_styles(accounts_file, tweets_file):
    accounts_ids = get_accounts_ids(accounts_file,tweets_file)
    tweets_df = pd.read_csv(tweets_file)
    df = pd.DataFrame()
    for i, id in enumerate(accounts_ids):
        tweets = tweets_df[tweets_df['userid'] == id]
        tweets = tweets[tweets["tweet_language"] == "en"]
        informal, drives, personal, perceptual, cognitive, social, pronouns, functions = linguistic_styles(tweets)
        df.loc[i, "userid"] = str(id)
        df.loc[i, "informal"] = str(round(informal, 2))
        df.loc[i, "drives"] = str(round(drives, 2))
        df.loc[i, "personal"] = str(round(personal, 2))
        df.loc[i, "perceptual"] = str(round(perceptual, 2))
        df.loc[i, "cognitive"] = str(round(cognitive, 2))
        df.loc[i, "social"] = str(round(social, 2))
        df.loc[i, "pronouns"] = str(round(pronouns, 2))
        df.loc[i, "functions"] = str(round(functions, 2))
    output_file = feature_dir + "linguistic_styles_" + accounts_file.split("/")[1]
    df.to_csv(output_file, index=False)


def linguistic_styles(tweets):
    parse, category_names = liwc.load_token_parser('LIWC2015_English_Flat.dic')
    texts = tweets["tweet_text"]
    liwc_values = {
        'Informal language': 0,
        'Drives': 0,
        'Personal concerns': 0,
        'Perceptual processes': 0,
        'Cognitive processes': 0,
        'Social processes': 0,
        'Total pronouns': 0,
        'Total function words': 0 }

    for text in texts:
        text = clean_tweet(text)
        word_tokens = word_tokenize(text)
        liwc_counter = Counter(category for token in word_tokens for category in parse(token))

        if liwc_counter['informal'] > 0 or liwc_counter['netspeak'] > 0 or liwc_counter['assent'] > 0 or \
                liwc_counter['nonflu'] > 0 or liwc_counter['filler'] > 0 or liwc_counter['swear'] > 0:
            liwc_values['Informal language'] = liwc_values['Informal language'] + 1

        if liwc_counter['drives'] > 0 or liwc_counter['affiliation'] > 0 or liwc_counter['power'] > 0 or \
                liwc_counter['reward'] > 0 or liwc_counter['risk'] > 0 or liwc_counter['achiev'] > 0:
            liwc_values['Drives'] = liwc_values['Drives'] + 1

        if liwc_counter['work'] > 0 or liwc_counter['leisure'] > 0 or liwc_counter['home'] > 0 or \
                liwc_counter['money'] > 0 or liwc_counter['relig'] > 0 or liwc_counter['death'] > 0:
            liwc_values['Personal concerns'] = liwc_values['Personal concerns'] + 1

        if liwc_counter['percept'] > 0 or liwc_counter['see'] > 0 or liwc_counter['hear'] > 0 or liwc_counter['feel'] > 0:
            liwc_values['Perceptual processes'] = liwc_values['Perceptual processes'] + 1

        if liwc_counter['cogproc'] > 0 or liwc_counter['insight'] > 0 or liwc_counter['cause'] > 0 or liwc_counter['discrep'] > 0 or \
                liwc_counter['differ'] > 0 or liwc_counter['tentat'] > 0 or liwc_counter['certain'] > 0:
            liwc_values['Cognitive processes'] = liwc_values['Cognitive processes'] + 1

        if liwc_counter['family'] > 0 or liwc_counter['friend'] > 0 or liwc_counter['social'] > 0 or \
                liwc_counter['female'] > 0 or liwc_counter['male'] > 0:
            liwc_values['Social processes'] = liwc_values['Social processes'] + 1

        if liwc_counter['pronoun'] > 0 or liwc_counter['ppron'] > 0 or liwc_counter['i'] > 0 or liwc_counter['we'] > 0\
                or liwc_counter['you'] > 0 or liwc_counter['shehe'] > 0 or liwc_counter['they'] > 0:
            liwc_values['Total pronouns'] = liwc_values['Total pronouns'] + 1

        if liwc_counter['pronoun'] > 0 or liwc_counter['ppron'] > 0 or liwc_counter['i'] > 0 or liwc_counter['we'] > 0 \
                or liwc_counter['you'] > 0 or liwc_counter['shehe'] > 0 or liwc_counter['they'] > 0 or liwc_counter['funct'] > 0 \
                or liwc_counter['ipron'] > 0 or liwc_counter['article'] > 0 or liwc_counter['prep'] > 0 or liwc_counter['auxverb'] > 0 \
                or liwc_counter['adverb'] > 0 or liwc_counter['conj'] > 0 or liwc_counter['negate'] > 0:
            liwc_values['Total function words'] = liwc_values['Total function words'] + 1

    c = len(texts)
    return liwc_values['Informal language']/c, liwc_values['Drives']/c, liwc_values['Personal concerns']/c, liwc_values['Perceptual processes']/c,\
           liwc_values['Cognitive processes'] / c, liwc_values['Social processes'] / c, liwc_values['Total pronouns'] / c, liwc_values['Total function words'] / c
    
