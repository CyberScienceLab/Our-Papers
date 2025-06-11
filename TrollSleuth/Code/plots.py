import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def box_plot_feature(feature_name):
    df = pd.read_csv(feature_dir + feature_name + "_iran_users_oct_2018.csv")
    Iran = df[feature_name].tolist()
    df = pd.read_csv(feature_dir + feature_name + "_russia_users_jan_2019.csv")
    Russia = df[feature_name].tolist()
    df = pd.read_csv(feature_dir + feature_name + "_indonesia_users_apr_2020.csv")
    Indonesia = df[feature_name].tolist()
    df = pd.read_csv(feature_dir + feature_name + "_uganda_users_dec_2021.csv")
    Uganda = df[feature_name].tolist()

    data = {
        'campaign': ['Iran'] * len(Iran) + ['Russia'] * len(Russia) + ['Indonesia'] * len(Indonesia) + ['Uganda'] * len(Uganda),
        feature_name: Iran + Russia + Indonesia + Uganda
    }
    df = pd.DataFrame(data)
    base_color = sns.color_palette("Blues", 4)
    ax = sns.boxplot(data=df, x='campaign', y=feature_name, palette=base_color, showfliers=False)
    plt.xlabel('Campaign', fontsize=12)
    plt.ylabel("Readability", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()


def bar_plot_activity_day():
    labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    num_vars = len(labels)
    base_color = sns.color_palette("Blues", 5)
    plt.figure(figsize=(9, 4))
    bar_width = 0.17
    index = np.arange(num_vars)
    plt.bar(index, values1, bar_width, label='Iran', color=base_color[1])
    plt.bar(index + bar_width, values2, bar_width, label='Russia', color=base_color[2])
    plt.bar(index + 2 * bar_width, values3, bar_width, label='Indonesia', color=base_color[3])
    plt.bar(index + 3 * bar_width, values4, bar_width, label='Uganda', color=base_color[4])
    plt.xlabel('Days')
    plt.ylabel('Proportion')
    plt.xticks(index + 1.5 * bar_width, labels)  
    plt.legend(loc='upper right', fontsize='small')
    plt.show()


def pie_chart_activity_time():
    campaigns = ['Iran', 'Russia', 'Indonesia', 'Uganda']
    time_intervals = ['T1', 'T2', 'T3', 'T4']
    interval_descriptions = {
        'T1': '0-6',
        'T2': '6-12',
        'T3': '12-18',
        'T4': '18-24'
    }
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    for i, campaign in enumerate(campaigns):
        total_tweets = [time1[i], time2[i], time3[i], time4[i]]
        wedges, texts, autotexts = axs[i].pie(total_tweets, autopct='%1.1f%%',
                                              colors=sns.color_palette("Blues", 4),
                                              wedgeprops={'linewidth': 0.7, 'edgecolor': 'black'})

        axs[i].set_title(campaign)
        axs[i].axis('equal')  

    legend_handles = [interval_descriptions[interval] for interval in time_intervals]
    fig.legend(wedges, legend_handles, title="Time Interval", loc='center right')
    plt.tight_layout(rect=[0, 0, 0.9, 0.9])  
    plt.subplots_adjust(wspace=0.1)  
    plt.show()


def plot_client_utilization():
    df_iran = pd.read_csv(feature_dir + "clients_iran_users_oct_2018.csv")
    df_russia = pd.read_csv(feature_dir + "clients_russia_users_jan_2019.csv")
    df_indonesia = pd.read_csv(feature_dir + "clients_indonesia_users_apr_2020.csv")
    df_uganda = pd.read_csv(feature_dir + "clients_uganda_users_dec_2021.csv")
    clients = ['web_client', 'web_app', 'android', 'iphone', 'deck']
    campaigns = ['Iran', 'Russia', 'Indonesia', 'Uganda']
    clients = ['web_client', 'web_app', 'android', 'iphone', 'deck']
    data = pd.DataFrame(index=campaigns, columns=clients)
    for client in clients:
        data[client] = [
            df_iran[client].mean(),
            df_russia[client].mean(),
            df_indonesia[client].mean(),
            df_uganda[client].mean()]
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(campaigns))
    colors = sns.color_palette("Blues", 5)
    for i, client in enumerate(clients):
        ax.bar(campaigns, data[client], bottom=bottom, label=client, color=colors[4 - i])
        bottom += data[client]
    ax.set_xlabel('Campaigns', fontsize=14)
    ax.set_ylabel('Client Usage', fontsize=14)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)
    plt.tight_layout()
    plt.show()


def violin_plot_readability():
    df_iran = pd.read_csv(feature_dir + "readability_iran_users_oct_2018.csv")
    Iran = df_iran[feature_name].tolist()

    df_russia = pd.read_csv(feature_dir + "readability_russia_users_jan_2019.csv")
    Russia = df_russia[feature_name].tolist()

    df_indonesia = pd.read_csv(feature_dir + "readability_indonesia_users_apr_2020.csv")
    Indonesia = df_indonesia[feature_name].tolist()

    df_uganda = pd.read_csv(feature_dir + "readability_uganda_users_dec_2021.csv")
    Uganda = df_uganda[feature_name].tolist()

    data = {
        'Campaign': ['Iran'] * len(Iran) + ['Russia'] * len(Russia) + ['Indonesia'] * len(Indonesia) + ['Uganda'] * len(
            Uganda),
        feature_name: Iran + Russia + Indonesia + Uganda
    }
    df = pd.DataFrame(data)
    base_color = sns.color_palette("Blues", 4)
    plt.figure(figsize=(10, 4))
    sns.violinplot(x='Campaign', y='Readability', data=df, palette=base_color)
    plt.xlabel('Campaign')
    plt.ylabel('Readability')
    plt.show()



def violin_plot_liwc_features():
    df1 = pd.read_csv(feature_dir + "linguistic_styles_iran_users_oct_2018.csv")
    df2 = pd.read_csv(feature_dir + "linguistic_styles_russia_users_jan_2019.csv")
    df3 = pd.read_csv(feature_dir + "linguistic_styles_indonesia_users_apr_2020.csv")
    df4 = pd.read_csv(feature_dir + "linguistic_styles_uganda_users_dec_2021.csv")
    features = ['informal', 'personal', 'perceptual', 'cognitive', 'social', 'functions']
    data = {
        'Score': [],
        'Feature': [],
        'Campaign': []
    }
    def add_data(df, campaign_name):
        for feature in features:
            scores = df[feature].tolist()
            data['Score'].extend(scores)
            data['Feature'].extend([feature] * len(scores))
            data['Campaign'].extend([campaign_name] * len(scores))

    add_data(df1, 'Iran')
    add_data(df2, 'Russia')
    add_data(df3, 'Indonesia')
    add_data(df4, 'Uganda')
    df = pd.DataFrame(data)
    plt.figure(figsize=(11, 5))
    sns.violinplot(x='Feature', y='Score', hue='Campaign', data=df, dodge=True, palette='Blues', inner=None,
                   linewidth=1.1, scale='width', width=0.5)
    sns.despine(offset=10, trim=True)
    plt.grid(False)
    plt.xlabel('', fontsize=14)
    plt.ylabel('', fontsize=14)
    feature_labels = [
        'Informal language',
        'Personal concerns',
        'Perceptual processes',
        'Cognitive Processes',
        'Social Processes',
        'Total function words'
    ]
    plt.xticks(ticks=range(len(features)), labels=feature_labels, fontsize='9')
    plt.legend(title='Campaign', title_fontsize='9', fontsize='9', loc='upper left')
    plt.tight_layout()
    plt.show()
