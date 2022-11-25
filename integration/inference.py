import os
import folium
import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import pycaret.classification
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from folium.plugins import MarkerCluster

import warnings
warnings.filterwarnings('ignore')
##################################### DATA PATHS / VARIABLES #####################################

cnn_weights = 'weights/best_bottleneck_finetuned_model.hdf5'
distribution_csv_file = 'files/distribution.csv'
herbal_csv_file = 'files/herbal.csv'

class_dict_cnn = {
                0: 'Adathoda',
                1: 'Araththa',
                2: 'Aththora',
                3: 'Edaru',
                4: 'Iguru Piyali',
                5: 'Nika'
            }

pie_chart_path = 'visualization/pie_chart.png'
folium_map_path = 'visualization/folium_map.html'

######################################### CREATE / LOAD MODELS ############################################

cnn_model = tf.keras.models.load_model(cnn_weights)
automl_model = pycaret.classification.load_model('weights/Final SVM Model')

######################################### INFERENCE FUNCTIONS ############################################

def predict_automl(herbal):
    df_herbal = pd.read_csv(herbal_csv_file)
    df_herbal = df_herbal[['pnse', 'diseases', 'Treatments']]

    predictions = pycaret.classification.predict_model(automl_model, data=df_herbal)
    predictions['Label'] = predictions['pnse'].values
    del predictions ['pnse']

    Pherbal = predictions[predictions['Label'] == herbal]
    Pherbal = Pherbal[['diseases', 'Treatments']]
    diseases = Pherbal['diseases'].values
    treatments = Pherbal['Treatments'].values
    return diseases, treatments

def predict_cnn(image_path):
    image = cv.imread(image_path)
    image = cv.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = cnn_model.predict(image)
    prediction = np.argmax(prediction)
    return class_dict_cnn[prediction]

def create_distribution_visualizations():
    distribution_df = pd.read_csv(distribution_csv_file)
    colors = sns.color_palette('pastel')[0:5]
    df_dest = distribution_df[["District", "Type"]].groupby("Type").agg(['count'])['District']['count']
    plt.pie(df_dest, labels = df_dest.index, colors = colors, autopct='%.0f%%')

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10, forward=True)
    plt.title("Distribution of Plant Types")
    plt.savefig(pie_chart_path)

    df_map = distribution_df[distribution_df['Lat'].notna() & distribution_df['Lon'].notna() ]
    # concatenate District and Type
    df_text = df_map['District'] + ' - ' + df_map['Type']

    xlat = df_map['Lat'].tolist()
    xlon = df_map['Lon'].tolist()
    locations = list(zip(xlon, xlat))
    folium_map = folium.Map(location=[7.8731, 80.7718],
                            zoom_start=1,
                            tiles='CartoDB dark_matter')
    marker_cluster = MarkerCluster().add_to(folium_map)
    for point in range(0, len(locations)):
            folium.Marker(
                        locations[point], 
                        popup = folium.Popup(df_text.values[point])).add_to(marker_cluster)    
    folium_map.save(folium_map_path)

########################################### RECOMMENDATION FUNCTIONS ############################################

def data_recommendation_processing():
    posts = pd.read_csv('files/posts_data.csv')
    posts.rename(columns={'_id': 'postid', ' post_type': 'post_type'}, inplace=True)
    posts.category.fillna("General", inplace=True)

    users = pd.read_csv('files/user_data.csv')
    users.rename(columns={'id': 'user_id'}, inplace=True)

    views = pd.read_csv('files/view_data.csv')

    df = views.merge(posts, on='postid', how='left')
    df = df.merge(users, on='user_id', how='left')

    df_orginal = df.copy()

    # creating user engagement data frame
    df_user_unq_post = df.groupby(["user_id"]).agg({"postid": 'nunique'})
    df_user_unq_post.columns = ["num_diff_posts"]
    df_user_unq_post.reset_index(inplace=True)

    df_user_unq_post["num_diff_posts"].describe()

    thr_user = 0  # if you need to filterout less engaging users you can give a threshold here
    selected_user_ids = list(df_user_unq_post[df_user_unq_post["num_diff_posts"] >= thr_user]["user_id"])  # selecting user ids

    # creating post engagement data frame
    df_post_unq_user = df.groupby(["postid"]).agg({"user_id": 'nunique'})
    df_post_unq_user.columns = ["num_diff_users"]
    df_post_unq_user.reset_index(inplace=True)

    df_post_unq_user["num_diff_users"].describe()

    thr_post = 0  # if you need to filterout less engaging posts you can give a threshold here
    selected_postids = list(df_post_unq_user[df_post_unq_user["num_diff_users"] >= thr_post]["postid"])

    # filtering less engaging users and posts
    df["sel_users"] = df["user_id"].apply(lambda x: int(x in selected_user_ids))
    df["sel_posts"] = df["postid"].apply(lambda x: int(x in selected_postids))
    df = df[df["sel_users"] == 1]
    df = df[df["sel_posts"] == 1]
    df = df[["user_id", "postid"]]

    # creating the rating data frame
    df_rating = df[["user_id", "postid"]]
    df_rating["Quantity"] = 1
    df_rating = df_rating.groupby(["user_id", "postid"]).agg({'Quantity': 'sum'})
    df_rating.reset_index(inplace=True)
    df_rating = df_rating.merge(df_post_unq_user, on='postid', how='left')

    # creating the rating data frame
    df_post_pivot = df_rating.pivot(index="user_id", columns="postid", values='Quantity').fillna(0)
    df_post_corr = df_post_pivot.corr(method='spearman', min_periods=5)

    df_user_pivot = df_rating.pivot(index="postid", columns="user_id", values='Quantity').fillna(0)
    df_user_corr = df_user_pivot.corr(method='spearman', min_periods=5)

    return df_post_corr, df_user_corr, df_orginal, df_rating, posts, users

def get_top_posts(user_id, top_n, df_rating, posts):
    df_select = df_rating[df_rating["user_id"] == user_id].sort_values(['Quantity', 'num_diff_users'],
                                                                       ascending=False).head(top_n)
    df_select = posts.set_index("postid").loc[list(df_select["postid"])].reset_index()[["postid", "description", "category"]]
    return df_select

def sim_users(user_id, top_n, df_user_corr):
    df_user = df_user_corr.loc[user_id]
    df_user_sort = df_user.sort_values(ascending=False).head(top_n + 1)
    user_ids = list(df_user_sort.index)
    user_ids.remove(user_id)
    return (user_ids)

def get_col_recommendations_m1(postid, top_n, df_post_corr, df_orginal):
    df_post = df_post_corr.loc[postid]
    df_post_sort = df_post.sort_values(ascending=False).head(top_n + 1)
    df_post_sort = pd.merge(df_post_sort, df_orginal.drop_duplicates(subset=["postid"]), how='left', on="postid")

    df_ = df_post_sort[df_post_sort["postid"] != postid]
    df_["rank"] = [i + 1 for i in range(len(df_))]
    df_ = df_.set_index("rank")
    df_ = df_[["postid", "description", "category"]]
    return {postid_: description_ for postid_, description_ in zip(df_["postid"], df_["description"])}
    
def get_col_recommendations_m2(user_id, n_users, n_posts, df_user_corr, df_rating, posts):
    users_list = sim_users(user_id, n_users, df_user_corr)
    df_ = pd.DataFrame()
    for id_ in users_list:
        df_ = pd.concat([df_, get_top_posts(id_, n_posts, df_rating, posts)], ignore_index=True)
    df_ = df_.drop_duplicates(subset=['postid'])
    df_["rank"] = [i + 1 for i in range(len(df_))]
    df_ = df_.set_index("rank")
    df_ = df_[["postid", "description", "category"]]
    return {postid_: description_ for postid_, description_ in zip(df_["postid"], df_["description"])}

def run_recommendation(userid, postid):
    df_post_corr, df_user_corr, df_orginal, df_rating, posts, _ = data_recommendation_processing()
    m1_json = get_col_recommendations_m1(postid, 5, df_post_corr, df_orginal)
    m2_json = get_col_recommendations_m2(userid, 5, 3, df_user_corr, df_rating, posts)
    return m1_json, m2_json