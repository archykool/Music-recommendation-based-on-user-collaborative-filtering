import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import math as mt
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix

# 读取数据，并为数据添加表头
triplet_dataset = pd.read_csv (filepath_or_buffer='./data/train_triplets.txt',
                               nrows=10000, sep='\t', header=None,
                               names=['user', 'song', 'play_count'])
#######################
##### 数据预处理 ######
#######################

# 缩减数据规模
# 统计用户播放情况user_playcount_df.csv
output_dict = {}
with open ('./data/train_triplets.txt') as f:
    for line_number, line in enumerate (f):
        user = line.split ('\t')[0]
        play_count = int (line.split ('\t')[2])
        if user in output_dict:
            play_count += output_dict[user]
            output_dict.update ({user: play_count})
        output_dict.update ({user: play_count})
output_list = [{'user': k, 'play_count': v} for k, v in output_dict.items ()]
play_count_df = pd.DataFrame (output_list)
play_count_df = play_count_df.sort_values (by='play_count', ascending=False)
play_count_df.to_csv (path_or_buf='./data/user_playcount_df.csv', index=False)
# 统计歌曲播放情况,存为song_playcount_df.csv
output_dict = {}
with open ('./data/train_triplets.txt') as f:
    for line_number, line in enumerate (f):
        song = line.split ('\t')[1]
        play_count = int (line.split ('\t')[2])
        if song in output_dict:
            play_count += output_dict[song]
            output_dict.update ({song: play_count})
        output_dict.update ({song: play_count})
output_list = [{'song': k, 'play_count': v} for k, v in output_dict.items ()]
song_count_df = pd.DataFrame (output_list)
song_count_df = song_count_df.sort_values (by='play_count', ascending=False)
song_count_df.to_csv (path_or_buf='./data/song_playcount_df.csv', index=False)
# 进行目标用户抽取、过滤非目标歌曲triplet_dataset_sub_song.csv
total_play_count = sum (song_count_df.play_count)
play_count_subset = play_count_df.head (n=100000)
song_count_subset = song_count_df.head (n=30000)
user_subset = list (play_count_subset.user)
song_subset = list (song_count_subset.song)
triplet_dataset = pd.read_csv (filepath_or_buffer='./data/train_triplets.txt', sep='\t',
                               header=None, names=['user', 'song', 'play_count'])
triplet_dataset_sub = triplet_dataset[triplet_dataset.user.isin (user_subset)]
del (triplet_dataset)
triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub.song.isin (song_subset)]
del (triplet_dataset_sub)
triplet_dataset_sub_song.to_csv ('./data/triplet_dataset_sub_song.csv', index=False)
# 数据集2,用于过滤triplet_dataset_sub_song_merged.csv
conn = sqlite3.connect ('./data/track_metadata.db')
cur = conn.cursor ()
cur.execute ("SELECT name FROM sqlite_master WHERE type='table'")
cur.fetchall ()
track_metadata_df = pd.read_sql (con=conn, sql='select * from songs')
track_metadata_df_sub = track_metadata_df[track_metadata_df.song_id.isin (song_subset)]
# 合并数据
del (track_metadata_df_sub['track_id'])
del (track_metadata_df_sub['artist_mbid'])
track_metadata_df_sub = track_metadata_df_sub.drop_duplicates (['song_id'])
triplet_dataset_sub_song_merged = pd.merge (triplet_dataset_sub_song, track_metadata_df_sub, how='left', left_on='song',
                                            right_on='song_id')
triplet_dataset_sub_song_merged.rename (columns={'play_count': 'listen_count'}, inplace=True)
# 删除无用字段
del (triplet_dataset_sub_song_merged['song_id'])
del (triplet_dataset_sub_song_merged['artist_id'])
del (triplet_dataset_sub_song_merged['duration'])
del (triplet_dataset_sub_song_merged['artist_familiarity'])
del (triplet_dataset_sub_song_merged['artist_hotttnesss'])
del (triplet_dataset_sub_song_merged['track_7digitalid'])
del (triplet_dataset_sub_song_merged['shs_perf'])
del (triplet_dataset_sub_song_merged['shs_work'])
# 保存数据
triplet_dataset_sub_song_merged.to_csv ('./data/triplet_dataset_sub_song_merged.csv', encoding='utf-8', index=False)

#################################
##### 基于歌曲相似度的模型 ######
#################################

# 进一步缩减数据库
song_count_subset = song_count_df.head (n=5000)  # 选择最流行的5000首歌
user_subset = list (play_count_subset.user)
song_subset = list (song_count_subset.song)
triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[
    triplet_dataset_sub_song_merged.song.isin (song_subset)]


# 构建模型
class item_similarity_recommender_py ():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None

    # 获取用户听歌列表、获取歌曲听众列表
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list (user_data[self.item_id].unique ())
        return user_items

    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set (item_data[self.user_id].unique ())
        return item_users

    # 获取目标数据
    def get_all_items_train_data(self):
        all_items = list (self.train_data[self.item_id].unique ())
        return all_items

    # 构造歌曲相似度矩阵
    def construct_cooccurence_matrix(self, user_songs, all_songs):
        user_songs_users = []
        for i in range (0, len (user_songs)):
            user_songs_users.append (self.get_item_users (user_songs[i]))
        cooccurence_matrix = np.matrix (np.zeros (shape=(len (user_songs), len (all_songs))), float)
        for i in range (0, len (all_songs)):
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set (songs_i_data[self.user_id].unique ())
            for j in range (0, len (user_songs)):
                users_j = user_songs_users[j]
                # 计算相似度
                users_intersection = users_i.intersection (users_j)
                if len (users_intersection) != 0:
                    users_union = users_i.union (users_j)
                    cooccurence_matrix[j, i] = float (len (users_intersection)) / float (len (users_union))
                else:
                    cooccurence_matrix[j, i] = 0
        return cooccurence_matrix

    # 使用矩阵进行建模
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print ("Non zero values in cooccurence_matrix :%d" % np.count_nonzero (cooccurence_matrix))
        user_sim_scores = cooccurence_matrix.sum (axis=0) / float (cooccurence_matrix.shape[0])
        user_sim_scores = np.array (user_sim_scores)[0].tolist ()
        sort_index = sorted (((e, i) for i, e in enumerate (list (user_sim_scores))), reverse=True)
        columns = ['user_id', 'song', 'score', 'rank']
        df = pd.DataFrame (columns=columns)
        rank = 1
        for i in range (0, len (sort_index)):
            if ~np.isnan (sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len (df)] = [user, all_songs[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1
        if df.shape[0] == 0:
            print ("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    # 根据模型进行推荐
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    def recommend(self, user):
        user_songs = self.get_user_items (user)
        print ("No. of unique songs for the user: %d" % len (user_songs))
        all_songs = self.get_all_items_train_data ()
        print ("no. of unique songs in the training set: %d" % len (all_songs))
        cooccurence_matrix = self.construct_cooccurence_matrix (user_songs, all_songs)
        df_recommendations = self.generate_top_recommendations (user, cooccurence_matrix, all_songs, user_songs)
        return df_recommendations

    def get_similar_items(self, item_list):
        user_songs = item_list
        all_songs = self.get_all_items_train_data ()
        print ("no. of unique songs in the training set: %d" % len (all_songs))
        cooccurence_matrix = self.construct_cooccurence_matrix (user_songs, all_songs)
        user = ""
        df_recommendations = self.generate_top_recommendations (user, cooccurence_matrix, all_songs, user_songs)
        return df_recommendations


# 开始训练
train_data, test_data = train_test_split (triplet_dataset_sub_song_merged_sub, test_size=0.30, random_state=0)
is_model = item_similarity_recommender_py ()
is_model.create (train_data, 'user', 'title')
# 指定用户进行推荐
user_id = list (train_data.user)[7]
user_items = is_model.get_user_items (user_id)
is_model.recommend (user_id)

#################################
##### 基于音乐-元素的模型 #######
#################################

triplet_dataset_sub_song_merged = pd.read_csv ('./data/triplet_dataset_sub_song_merged.csv', encoding='utf-8')
# 使用用户播百分比作为评分
triplet_dataset_sub_song_merged_sum_df = triplet_dataset_sub_song_merged[['user', 'listen_count']].groupby (
    'user').sum ().reset_index ()
triplet_dataset_sub_song_merged_sum_df.rename (columns={'listen_count': 'total_listen_count'}, inplace=True)
triplet_dataset_sub_song_merged = pd.merge (triplet_dataset_sub_song_merged, triplet_dataset_sub_song_merged_sum_df)
triplet_dataset_sub_song_merged['fractional_play_count'] = triplet_dataset_sub_song_merged['listen_count'] / \
                                                           triplet_dataset_sub_song_merged['total_listen_count']
# 准备好 用户-歌曲“评分”矩阵
small_set = triplet_dataset_sub_song_merged
user_codes = small_set.user.drop_duplicates ().reset_index ()
song_codes = small_set.song.drop_duplicates ().reset_index ()
user_codes.rename (columns={'index': 'user_index'}, inplace=True)
song_codes.rename (columns={'index': 'song_index'}, inplace=True)
song_codes['so_index_value'] = list (song_codes.index)
user_codes['us_index_value'] = list (user_codes.index)
small_set = pd.merge (small_set, song_codes, how='left')
small_set = pd.merge (small_set, user_codes, how='left')
mat_candidate = small_set[['us_index_value', 'so_index_value', 'fractional_play_count']]
data_array = mat_candidate.fractional_play_count.values
row_array = mat_candidate.us_index_value.values
col_array = mat_candidate.so_index_value.values
data_sparse = coo_matrix ((data_array, (row_array, col_array)), dtype=float)


# 计算SVD矩阵分解
def compute_svd(urm, K):
    U, s, Vt = svds (urm, K)
    dim = (len (s), len (s))
    S = np.zeros (dim, dtype=np.float32)
    for i in range (0, len (s)):
        S[i, i] = mt.sqrt (s[i])  # 求平方根
    U = csc_matrix (U, dtype=np.float32)
    S = csc_matrix (S, dtype=np.float32)
    Vt = csc_matrix (Vt, dtype=np.float32)
    return U, S, Vt


def compute_estimated_matrix(urm, U, S, Vt, uTest, K):
    rightTerm = S * Vt
    max_recommendation = 250
    estimatedRatings = np.zeros (shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros (shape=(MAX_UID, max_recommendation), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :] * rightTerm
        estimatedRatings[userTest, :] = prod.todense ()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort ()[:max_recommendation]
    return recomendRatings


K = 50
urm = data_sparse
MAX_PID = urm.shape[1]
MAX_UID = urm.shape[0]
U, S, Vt = compute_svd (urm, K)
uTest = [4, 5, 6, 7, 8, 873, 23]
uTest_recommended_items = compute_estimated_matrix (urm, U, S, Vt, uTest, K)
# 进行用户推荐
for user in uTest:
    print
    u"Recommendation for user with user id {}".format (user)
    rank_value = 1
    for i in uTest_recommended_items[user, 0:10]:
        song_details = small_set[small_set.so_index_value == i].drop_duplicates ('so_index_value')[
            ['title', 'artist_name']]
        print
        u"The number {} recommended song is {} BY {}".format (rank_value, list (song_details['title'])[0],
                                                              list (song_details['artist_name'])[0])
        rank_value += 1
