

class ProjectConfig:
    def __init__(self, project_path):
        self.project_path = project_path
        self.m1m_data_path = f"{project_path}/data/movielens"
        self.m1m_train_data = f"{self.m1m_data_path}/movielens_df_train.csv"
        self.m1m_test_data = f"{self.m1m_data_path}/movielens_df_test.csv"
        self.u_i_cols = ['user', 'movie_id']
        self.u_cat_cols = ['gender', 'age', 'occupation']
        self.i_cat_cols = ['movie_genre_1']
        self.output_col = 'label'


class ModelConfig:
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 2048
