# https://www.toutiao.com/i6648540624171565572/?tt_from=copy_link&utm_campaign=client_share&timestamp=1547987885&app=news_article&utm_source=copy_link&iid=57493149167&utm_medium=toutiao_ios&group_id=6648540624171565572
# 使用贝叶斯优化为机器学习模型找到最佳超参数

import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from bayes_opt import BayesianOptimization

h2o.init()
h2o.remove_all()

data = h2o.upload_file("../datasets/learning/winequality-red.csv")
train_cols = [x for x in data.col_names if x not in ['quality']]
target = "quality"
train, test = data.split_frame(ratios=[0.7])


def train_model(max_depth,
                ntrees,
                min_rows,
                learn_rate,
                sample_rate,
                col_sample_rate):

    params = {
        'max_depth': int(max_depth),
        'ntrees': int(ntrees),
        'min_rows': int(min_rows),
        'learn_rate': learn_rate,
        'sample_rate': sample_rate,
        'col_sample_rate': col_sample_rate
    }
    model = H2OGradientBoostingEstimator(nfolds=5, **params)
    model.train(x=train_cols, y=target, training_frame=train)
    return -model.rmse()


def run():
    bounds = {
        'max_depth': (5, 10),
        'ntrees': (100, 500),
        'min_rows': (10, 30),
        'learn_rate': (0.001, 0.01),
        'sample_rate': (0.5, 0.8),
        'col_sample_rate': (0.5, 0.8)
    }

    optimizer = BayesianOptimization(
        f=train_model,
        pbounds=bounds,
        random_state=1,
    )

    optimizer.maximize(init_points=10, n_iter=50)
    optimizer.max


if __name__ == '__main__':
    run()
