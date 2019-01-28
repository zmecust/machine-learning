# https://blog.csdn.net/a819825294/article/details/51775418
# coding=UTF-8

import pandas as pd
import xgboost as xgb
import numpy as np
import utils as util
from hyperopt import fmin, hp, tpe
import hyperopt
from time import clock
from utils import *

model_name = 'xgb'


def xgb_train(dtrain, dtest, param, offline=True, verbose=True, num_boost_round=1000):
    if verbose:
        if offline:
            watchlist = [(dtrain, 'train'), (dtest, 'test')]
        else:
            watchlist = [(dtrain, 'train')]
        model = xgb.train(
            param, dtrain, num_boost_round=num_boost_round, evals=watchlist)
        feature_score = model.get_fscore()
        feature_score = sorted(feature_score.items(),
                               key=lambda x: x[1], reverse=True)
        fs = []
        for key, value in feature_score:
            fs.append("{0},{1}\n".format(key, value))
        if offline:
            feature_score_file = './feature_score/offline_feature_score' + '.csv'
        else:
            feature_score_file = './feature_score/online_feature_score' + '.csv'
        f = open(feature_score_file, 'w')
        f.writelines("feature,score\n")
        f.writelines(fs)
        f.close()
    else:
        model = xgb.train(param, dtrain, num_boost_round=num_boost_round)
    return model


def xgb_predict(model, dtest):
    print 'model_best_ntree_limit : {0}\n'.format(model.best_ntree_limit)
    pred_y = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    return pred_y


def tune_xgb(dtrain, dtest):
    tune_reuslt_file = "./log/tune_" + model_name + ".csv"
    f_w = open(tune_reuslt_file, 'w')

    def objective(args):
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': weight,
            # 'lambda': 1000,
            'nthread': n_jobs,
            'eta': args['learning_rate'],
            # 'gamma': args['gamma'],
            'colsample_bytree': args['colsample_bytree'],
            'max_depth': args['max_depth'],
            'subsample': args['subsample']
        }

        # if fs verbose = False
        model = xgb_train(dtrain, dtest, params, offline=True,
                          verbose=False, num_boost_round=int(args['n_estimators']))

        # model.save_model('xgb.model')
        model.dump_model('dump_model_txt')

        pred_y = xgb_predict(model, dtest)
        pred_y[pred_y > 0.5] = 1
        pred_y[pred_y <= 0.5] = 0
        test_y = dtest.get_label()
        F1 = evalF1(test_y, pred_y)

        xgb_log.write(str(args))
        xgb_log.write('\n')
        xgb_log.write(str(F1))
        xgb_log.write('\n')
        return F1*(-1.0)

    # Searching space
    space = {
        'n_estimators': hp.quniform("n_estimators", 100, 200, 20),
        # 'reg_lambda': hp.loguniform("reg_lambda", np.log(1), np.log(1500)),
        # 'gamma': hp.loguniform("gamma", np.log(0.1), np.log(100)),
        'learning_rate': hp.uniform("learning_rate", 0.05, 0.15),
        'max_depth': 8,
        'subsample': hp.uniform("subsample", 0.5, 0.9),
        'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 0.9),
    }
    best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=150)
    #best_sln = fmin(objective, space, algo=hyperopt.anneal.suggest, max_evals=300)
    pickle.dump(best_sln, f_w, True)
    best_F1 = objective(best_sln)
    xgb_log.write(str(best_F1) + '\n')
    f_w.close()


def test(dtrain, dtest, best_n_estimators):
    final_result = "./log/xgb_online_result.csv"
    f_w = open(final_result, 'w')
    model = xgb_train(dtrain, dtest, init_params, offline,
                      verbose=True, num_boost_round=best_n_estimators)
    pred_y = xgb_predict(model, dtest)
    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y <= 0.5] = 0
    test_y = dtest.get_label()
    F1 = evalF1(test_y, pred_y)
    f_w.write(str(F1))
    f_w.close()


if __name__ == '__main__':
    t_start = clock()
    offline = False
    train_x, train_y, test_x, test_y, TF_id, TG_id = load_data(offline)

    # feature selection
    # fs = list(pd.read_csv('./feature_score/offline_feature_score.csv')['feature'])
    # train_x = train_x[fs]
    # test_x = test_x[fs]

    weight = float(len(train_y[train_y == 0]))/len(train_y[train_y == 1])
    class_weight = {1: weight, 0: 1}

    print 'Feature Dims : '
    print train_x.shape
    print test_x.shape

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    del train_x, train_y, test_x, test_y
    gc.collect()

    if offline:
        xgb_log = open(name='./log/xgb_log.txt', mode='w')
        tune_xgb(dtrain, dtest)
        xgb_log.close()
    else:
        tune_reuslt_file = "./log/tune_" + model_name + ".csv"
        f_w = open(tune_reuslt_file, 'r')
        tune_xgb = pickle.load(f_w)
        f_w.close()

        best_n_estimators = int(tune_xgb['n_estimators'])
        best_learning_rate = tune_xgb['learning_rate']
        # best_max_depth = int(tune_xgb['max_depth'])
        best_subsample = tune_xgb['subsample']
        best_colsample_bytree = tune_xgb['colsample_bytree']

        init_params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'scale_pos_weight': weight,
            'max_depth': 8,
            'subsample': best_subsample,
            'nthread': n_jobs,
            'eval_metric': 'auc',
            'colsample_bytree': best_colsample_bytree,
            'eta': best_learning_rate
        }
        test(dtrain, dtest, best_n_estimators)

    t_finish = clock()
    print('==============Costs time : %s s==============' %
          str(t_finish - t_start))
