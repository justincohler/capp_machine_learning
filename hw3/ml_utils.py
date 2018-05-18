"""
Attribution to https://github.com/rayidghani/magicloops for the following helper functions:
- plot functions
- precision_at_k
- generate_binary_at_k
- join_sort_descending
- recall_at_k 
"""
import os
import sys
import numpy as np
import pandas as pd
import random
import logging
from datetime import date, datetime, timedelta
from dateutil import parser
from dateutil.relativedelta import relativedelta
from pandas.core.common import array_equivalent
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import *
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from scipy import optimize

logger = logging.getLogger('hw3')
sh = logging.StreamHandler(sys.stdout)
logger.addHandler(sh)
logger.setLevel(logging.INFO)


class Pipeline():

    def __init__(self):
        self.le = preprocessing.LabelEncoder()
        self.classifiers = None

    def ingest(self, source, type="CSV"):
        """Return a pandas dataframe of the data from a given source string."""
        if type == "CSV":
            return pd.read_csv(source)
        else:
            raise TypeError("ingest only supports type: CSV")

    def distribution(self, data):
        """Return the distribution in the dataframe."""
        return pandas_profiling.ProfileReport(data)

    def dummify(self, data, categoricals):
        """Return an updated dataframe with binary/dummy fields from the given categorical field."""
        data = pd.get_dummies(data=data, columns=categoricals)
        return data

    def plot_precision_recall_n(self, y_true, y_prob, model_name, output_type):
        from sklearn.metrics import precision_recall_curve
        y_score = y_prob
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
        precision_curve = precision_curve[:-1]
        recall_curve = recall_curve[:-1]
        pct_above_per_thresh = []
        number_scored = len(y_score)
        for value in pr_thresholds:
            num_above_thresh = len(y_score[y_score >= value])
            pct_above_thresh = num_above_thresh / float(number_scored)
            pct_above_per_thresh.append(pct_above_thresh)
        pct_above_per_thresh = np.array(pct_above_per_thresh)

        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot(pct_above_per_thresh, precision_curve, 'b')
        ax1.set_xlabel('percent of population')
        ax1.set_ylabel('precision', color='b')
        ax2 = ax1.twinx()
        ax2.plot(pct_above_per_thresh, recall_curve, 'r')
        ax2.set_ylabel('recall', color='r')
        ax1.set_ylim([0, 1])
        ax1.set_ylim([0, 1])
        ax2.set_xlim([0, 1])

        name = model_name
        plt.title(name)
        if (output_type == 'save'):
            plt.savefig(name, close=True)
        elif (output_type == 'show'):
            plt.show()
        else:
            plt.show()

    def plot_roc(self, name, probs, true, output_type):
        fpr, tpr, thresholds = roc_curve(true, probs)
        roc_auc = auc(fpr, tpr)
        pl.clf()
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.05])
        pl.ylim([0.0, 1.05])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title(name)
        pl.legend(loc="lower right")
        if (output_type == 'save'):
            plt.savefig(name, close=True)
        elif (output_type == 'show'):
            plt.show()
        else:
            plt.show()

    def plot_precision_recall_n(self, y_true, y_prob, model_name, output_type):
        y_score = y_prob
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
        precision_curve = precision_curve[:-1]
        recall_curve = recall_curve[:-1]
        pct_above_per_thresh = []
        number_scored = len(y_score)
        for value in pr_thresholds:
            num_above_thresh = len(y_score[y_score >= value])
            pct_above_thresh = num_above_thresh / float(number_scored)
            pct_above_per_thresh.append(pct_above_thresh)
        pct_above_per_thresh = np.array(pct_above_per_thresh)

        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot(pct_above_per_thresh, precision_curve, 'b')
        ax1.set_xlabel('percent of population')
        ax1.set_ylabel('precision', color='b')
        ax2 = ax1.twinx()
        ax2.plot(pct_above_per_thresh, recall_curve, 'r')
        ax2.set_ylabel('recall', color='r')
        ax1.set_ylim([0, 1])
        ax1.set_ylim([0, 1])
        ax2.set_xlim([0, 1])

        name = model_name
        plt.title(name)
        if (output_type == 'save'):
            plt.savefig(name, close=True)
        elif (output_type == 'show'):
            plt.show()
        else:
            plt.show()

    def presplit(self, df):
        logger.info("Generating features...")
        df = df.fillna(df.median())

        df = self.dummify(df, ['grade_level', 'secondary_focus_subject', 'secondary_focus_area', 'primary_focus_area',
                               'primary_focus_subject', 'teacher_prefix', 'school_metro', 'poverty_level', 'resource_type'])
        logger.info("Generated dummy features.")
        df = self.labelize(df, ['schoolid', 'school_district', 'school_county',
                                'teacher_acctid', 'school_state', 'school_city'])
        logger.info("Generated labelized features.")
        df = self.replace_tfs(df, ['school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp',
                                   'school_charter_ready_promise', 'teacher_ny_teaching_fellow', 'teacher_teach_for_america',
                                   'eligible_double_your_impact_match', 'eligible_almost_home_match'])
        logger.info("Made binaries from 't' 'f' pair columns.")

        logger.info("Finished feature generation.")
        return df

    def preprocess(self, df):
        y = self.replace_tfs(df, ['fully_funded'])['fully_funded']
        X = df.drop(['projectid', 'fully_funded', 'date_posted'], axis=1)
        return X, y

    def labelize(self, df, fields):
        for field in fields:
            series = df[field]
            df[field] = self.le.fit_transform(series.tolist())

        return df

    def replace_tfs(self, df, fields):
        for field in fields:
            arr = df[field]
            df[field] = np.where(arr == 't', 1, 0)
        return df

    def discretize(self, data, field, bins=None, labels=None):
        """Return a discretized Series of the given field."""
        if not bins and not labels:
            series = pd.qcut(data[field], q=4)
        elif not labels and bins != None:
            series = pd.qcut(data[field], q=bins)
        elif not bins and labels != None:
            series = pd.qcut(data[field], q=len(labels), labels=labels)
        elif bins != len(labels):
            raise IndexError("Bin size and label length must be equal.")
        else:
            series = pd.qcut(data[field], q=bins, labels=labels)

        return series

    def generate_outcome_table(self, results):
        return pd.DataFrame.from_records(results, columns=('model_type', 'clf', 'parameters', 'auc-roc', 'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50', 'r_at_1', 'r_at_2', 'r_at_5', 'r_at_10', 'r_at_20', 'r_at_30', 'r_at_50'))

    def populate_outcome_table(self, model_key, classifier, params, y_test, y_pred_probs):
        y_pred_probs_sorted, y_test_sorted = zip(
            *sorted(zip(y_pred_probs, y_test), reverse=True))

        return (model_key, classifier, params,
                roc_auc_score(y_test, y_pred_probs),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 1.0),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 2.0),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 5.0),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 10.0),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 20.0),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 30.0),
                self.precision_at_k(
                    y_test_sorted, y_pred_probs_sorted, 50.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 1.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 2.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 5.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 10.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 20.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 30.0),
                self.recall_at_k(
                    y_test_sorted, y_pred_probs_sorted, 50.0),
                )

    def joint_sort_descending(self, l1, l2):
        # l1 and l2 have to be numpy arrays
        idx = np.argsort(l1)[:: -1]
        return l1[idx], l2[idx]

    def generate_binary_at_k(self, y_scores, k):
        cutoff_index = int(len(y_scores) * (k / 100.0))
        test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
        return test_predictions_binary

    def precision_at_k(self, y_true, y_scores, k):
        y_scores, y_true = self.joint_sort_descending(np.array(y_scores), np.array(y_true))
        preds_at_k = self.generate_binary_at_k(y_scores, k)
        precision = precision_score(y_true, preds_at_k)
        return precision

    def recall_at_k(self, y_true, y_scores, k):
        y_scores_sorted, y_true_sorted = self.joint_sort_descending(
            np.array(y_scores), np.array(y_true))
        preds_at_k = self.generate_binary_at_k(y_scores_sorted, k)
        recall = recall_score(y_true_sorted, preds_at_k)
        return recall

    def generate_classifiers(self):

        self.classifiers = {'RF': {
            "type": RandomForestClassifier(),
            "params": {'n_estimators': [10], 'max_depth': [5, 50], 'max_features': ['sqrt'], 'min_samples_split': [10]}
        },
            'LR': {
            "type": LogisticRegression(),
            "params": {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1]}
        },
            'NB': {
            "type": GaussianNB(),
            "params": {}
        },
            'SVM': {
            "type": svm.SVC(probability=True, random_state=0),
            "params": {'C': [1, 10], 'kernel': ['linear']}
        },
            'GB': {
            "type": GradientBoostingClassifier(),
            "params": {'n_estimators': [5, 10], 'learning_rate': [0.5], 'subsample': [0.5], 'max_depth': [1, 5]}
        },
            'BAG': {
            "type": BaggingClassifier(),
            "params": {'n_estimators': [10], 'max_samples': [5], 'max_features': [5, 20], 'bootstrap_features': [False, True]}
        },
            'DT': {
            "type": DecisionTreeClassifier(),
            "params": {'criterion': ['gini', 'entropy'], 'max_depth': [5, 50], 'min_samples_split': [2, 10]}
        },
            'KNN': {
            "type": KNeighborsClassifier(),
            "params": {'n_neighbors': [10, 20], 'weights': ['uniform', 'distance'], 'algorithm': ['ball_tree', 'kd_tree']}
        }
        }

        return

    def run_temporal(self, models_to_run, projects, outcomes, start, end, baselines_to_run=None):

        prediction_window = update_window = 2

        results = []

        test_end = end
        while (test_end >= start + 2 * relativedelta(months=+prediction_window)):
            test_start = test_end - relativedelta(months=+prediction_window)
            train_end = test_start - relativedelta(days=+1)  # minus 1 day
            train_start = train_end - relativedelta(months=+prediction_window)
            while (train_start >= start):
                logger.info("Temporally validating on:\nTrain: {} - {}\nTest: {} - {}\nPrediction window: {} months".format(train_start, train_end,
                                                                                                                            test_start, test_end, prediction_window))
                train_start -= relativedelta(months=+prediction_window)

                # projects = projects.drop(['date_posted'], axis=1)
                projects.to_csv("agh.csv")
                projects_train = projects[projects['date_posted'].between(
                    train_start, train_end, inclusive=True)]
                projects_test = projects[projects['date_posted'].between(
                    test_start, test_end, inclusive=True)]
                projects_train.set_index(["projectid"])
                projects_test.set_index(["projectid"])

                logger.info("Merging dataframes...")
                df_train = outcomes.merge(projects_train, on="projectid")
                df_test = outcomes.merge(projects_test, on="projectid")
                logger.info("DataFrames merged with dimensions: train{}, test{}.".format(
                    df_train.shape, df_test.shape))

                logger.info("Splitting X and y, train and test...")
                X_train, y_train = self.preprocess(df_train)
                X_test, y_test = self.preprocess(df_test)

                results.extend(self.classify(
                    models_to_run, X_train, X_test, y_train, y_test, baselines_to_run))
            test_end -= relativedelta(months=+update_window)

        print(results)
        results_df = self.generate_outcome_table(results)
        return results_df

    def classify(self, models_to_run, X_train, X_test, y_train, y_test, baselines_to_run=None):

        self.generate_classifiers()
        results = []
        for model_key in models_to_run:
            count = 1
            logger.info("Running {}...".format(model_key))
            classifier = self. classifiers[model_key]["type"]
            grid = ParameterGrid(self.classifiers[model_key]["params"])
            for params in grid:
                logger.info("Running with params {}".format(params))
                try:
                    classifier.set_params(**params)
                    fit = classifier.fit(X_train, y_train)
                    y_pred_probs = fit.predict_proba(X_test)[:, 1]
                    results.append(self.populate_outcome_table(
                        model_key, classifier, params, y_test, y_pred_probs))

                    self.plot_precision_recall_n(
                        y_test, y_pred_probs, model_key+str(count), 'save')
                    count = count + 1

                except IndexError as e:
                    print('Error:', e)
                    continue
            logger.info("{} finished.".format(model_key))

        if baselines_to_run != None:
            for baseline in baselines_to_run:
                if baseline == "RAND":
                    pct_negative = len(y_train[y_train == 0])/len(y_train)
                    y_pred_probs = np.random.rand(len(y_test))
                    y_pred_probs = [1 if row > pct_negative else 0 for row in y_pred_probs]
                    results.append(self.populate_outcome_table(
                        baseline, baseline, {}, y_test, y_pred_probs))
        return results

"""
if __name__ == "__main__":
    pd.options.display.max_rows = 999

    pipeline = Pipeline()
    start = parser.parse("2013-07-31")
    end = parser.parse("2013-12-31")

    logger.info("Ingesting dataframes...")
    outcomes = pipeline.ingest('data/outcomes.csv')
    projects = pipeline.ingest('data/projects.csv')
    outcomes.set_index("projectid")
    outcomes = outcomes.drop(['is_exciting', 'at_least_1_teacher_referred_donor',  'at_least_1_green_donation', 'great_chat', 'three_or_more_non_teacher_referred_donors',
                              'one_non_teacher_referred_donor_giving_100_plus', 'donation_from_thoughtful_donor', 'great_messages_proportion', 'teacher_referred_count', 'non_teacher_referred_count'], axis=1)

    projects["date_posted"] = pd.to_datetime(projects["date_posted"])
    projects.set_index("date_posted")
    projects = projects[projects['date_posted'].between(
        start, end, inclusive=True)]

    projects = pipeline.presplit(projects)
    logger.info("Project dimensions after feature generation: {}".format(projects.shape))

    # 'LR', 'KNN', 'DT', 'RF', 'GB', 'NB'
    results_df = pipeline.run_temporal(['LR', 'KNN'], projects, outcomes, start, end)
    logger.info(results_df)
"""
