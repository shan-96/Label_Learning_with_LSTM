import re

import numpy as np
import pandas as pd
import torch
from displayfunction import display
from metal import MajorityLabelVoter, EndModel
from metal.analysis import lf_summary, label_coverage
from metal.label_model import LabelModel
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from snorkel.labeling import labeling_function, PandasLFApplier

from scrap.GlobalVars import TRAIN_CSV_FILE, TEST_CSV_FILE, LF_CSV_FILE


class CommentSnorkel:
    # voting scores
    POSITIVE = 1
    NEGATIVE = -1
    ABSTAIN = 0

    # ensure csv data only has fixed and not fixed
    train = pd.read_csv(TRAIN_CSV_FILE)
    test = pd.read_csv(TEST_CSV_FILE)
    LF_set = pd.read_csv(LF_CSV_FILE)

    # labels/JIRA resolutions
    labels = {
        'Fixed': 0,
        'As Designed': 1,
        'Training': 2,
        'Not an Issue': 3,
        'Request Completed': 4,
        'Deferred': 5,
        'Duplicate': 6,
        'Cannot Reproduce': 7,
        '3rd Party Request Completed': 8,
        'Lack of Response': 9,
        'ASP Request Completed': 10,
        'Implementation': 11,
        'Design Gap': 12,
        'Fixed in a Later Version': 13,
        'Unresolved': 14,
        'Feature Request': 15,
        'Dropped': 16,
        'Done': 17,
        'Customer Accepted': 18,
        'Working as Expected': 19,
        'Functional Gap': 20,
        'Enhancement Required': 21,
        'Documentation Provided': 22,
        'Enhancement implemented': 23,
        'Training Provided': 24,
        'Won\'t Do': 25,
        'On hold': 26
    }

    # we will only try to label our ticket as fixed out of the above labels on our data
    # which means we need to take into consideration only data points labelled as 'fixed'

    # LF1 - try to mark the jira fixed
    @labeling_function()
    def lf_mark_fixed(self, comments):
        KEYS = r"\(mark|fix|done|ok|verify|verified|close|resolve)"
        return self.POSITIVE if re.search(KEYS, comments) else self.ABSTAIN

    # LF2 - try to check if it is not fixed
    @labeling_function()
    def lf_check_fixed(self, comments):
        KEYS = r"\(pending|incorrect|wrong|waiting|open|update|move)"
        return self.NEGATIVE if re.search(KEYS, comments) else self.ABSTAIN

    # LF3 - try to find false negatives
    @labeling_function()
    def lf_find_false_negatives(self, comments):
        KEYS = r"\b(question|why|not issue|not wrong)"
        return self.POSITIVE if re.search(KEYS, comments) else self.ABSTAIN

    # LF4 - try to find false positives
    @labeling_function()
    def lf_find_false_positives(self, comments):
        KEYS = r"\b(not fix|not done|not correct)"
        return self.NEGATIVE if re.search(KEYS, comments) else self.ABSTAIN

    # The list of my label functions
    LFs = [lf_mark_fixed, lf_check_fixed, lf_find_false_negatives, lf_find_false_positives]
    LF_names = {1: 'mark_fixed', 2: 'check_fixed', 3: 'find_false_negatives', 4: 'find_false_positives'}

    # on a fairly good amount of train data 2 LFs might just work
    # we can now start making the Ls matrix from the LFs by applying every LF on each of the data point
    def make_Ls_matrix(self, data, LFs):
        noisy_labels = np.empty((len(data), len(LFs)))
        for i, row in data.iterrows():
            for j, lf in enumerate(LFs):
                noisy_labels[i][j] = lf(row)
        return noisy_labels

    def get_label(self, text):
        return self.labels.get(text) if self.labels.get(text) is not None else -1

    def getTrainedModel1(self):

        # We build a matrix of LF votes for each comment ticket
        LF_matrix = self.make_Ls_matrix(self.LF_set['comments'], self.LFs)

        # Get true labels for LF set
        Y_LF_set = np.array(self.LF_set['resolution'])

        display(lf_summary(sparse.csr_matrix(LF_matrix),
                           Y=Y_LF_set,
                           lf_names=self.LF_names.values()))

        print("label coverage: " + label_coverage(LF_matrix))

        mv = MajorityLabelVoter()
        Y_train_majority_votes = mv.predict(LF_matrix)
        print("classification report:\n" + classification_report(Y_LF_set, Y_train_majority_votes))

        Ls_train = self.make_Ls_matrix(self.train, self.LFs)

        # You can tune the learning rate and class balance.
        model = LabelModel(k=2, seed=123)
        trainer = model.train_model(Ls_train, n_epochs=2000, print_every=1000,
                                    lr=0.0001,
                                    class_balance=np.array([0.2, 0.8]))

        Y_train = model.predict(Ls_train) + Y_LF_set

        print('Trained Label Model Metrics:')
        scores = model.score((Ls_train[1], Y_train[1]), metric=['accuracy', 'precision', 'recall', 'f1'])
        print(scores)

        return trainer, Y_train

    # another method
    def getTrainedModel2(self):
        # Apply the LFs to the unlabeled training data
        applier = PandasLFApplier(self.LFs)
        L_train = applier.apply(self.train['comments'])

        # Train the label model and compute the training labels
        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
        self.train['resolution'] = label_model.predict(L=L_train, tie_break_policy="abstain")
        df_train = self.train[self.train.resolution != self.ABSTAIN]

        train_text = df_train.comments.tolist()
        X_train = CountVectorizer(ngram_range=(1, 2)).fit_transform(train_text)

        clf = LogisticRegression(solver="lbfgs")
        clf.fit(X=X_train, y=df_train.resolution.values)
        prob = clf.predict_proba(self.test)

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        end_model = EndModel([1000, 10, 2], seed=123, device=device)

        end_model.train_model((self.train['comments'], self.test['comments']),
                              valid_data=(self.train['resolution'], self.test['comments']), lr=0.01, l2=0.01,
                              batch_size=256,
                              n_epochs=5, checkpoint_metric='accuracy', checkpoint_metric_mode='max')

        return prob
