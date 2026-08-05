import os

import numpy as np
from termcolor import cprint

from SubFunctions.Evaluate import Evaluation_Metrics, Evaluation_Metrics1
from SubFunctions.Model import Network
# from SubFunctions.Evaluate import Evaluation_Metrics
from sklearn.model_selection import KFold


def train_test_split(data, train_size):

    labels = data['labels']
    # Get the unique classes in the target variable 'y'
    num_classes = np.unique(labels)

    # Initialize empty lists to store training and testing data
    x_train1 = []
    x_train2 = []
    x_train3 = []
    x_train4 = []
    x_train5 = []
    x_train6 = []
    y_train = []

    x_test1 = []
    x_test2 = []
    x_test3 = []
    x_test4 = []
    x_test5 = []
    x_test6 = []
    y_test = []

    # Loop through each unique class
    for i in range(len(num_classes)):
        # Find indices of samples belonging to the current class
        indices = np.where(labels == num_classes[i])

        # Split the indices based on the specified 'train_size'
        train_index = indices[0][:int(len(indices[0]) * train_size)]
        test_index = indices[0][int(len(indices[0]) * train_size):]

        # Extract features and labels for training set
        train_feat1 = data['comparative1'][train_index]
        train_feat2 = data['comparative2'][train_index]
        train_feat3 = data['comparative3'][train_index]
        train_feat4 = data['comparative4'][train_index]
        train_feat5 = data['comparative5'][train_index]
        train_feat6 = data['proposed'][train_index]
        train_lab = labels[train_index]

        # Extract features and labels for testing set
        test_feat1 = data['comparative1'][test_index]
        test_feat2 = data['comparative2'][test_index]
        test_feat3 = data['comparative3'][test_index]
        test_feat4 = data['comparative4'][test_index]
        test_feat5 = data['comparative5'][test_index]
        test_feat6 = data['proposed'][test_index]
        test_lab = labels[test_index]

        # Extend the lists with the current class data
        x_train1.extend(train_feat1)
        x_train2.extend(train_feat2)
        x_train3.extend(train_feat3)
        x_train4.extend(train_feat4)
        x_train5.extend(train_feat5)
        x_train6.extend(train_feat6)

        y_train.extend(train_lab)

        x_test1.extend(test_feat1)
        x_test2.extend(test_feat2)
        x_test3.extend(test_feat3)
        x_test4.extend(test_feat4)
        x_test5.extend(test_feat5)
        x_test6.extend(test_feat6)

        y_test.extend(test_lab)

    # Convert the lists to numpy arrays
    x_train1 = np.array(x_train1)
    x_train2 = np.array(x_train2)
    x_train3 = np.array(x_train3)
    x_train4 = np.array(x_train4)
    x_train5 = np.array(x_train5)
    x_train6 = np.array(x_train6)

    y_train = np.array(y_train)

    x_test1 = np.array(x_test1)
    x_test2 = np.array(x_test2)
    x_test3 = np.array(x_test3)
    x_test4 = np.array(x_test4)
    x_test5 = np.array(x_test5)
    x_test6 = np.array(x_test6)
    y_test = np.array(y_test)

    train_samples = x_train1.shape[0]
    train_indices = np.random.permutation(train_samples)

    x_trainC1 = x_train1[train_indices]
    x_trainC2 = x_train2[train_indices]
    x_trainC3 = x_train3[train_indices]
    x_trainC4 = x_train4[train_indices]
    x_trainC5 = x_train5[train_indices]
    x_trainC6 = x_train6[train_indices]
    y_train = y_train[train_indices]

    test_samples = x_test1.shape[0]
    test_indices = np.random.permutation(test_samples)

    x_testC1 = x_test1[test_indices]
    x_testC2 = x_test2[test_indices]
    x_testC3 = x_test3[test_indices]
    x_testC4 = x_test4[test_indices]
    x_testC5 = x_test5[test_indices]
    x_testC6 = x_test6[test_indices]
    y_test = y_test[test_indices]

    # Separate features and labels after shuffling
    return [x_trainC1, x_trainC2, x_trainC3, x_trainC4, x_trainC5, x_trainC6,
            x_testC1, x_testC2, x_testC3, x_testC4, x_testC5, x_testC6,
            y_train.astype(int), y_test.astype(int)]




class TPAnalysis:

    def __init__(self, data):
        """
        Initialize the Analysis class.

        Args:
        - Features: The feature data for analysis.
        - Labels: The labels corresponding to the feature data.
        """
        self.data = data
        self.epochs = 500
        self.perf_epochs = [100, 200, 300, 400, 500]

    def ComparativeAnalysis(self):
        """
        Perform Comparative Analysis to compare the proposed method with existing methods.

        Vary the training percentage and use different classification methods.

        Save the results in numpy files for each method and training percentage.
        """
        # Initialize lists to store comparative analysis results
        ComparativeResults = []

        TrainingPercentage = 0.4

        for i in range(6):
            cprint(f"[⚠️] Comparative Analysis Count Is {i} Out Of 6", 'cyan', on_color='on_grey')

            # Split the data into training and testing sets based on the training percentage
            data = train_test_split(self.data, train_size=TrainingPercentage)

            params = {'x_train1': data[0], 'x_train2': data[1], 'x_train3': data[2], 'x_train4': data[3],
                      'x_train5': data[4], 'x_train6': data[5],
                      'x_test1': data[6], 'x_test2': data[7], 'x_test3': data[8], 'x_test4': data[9],
                      'x_test5': data[10], 'x_test6': data[11],
                      'y_train': data[12], 'y_test': data[13], 'epochs': self.epochs}



            Ne = Network(**params)

            # Perform cl classification using different methods and get predictions
            output = [
                Ne.EfficientNet(),
                Ne.STIDNet(),
                Ne.CNN(),
                Ne.GLCM(),
                Ne.ViTDCNN(),
                Ne.ThreeDCNNLSTM(opt=1),
                Ne.ThreeDCNNLSTM(opt=2),
                Ne.ThreeDCNNLSTM(opt=3)]

            # Calculating the Performance
            ComparativeResults.append([Evaluation_Metrics(data[13], y_pred) for y_pred in output])
            # Increase the training percentage for the next iteration
            TrainingPercentage += 0.1

        perf_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        # , 'L', 'M'
        file_names = [f'Analysis1\\TP\\COM_{name}.npy' for name in perf_names]

        for i, file_name in enumerate(file_names):
            np.save(file_name, [perf[i] for perf in ComparativeResults])
        cprint("[✅] Execution of Comparative Analysis Completed ", 'green', on_color='on_grey')


    def PerformanceAnalysis(self):
        """
        Perform Performance Analysis to check the maximum performance of the proposed method.

        Vary the training percentage and epochs.

        Save the results in numpy files for each training percentage and epoch combination.
        """
        # Initialize lists to store performance analysis results
        PerformanceResults = []

        TrainingPercentage = 0.4

        for i in range(6):
            cprint(f"[⚠️] Performance Analysis Count Is {i} Out Of 6", 'cyan', on_color='on_grey')

            # Split the data into training and testing sets based on the training percentage
            data = train_test_split(self.data, train_size=TrainingPercentage)

            params = {'x_train1': data[0], 'x_train2': data[1], 'x_train3': data[2], 'x_train4': data[3],
                      'x_train5': data[4], 'x_train6': data[5],
                      'x_test1': data[6], 'x_test2': data[7], 'x_test3': data[8], 'x_test4': data[9],
                      'x_test5': data[10], 'x_test6': data[11],
                      'y_train': data[12], 'y_test': data[13], 'epochs': self.epochs}

            Ne = Network(**params)

            # Perform cl classification using different methods and get predictions
            output = [
                Ne.ThreeDCNNLSTM(epochs=self.perf_epochs[0]),
                Ne.ThreeDCNNLSTM(epochs=self.perf_epochs[1]),
                Ne.ThreeDCNNLSTM(epochs=self.perf_epochs[2]),
                Ne.ThreeDCNNLSTM(epochs=self.perf_epochs[3]),
                Ne.ThreeDCNNLSTM(epochs=self.perf_epochs[4])]

            # Calculating the Performance
            PerformanceResults.append([Evaluation_Metrics(data[13], y_pred) for y_pred in output])
            # Increase the training percentage for the next iteration
            TrainingPercentage += 0.1

        perf_names = ['A', 'B', 'C', 'D', 'E']
        # , 'L', 'M'
        file_names = [f'Analysis1\\TP\\PERF_{name}.npy' for name in perf_names]

        for i, file_name in enumerate(file_names):
            np.save(file_name, [perf[i] for perf in PerformanceResults])


        # Print a completion message
        cprint("[✅] Execution of Performance Analysis Completed ", 'green', on_color='on_grey')


    def RocAnalysis(self):
        cprint("[INFO] Executing Analysis", 'grey', on_color='on_white')

        FPR = []
        TPR = []

        # Define a list of training set percentages.
        Tr_Per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for i in range(len(Tr_Per)):
            cprint("Analysis Count Is {0} Out Of 9".format(i + 1), 'blue', on_color='on_grey')

            # Split the data into training and testing sets based on the training percentage
            data = train_test_split(self.data, train_size=Tr_Per[i])

            params = {'x_train1': data[0], 'x_train2': data[1], 'x_train3': data[2], 'x_train4': data[3],
                      'x_train5': data[4], 'x_train6': data[5],
                      'x_test1': data[6], 'x_test2': data[7], 'x_test3': data[8], 'x_test4': data[9],
                      'x_test5': data[10], 'x_test6': data[11],
                      'y_train': data[12], 'y_test': data[13], 'epochs': self.epochs}

            Ne = Network(**params)

            # Perform cl classification using different methods and get predictions

            # Perform cl classification using different methods and get predictions
            pred_c1 = Ne.EfficientNet()
            pred_c2 = Ne.STIDNet()
            pred_c3 = Ne.CNN()
            pred_c4 = Ne.GLCM()
            pred_c5 = Ne.ViTDCNN()
            pred_c6 = Ne.ThreeDCNNLSTM(opt=1)
            pred_c7 = Ne.ThreeDCNNLSTM(opt=2)
            pred_c8 = Ne.ThreeDCNNLSTM(opt=3)


            # Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for each classifier.
            [TPR1c, FPR1c] = Evaluation_Metrics1(data[13], pred_c1)
            [TPR2c, FPR2c] = Evaluation_Metrics1(data[13], pred_c2)
            [TPR3c, FPR3c] = Evaluation_Metrics1(data[13], pred_c3)
            [TPR4c, FPR4c] = Evaluation_Metrics1(data[13], pred_c4)
            [TPR5c, FPR5c] = Evaluation_Metrics1(data[13], pred_c5)
            [TPR6c, FPR6c] = Evaluation_Metrics1(data[13], pred_c6)
            [TPR7c, FPR7c] = Evaluation_Metrics1(data[13], pred_c7)
            [TPR8c, FPR8c] = Evaluation_Metrics1(data[13], pred_c8)


            # Store FPR and TPR for all classifiers.
            FPR_all = [FPR1c, FPR2c, FPR3c, FPR4c, FPR5c, FPR6c, FPR7c, FPR8c]
            TPR_all = [TPR1c, TPR2c, TPR3c, TPR4c, TPR5c, TPR6c, TPR7c, TPR8c]

            FPR.append(FPR_all)
            TPR.append(TPR_all)

        np.save(f'{os.getcwd()}\\Analysis1\\TPR.npy', TPR)
        np.save(f'{os.getcwd()}\\Analysis1\\FPR.npy', FPR)


        cprint("[INFO] Analysis Completed", 'green', on_color='on_grey')




class KFAnalysis:
    """
    K-fold analysis, often referred to as k-fold cross-validation, is a common technique used in machine learning
    and statistics to assess the performance and robustness of a predictive model. It is particularly useful when
    you have a limited amount of data and want to ensure that your model is not overfitting (performing well on
    training data but poorly on new, unseen data).
    """

    def __init__(self, data):
        """
        Initialize the Analysis class.

        Args:
        - Features: The feature data for analysis.
        - Labels: The labels corresponding to the feature data.
        """
        self.data = data
        self.epochs = 500
        self.folds = [6, 7, 8, 9, 10]
        self.perf_epochs = [100, 200, 300, 400, 500]


    @staticmethod
    def train_test_split(train, test, data):
        # A static method to extract training and testing data based on indices
        x_train1 = []
        x_train2 = []
        x_train3 = []
        x_train4 = []
        x_train5 = []
        x_train6 = []

        y_train = []

        x_test1 = []
        x_test2 = []
        x_test3 = []
        x_test4 = []
        x_test5 = []
        x_test6 = []

        y_test = []

        for i in range(len(data['image'])):
            if i in train:
                x_train1.append(data['comparative1'][i])
                x_train2.append(data['comparative2'][i])
                x_train3.append(data['comparative3'][i])
                x_train4.append(data['comparative4'][i])
                x_train5.append(data['comparative5'][i])
                x_train6.append(data['proposed'][i])
                y_train.append(data['labels'][i])
            else:

                x_test1.append(data['comparative1'][i])
                x_test2.append(data['comparative2'][i])
                x_test3.append(data['comparative3'][i])
                x_test4.append(data['comparative4'][i])
                x_test5.append(data['comparative5'][i])
                x_test6.append(data['proposed'][i])

                y_test.append(data['labels'][i])

        x_train1 = np.array(x_train1)
        x_train2 = np.array(x_train2)
        x_train3 = np.array(x_train3)
        x_train4 = np.array(x_train4)
        x_train5 = np.array(x_train5)
        x_train6 = np.array(x_train6)

        x_test1 = np.array(x_test1)
        x_test2 = np.array(x_test2)
        x_test3 = np.array(x_test3)
        x_test4 = np.array(x_test4)
        x_test5 = np.array(x_test5)
        x_test6 = np.array(x_test6)

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Separate features and labels after shuffling
        return [x_train1, x_train2, x_train3, x_train4, x_train5, x_train6,
                x_test1, x_test2, x_test3, x_test4, x_test5, x_test6,
                y_train.astype(int), y_test.astype(int)]

    def ComparativeAnalysis(self):
        # Perform Comparative Analysis
        ComparativeResults_all = []

        for i in range(len(self.folds)):
            # Iterate through different fold values
            # Iterate through different fold values
            cprint(f"[⚠️] No.of Fold : {self.folds[i]} ", 'cyan', on_color='on_grey')

            k_fold = KFold(n_splits=self.folds[i], random_state=1, shuffle=True)

            ComparativeResults = []

            for j, [train, test] in enumerate(k_fold.split(self.data['proposed'])):

                # Iterate through K-fold splits
                data = self.train_test_split(train, test, self.data)

                params = {'x_train1': data[0], 'x_train2': data[1], 'x_train3': data[2], 'x_train4': data[3],
                          'x_train5': data[4], 'x_train6': data[5],
                          'x_test1': data[6], 'x_test2': data[7], 'x_test3': data[8], 'x_test4': data[9],
                          'x_test5': data[10], 'x_test6': data[11],
                          'y_train': data[12], 'y_test': data[13], 'epochs': self.epochs}

                Ne = Network(**params)

                # Perform cl classification using different methods and get predictions
                output = [
                    Ne.EfficientNet(),
                    Ne.STIDNet(),
                    Ne.CNN(),
                    Ne.GLCM(),
                    Ne.ViTDCNN(),
                    Ne.ThreeDCNNLSTM(opt=1),
                    Ne.ThreeDCNNLSTM(opt=2),
                    Ne.ThreeDCNNLSTM(opt=3)]
                # Calculating the Performance
                ComparativeResults.append([Evaluation_Metrics(data[13], y_pred) for y_pred in output])

                # Increase the training percentage for the next iteration

                # Compute the mean of performance metrics for each method and fold
            ComparativeResults_all.append(np.mean(np.array(ComparativeResults), axis=0))

            # Save the results as numpy arrays

        perf_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        # , 'L', 'M'
        file_names = [f'Analysis1\\KF\\COM_{name}.npy' for name in perf_names]

        for i, file_name in enumerate(file_names):
            np.save(file_name, [perf[i] for perf in ComparativeResults_all])

        cprint("[✅] Execution of Comparative Analysis Completed ", 'green', on_color='on_grey')

    def PerformanceAnalysis(self):
        # Perform Comparative Analysis
        PerformanceResults_all = []

        for i in range(len(self.folds)):
            # Iterate through different fold values
            cprint(f"[⚠️] No.of Fold : {self.folds[i]} ", 'cyan', on_color='on_grey')

            k_fold = KFold(n_splits=self.folds[i], random_state=1, shuffle=True)

            PerformanceResults = []

            for j, [train, test] in enumerate(k_fold.split(self.data['proposed'])):
                # Iterate through K-fold splits
                data = self.train_test_split(train, test, self.data)

                params = {'x_train1': data[0], 'x_train2': data[1], 'x_train3': data[2], 'x_train4': data[3],
                          'x_train5': data[4], 'x_train6': data[5],
                          'x_test1': data[6], 'x_test2': data[7], 'x_test3': data[8], 'x_test4': data[9],
                          'x_test5': data[10], 'x_test6': data[11],
                          'y_train': data[12], 'y_test': data[13], 'epochs': self.epochs}

                Ne = Network(**params)

                # Perform cl classification using different methods and get predictions
                output = [
                    Ne.ThreeDCNNLSTM(epochs=self.perf_epochs[0]),
                    Ne.ThreeDCNNLSTM(epochs=self.perf_epochs[1]),
                    Ne.ThreeDCNNLSTM(epochs=self.perf_epochs[2]),
                    Ne.ThreeDCNNLSTM(epochs=self.perf_epochs[3]),
                    Ne.ThreeDCNNLSTM(epochs=self.perf_epochs[4])]
                # Calculating the Performance
                PerformanceResults.append([Evaluation_Metrics(data[13], y_pred) for y_pred in output])

                # Increase the training percentage for the next iteration

                # Compute the mean of performance metrics for each method and fold
            PerformanceResults_all.append(np.mean(np.array(PerformanceResults), axis=0))

        # Save the results as numpy arrays
        perf_names = ['A', 'B', 'C', 'D', 'E']
        # , 'L', 'M'
        file_names = [f'Analysis1\\KF\\PERF_{name}.npy' for name in perf_names]

        for i, file_name in enumerate(file_names):
            np.save(file_name, [perf[i] for perf in PerformanceResults_all])
        # Print a completion message
        cprint("[✅] Execution of Performance Analysis Completed ", 'green', on_color='on_grey')

