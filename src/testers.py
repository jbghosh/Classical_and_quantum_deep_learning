import numpy as np
import pandas as pd

class Tester:
    """Class for benchmarking the algorithms. 
    
    It takes into account possible time shift of the detection.
    """

    def __init__(self, alarms_path, shutdown_path, data_len):
        self.true_anomaly = np.array([-1] * data_len)
        
        if not alarms_path is None:
            self.alarms    = pd.read_csv(alarms_path)['0'].values.astype(int)
            self.true_anomaly[self.alarms] = 1
        if not shutdown_path is None:
            self.shutdowns = pd.read_csv(shutdown_path)['0'].values.astype(int)
            self.true_anomaly[self.shutdowns] = 1

        self.true_anomaly_idx = np.where(self.true_anomaly == 1)[0]

        self.tolerance_backwards = 5
        self.tolerance_forwards = 1
        self.tolerance_steps = np.arange(-self.tolerance_backwards, self.tolerance_forwards + 1)
        self.tolerance_steps = self.tolerance_steps[self.tolerance_steps != 0]

    def _get_true_positives(self, predicted):
        """Calculate true positives with a time tolerance.

        Args:
            predicted (list): list of predicted inliers and outliers, where outliers
                are marked with -1

        Returns:
            int: true positives count
        """
        true_positive_tolerated = 0

        for i in self.true_anomaly_idx:
            found = predicted[i] == 1
            for j in np.arange(-self.tolerance_backwards, self.tolerance_forwards + 1):
                if found or i + j > len(predicted): 
                    break
                found = predicted[i + j] == 1
            true_positive_tolerated += found
            
        return true_positive_tolerated
    
    def _get_false_positives(self, predicted):
        """Calculate false positives with a time tolerance.

        Args:
            predicted (list): list of predicted inliers and outliers, where outliers
                are marked with -1

        Returns:
            int: false positives count
        """
        false_positive_tolerated = 0

        for i in np.where(predicted==1)[0]:
            found = self.true_anomaly[i] == 1
            for j in np.arange(-self.tolerance_forwards, self.tolerance_backwards + 1):
                if found or i + j > len(self.true_anomaly) - 1: 
                    break
                found = self.true_anomaly[i + j] == 1
                
            false_positive_tolerated += not found
            
        return false_positive_tolerated
    
    def _get_false_negatives(self, predicted):
        """Calculate false negatives with a time tolerance.

        Args:
            predicted (list): list of predicted inliers and outliers, where outliers
                are marked with -1

        Returns:
            int: false negatives count
        """
        false_negative_tolerated = 0

        for i in self.true_anomaly_idx:
            found = predicted[i] == 1
            for j in np.arange(-self.tolerance_backwards, self.tolerance_forwards + 1):
                if found or i + j > len(predicted): 
                    break
                found = predicted[i + j] == 1
            
            false_negative_tolerated += not found
            
        return false_negative_tolerated

    def _get_true_negatives(self, predicted):
        """Calculate true negatives

        Args:
        predicted (list): list of predicted inliers and outliers, where outliers
                are marked with -1

        Returns:
            int: true negatives count
        """
        detected_idx = np.where(predicted == -1)[0]
        return sum(self.true_anomaly[detected_idx] == -1)

    def get_precision(self, predicted):
        """Calculate fraction of relevant instances among the retrieved instances

        Args:
            predicted (list): list of predicted inliers and outliers, where outliers
                are marked with -1

        Returns:
            float: precision score [0, 1]
        """
        true_positives  = self._get_true_positives(predicted)
        false_positives = self._get_false_positives(predicted)

        if true_positives + false_positives == 0:
            return 0

        return true_positives / (true_positives + false_positives)

    def get_recall(self, predicted):
        """Calculate the fraction of positive samples retrieved

        Args:
            predicted (list): list of predicted inliers and outliers, where outliers
                are marked with -1

        Returns:
            float: recall score [0, 1]
        """
        true_positives  = self._get_true_positives(predicted)
        false_negatives = self._get_false_negatives(predicted)

        if true_positives + false_negatives == 0:
            return 0

        return true_positives / (true_positives + false_negatives)

    def get_f1_score(self, predicted):
        """Calculate the weighted average of precision and recall

        Args:
            predicted (list): list of predicted inliers and outliers, where outliers
                are marked with -1

        Returns:
            float: f1 score [0, 1]
        """
        precision = self.get_precision(predicted)
        recall = self.get_recall(predicted)

        if precision + recall == 0:
            return 0

        return 2 * precision * recall / (precision + recall)
