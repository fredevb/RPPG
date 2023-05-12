import os
from lib.models.rppg_dataset import RPPGDataset

class ProjectDataHandler():

    def __init__(self, data_root):
        self.data_root = data_root

    def load_data(self, dataset_name, transform=None):
        data_root = self.get_dataset_root(dataset_name)
        csv_path = self.get_data_csv_path(dataset_name)
        dataset = RPPGDataset(csv_path, data_root, transform=transform)
        return dataset

    def load_train_validation_test(self, dataset_name, transform=None):
        data_root = self.get_dataset_root(dataset_name)
        train_path, validation_path, test_path = self.get_train_validation_test_csv_paths(dataset_name)
        train_dataset = RPPGDataset(train_path, data_root, transform=transform)
        validation_dataset = RPPGDataset(validation_path, data_root, transform=transform)
        test_dataset = RPPGDataset(test_path, data_root, transform=transform)
        return train_dataset, validation_dataset, test_dataset
    
    def get_dataset_root(self, dataset_name):
        data_root = os.path.join(self.data_root, dataset_name)
        return data_root

    def get_data_csv_path(self, dataset_name):
        csv_path = os.path.join(self.data_root, dataset_name, 'data.csv')
        return csv_path
    
    def get_train_validation_test_csv_paths(self, dataset_name):
        train_path = os.path.join(self.data_root, dataset_name, 'train.csv')
        validation_path = os.path.join(self.data_root, dataset_name, 'validation.csv')
        test_path = os.path.join(self.data_root, dataset_name, 'test.csv')
        return train_path, validation_path, test_path