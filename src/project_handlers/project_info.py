import os
import json

class ProjectInfo():

    def __init__(self):
        self.info = self.get_project_info()
        self.sampling_rate = self.info['sampling_rate']
        self.t_size = self.info['t_size']
        self.resource_root = self.info['resource_root']
        self.data_root = self.info['data_root']
        self.model_root = self.info['models_root']
        self.traingulation_root = self.info['triangulation_root']

    def get_project_info(self):
        project_file_name = 'project_info.json'
        while not os.path.exists(project_file_name):
            os.chdir('..')
        with open(project_file_name, 'r') as f:
            project_info = json.load(f)
        return project_info