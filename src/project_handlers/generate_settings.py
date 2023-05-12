import json

project_info_path = 'project_info.json'

data = {
    'sampling_rate' : 0.03,
    't_size' : 200,
    'resource_root' : 'resources',
    'data_root' : 'resources/data',
    'models_root' : 'resources/models',
    'mber_path' : 'resources/models/mber.pickle',
    'resnet_path' : 'resources/models/resnet.pth',
    'rnn_resnet_path' : 'resources/models/rnn_resnet.pth',
    'triangulation_root' : 'resources/mesh'
    }

with open(project_info_path, 'w') as f:
    json.dump(data, f)

print('Generated ' + project_info_path)