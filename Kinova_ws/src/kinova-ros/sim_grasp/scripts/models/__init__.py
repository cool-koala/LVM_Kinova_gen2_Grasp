def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from models.ggcnn.ggcnn import GGCNN
        return GGCNN
    
    elif network_name == 'ggcnn2':
        from models.ggcnn.ggcnn2 import GGCNN2
        return GGCNN2

    elif network_name == 'deeplab-v3':
        from models.deeplabv3.deeplab import DeepLab
        return DeepLab

    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
