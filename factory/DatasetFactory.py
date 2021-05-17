from .DatasetWrapperFactory import get_datasetWrapper

def get_dataset(wrapper, config):
    wrapper = get_datasetWrapper(**wrapper)
    wrapper.load_data(**config)
    return wrapper