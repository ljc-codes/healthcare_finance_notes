import os
import dill as pickle


def save_component(save_folder,
                   save_path,
                   component):
    """
    Saves a picklable object to disk, creating folders if necessary

    Arguments:
        save_folder (str): folder to save object to
        save_path (str): full path of object file to save
        component (Picklable object): object to save
    """

    # create folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # save vectorizer
    with open(save_path, "wb") as f:
        pickle.dump(component, f)


def load_component(function,
                   data_input,
                   component_folder,
                   component_name,
                   component_config,
                   label):
    """
    Loads a picklable object if it exists, otherwise create the object with the provided function

    Arguments:
        function (function): function to create object if it does not exist
        data (data object): data object to be consumed by function
        component_folder (str): folder to save component to
        component_name (str): name to save component as
        component_config (dict): configuration for a specific fit function
        label (str): component label for logging

    Returns:
        component (object): object to use in transforming data
    """

    # initialize component path
    component_path = os.path.join(component_folder, component_name) + ".pkl"

    if os.path.isfile(component_path):
        print("Loading {}...".format(label))
        with open(component_path, "rb") as f:
            component = pickle.load(f)
    else:
        # fit component if it doesn't already exist
        print("Creating {}...".format(label))
        component = function(data_input=data_input,
                             config=component_config)

        save_component(component_folder=component_folder,
                       component_path=component_path,
                       component=component)

    return component
