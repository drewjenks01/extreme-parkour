import isaacgym
import yaml

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

with open('/home/andrewjenkins/Documents/extreme-parkour/legged_gym/logs/parkour_new/000-00-phase1_no_contact/config_preferred.yaml', 'r') as file:
    all_cfg = yaml.load(file, Loader=yaml.FullLoader)['Cfg']
    cfg = class_to_dict(all_cfg)


print(cfg['asset'].keys())