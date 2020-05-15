import pdb

def write_aug_list(aug_transform_parameters, aug_order):
    def parse_five_crop(param):
        pos_map = {'c ':0, 'lt':1, 'rt':2, 'lb':3, 'rb':4}
        pos  = str(param.func).split('_')[1][:2]
        crop_h = param.keywords['crop_h']
        crop_w = param.keywords['crop_w']
        return pos_map[pos]
    
    def parse_modified_five_crop(param):
        pos_map = {'or': 5, 'lt': 1, 'rt': 2, 'lb': 3, 'rb': 4, 'c ': 0}
        pos  = str(param.func).split('_')[1][:2]
        return pos_map[pos] 

    def parse_flip(param):
        return int(param)

    def parse_color_jitter(param):
        return int(param)

    def parse_rotation(param):
        return param.keywords['angle']

    def parse_scale(param):
        return float(param) - 1

    def parse_pil(param):
        param_level = str(param).split(',')[1][0]
        param_name = str(param.name).split('(')[0]
        name_base_dict = {'Posterize': 10, 'TranslateY': 20 , 'Brightness': 30, 
                          'ShearY': 40, 'Color': 50, 'FlipLR': 60, 
                          'ShearX': 70, 'FlipUD': 80, 'Equalize': 90, 
                          'Contrast': 100, 'Blur': 110, 'Rotate': 0, 
                          'TranslateX': 130, 'AutoContrast': 140, 'Invert': 150, 
                          'Solarize':160, 'Smooth':170, 'Cutout':180, 
                          'CropBilinear':190, 'Sharpness':120}
        return name_base_dict[param_name] + int(param_level)
    parse_fs_dict = {'five_crop': parse_five_crop, 'rotation': parse_rotation,
                'colorjitter': parse_color_jitter, 'hflip': parse_flip, 'scale': parse_scale, 
                'modified_five_crop': parse_modified_five_crop, 'pil': parse_pil, 'vflip': parse_flip}

    parsed_params = []
    parse_fs = [parse_fs_dict[aug] for aug in aug_order]
    for params in aug_transform_parameters:
        parsed = [parse_f(param) for parse_f,param in zip(parse_fs,params)]
        parsed_params.append(parsed)
    return parsed_params
