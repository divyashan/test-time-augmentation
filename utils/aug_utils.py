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

    def parse_flips(param):
        if 'hflip' in str(param):
            return 1
        elif 'vflip' in str(param):
            return 2
        return 0
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
                'modified_five_crop': parse_modified_five_crop, 'pil': parse_pil, 'vflip': parse_flip, 
                'flips': parse_flips}

    parsed_params = []
    parse_fs = [parse_fs_dict[aug] for aug in aug_order]
    for params in aug_transform_parameters:
        parsed = [parse_f(param) for parse_f,param in zip(parse_fs,params)]
        parsed_params.append(parsed)
    return parsed_params

def invert_aug_list(aug_list, aug_order):
    def parse_five_crop(num):
        num_map = {0:'center crop', 1: 'upper left crop', 2: 'upper right crop', 3: 'lower left crop',
                   4: 'lower right', 5: 'uncropped original'}
        return num_map[num]
    def parse_flip(num):
        num_map = {0: 'no hflip', 1: 'hflip'}
        return num_map[num]
    def parse_scale(num):
        return str(num*100)[:3] + "% zoomed in"

    inv_parse_fs = {'modified_five_crop': parse_five_crop, 'hflip': parse_flip, 'scale': parse_scale,
                    'five_crop': parse_five_crop}

    all_descriptions = []
    for aug_nums in aug_list:
        
        descriptions = [inv_parse_fs[aug](aug_nums[i]) for i,aug in enumerate(aug_order)]
        descriptions = ','.join(descriptions)
        all_descriptions.append(descriptions)
    return all_descriptions
