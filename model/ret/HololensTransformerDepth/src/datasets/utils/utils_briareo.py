import json

# full 675, full_no_fingers 145, mod 192
def from_json_to_list(json_file):
    with open(json_file) as f:
        j = json.load(f)
        if j['frame'] != 'invalid':
            j_vector = [
                # palm
                j['frame']['right_hand']['palm_position'][0],
                j['frame']['right_hand']['palm_position'][1],
                j['frame']['right_hand']['palm_position'][2],
                j['frame']['right_hand']['palm_position'][3],
                j['frame']['right_hand']['palm_position'][4],
                j['frame']['right_hand']['palm_position'][5],
                j['frame']['right_hand']['palm_normal'][0],
                j['frame']['right_hand']['palm_normal'][1],
                j['frame']['right_hand']['palm_normal'][2],
                j['frame']['right_hand']['palm_normal'][3],
                j['frame']['right_hand']['palm_normal'][4],
                j['frame']['right_hand']['palm_normal'][5],
                j['frame']['right_hand']['palm_velocity'][0],
                j['frame']['right_hand']['palm_velocity'][1],
                j['frame']['right_hand']['palm_velocity'][2],
                j['frame']['right_hand']['palm_velocity'][3],
                j['frame']['right_hand']['palm_velocity'][4],
                j['frame']['right_hand']['palm_velocity'][5],
                j['frame']['right_hand']['palm_width'],
                j['frame']['right_hand']['pinch_strength'],
                j['frame']['right_hand']['grab_strength'],
                j['frame']['right_hand']['direction'][0],
                j['frame']['right_hand']['direction'][1],
                j['frame']['right_hand']['direction'][2],
                j['frame']['right_hand']['direction'][3],
                j['frame']['right_hand']['direction'][4],
                j['frame']['right_hand']['direction'][5],
                j['frame']['right_hand']['sphere_center'][0],
                j['frame']['right_hand']['sphere_center'][1],
                j['frame']['right_hand']['sphere_center'][2],
                j['frame']['right_hand']['sphere_center'][3],
                j['frame']['right_hand']['sphere_center'][4],
                j['frame']['right_hand']['sphere_center'][5],
                j['frame']['right_hand']['sphere_radius'],
                # wrist
                j['frame']['right_hand']['wrist_position'][0],
                j['frame']['right_hand']['wrist_position'][1],
                j['frame']['right_hand']['wrist_position'][2],
                j['frame']['right_hand']['wrist_position'][3],
                j['frame']['right_hand']['wrist_position'][4],
                j['frame']['right_hand']['wrist_position'][5],
                # pointables
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_0']['direction'][0],
                j['frame']['right_hand']['pointables']['p_0']['direction'][1],
                j['frame']['right_hand']['pointables']['p_0']['direction'][2],
                j['frame']['right_hand']['pointables']['p_0']['direction'][3],
                j['frame']['right_hand']['pointables']['p_0']['direction'][4],
                j['frame']['right_hand']['pointables']['p_0']['direction'][5],
                j['frame']['right_hand']['pointables']['p_0']['width'],
                j['frame']['right_hand']['pointables']['p_0']['length'],
                float(j['frame']['right_hand']['pointables']['p_0']['is_extended']),
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_1']['direction'][0],
                j['frame']['right_hand']['pointables']['p_1']['direction'][1],
                j['frame']['right_hand']['pointables']['p_1']['direction'][2],
                j['frame']['right_hand']['pointables']['p_1']['direction'][3],
                j['frame']['right_hand']['pointables']['p_1']['direction'][4],
                j['frame']['right_hand']['pointables']['p_1']['direction'][5],
                j['frame']['right_hand']['pointables']['p_1']['width'],
                j['frame']['right_hand']['pointables']['p_1']['length'],
                float(j['frame']['right_hand']['pointables']['p_1']['is_extended']),
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_2']['direction'][0],
                j['frame']['right_hand']['pointables']['p_2']['direction'][1],
                j['frame']['right_hand']['pointables']['p_2']['direction'][2],
                j['frame']['right_hand']['pointables']['p_2']['direction'][3],
                j['frame']['right_hand']['pointables']['p_2']['direction'][4],
                j['frame']['right_hand']['pointables']['p_2']['direction'][5],
                j['frame']['right_hand']['pointables']['p_2']['width'],
                j['frame']['right_hand']['pointables']['p_2']['length'],
                float(j['frame']['right_hand']['pointables']['p_2']['is_extended']),
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_3']['direction'][0],
                j['frame']['right_hand']['pointables']['p_3']['direction'][1],
                j['frame']['right_hand']['pointables']['p_3']['direction'][2],
                j['frame']['right_hand']['pointables']['p_3']['direction'][3],
                j['frame']['right_hand']['pointables']['p_3']['direction'][4],
                j['frame']['right_hand']['pointables']['p_3']['direction'][5],
                j['frame']['right_hand']['pointables']['p_3']['width'],
                j['frame']['right_hand']['pointables']['p_3']['length'],
                float(j['frame']['right_hand']['pointables']['p_3']['is_extended']),
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_4']['direction'][0],
                j['frame']['right_hand']['pointables']['p_4']['direction'][1],
                j['frame']['right_hand']['pointables']['p_4']['direction'][2],
                j['frame']['right_hand']['pointables']['p_4']['direction'][3],
                j['frame']['right_hand']['pointables']['p_4']['direction'][4],
                j['frame']['right_hand']['pointables']['p_4']['direction'][5],
                j['frame']['right_hand']['pointables']['p_4']['width'],
                j['frame']['right_hand']['pointables']['p_4']['length'],
                float(j['frame']['right_hand']['pointables']['p_4']['is_extended']),
            ]
        else:
            j_vector = False

        return j_vector, j