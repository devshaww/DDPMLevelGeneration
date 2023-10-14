import pandas as pd
from kaitaistruct import KaitaiStream
from tqdm import tqdm
from io import BytesIO
from level import Level
import zlib
import json


enum_reverse_mapping = {
    0: "goomba",
    1: "koopa",
    2: "piranha_flower",
    3: "hammer_bro",
    4: "block",
    5: "question_block",
    6: "hard_block",
    7: "ground",
    8: "coin",
    9: "pipe",
    10: "spring",
    11: "lift",
    12: "thwomp",
    13: "bullet_bill_blaster",
    14: "mushroom_platform",
    15: "bob_omb",
    16: "semisolid_platform",
    17: "bridge",
    18: "p_switch",
    19: "pow",
    20: "super_mushroom",
    21: "donut_block",
    22: "cloud",
    23: "note_block",
    24: "fire_bar",
    25: "spiny",
    26: "goal_ground",
    27: "goal",
    28: "buzzy_beetle",
    29: "hidden_block",
    30: "lakitu",
    31: "lakitu_cloud",
    32: "banzai_bill",
    33: "one_up",
    34: "fire_flower",
    35: "super_star",
    36: "lava_lift",
    37: "starting_brick",
    38: "starting_arrow",
    39: "magikoopa",
    40: "spike_top",
    41: "boo",
    42: "clown_car",
    43: "spikes",
    44: "big_mushroom",
    45: "shoe_goomba",
    46: "dry_bones",
    47: "cannon",
    48: "blooper",
    49: "castle_bridge",
    50: "jumping_machine",
    51: "skipsqueak",
    52: "wiggler",
    53: "fast_conveyor_belt",
    54: "burner",
    55: "door",
    56: "cheep_cheep",
    57: "muncher",
    58: "rocky_wrench",
    59: "track",
    60: "lava_bubble",
    61: "chain_chomp",
    62: "bowser",
    63: "ice_block",
    64: "vine",
    65: "stingby",
    66: "arrow",
    67: "one_way",
    68: "saw",
    69: "player",
    70: "big_coin",
    71: "half_collision_platform",
    72: "koopa_car",
    73: "cinobio",
    74: "spike_ball",
    75: "stone",
    76: "twister",
    77: "boom_boom",
    78: "pokey",
    79: "p_block",
    80: "sprint_platform",
    81: "smb2_mushroom",
    82: "donut",
    83: "skewer",
    84: "snake_block",
    85: "track_block",
    86: "charvaargh",
    87: "slight_slope",
    88: "steep_slope",
    89: "reel_camera",
    90: "checkpoint_flag",
    91: "seesaw",
    92: "red_coin",
    93: "clear_pipe",
    94: "conveyor_belt",
    95: "key",
    96: "ant_trooper",
    97: "warp_box",
    98: "bowser_jr",
    99: "on_off_block",
    100: "dotted_line_block",
    101: "water_marker",
    102: "monty_mole",
    103: "fish_bone",
    104: "angry_sun",
    105: "swinging_claw",
    106: "tree",
    107: "piranha_creeper",
    108: "blinking_block",
    109: "sound_effect",
    110: "spike_block",
    111: "mechakoopa",
    112: "crate",
    113: "mushroom_trampoline",
    114: "porkupuffer",
    115: "cinobic",
    116: "super_hammer",
    117: "bully",
    118: "icicle",
    119: "exclamation_block",
    120: "lemmy",
    121: "morton",
    122: "larry",
    123: "wendy",
    124: "iggy",
    125: "roy",
    126: "ludwig",
    127: "cannon_box",
    128: "propeller_box",
    129: "goomba_mask",
    130: "bullet_bill_mask",
    131: "red_pow_box",
    132: "on_off_trampoline",
}

df = pd.read_parquet('../../datasets/smm2/filtered_1.parquet')
tqdm.pandas()

for i, item in tqdm(df.iterrows(), desc='preprocessing %d samples' % df.shape[0], total=df.shape[0], position=0, leave=True):
    dic = {}
    level = Level(KaitaiStream(BytesIO(zlib.decompress(item["level_data"])))).overworld
    dic['width'] = (level.boundary_right - level.boundary_left) // 16
    dic['height'] = (level.boundary_top - level.boundary_bottom) // 16
    dic['difficulty'] = item['difficulty']
    dic['gamestyle'] = item['gamestyle']
    dic['theme'] = level.theme.value
    dic['object_count'] = level.object_count
    dic['ground_count'] = level.ground_count

    objects, grounds = [], []
    for j in range(level.object_count):
        obj = level.objects[j]
        element = {'name': enum_reverse_mapping[obj.id.value],
                   'x': obj.x // 160,
                   'y': obj.y // 160,
                   'id': obj.id.value,
                   'h': obj.height,
                   'w': obj.width}
        objects.append(element)
    dic['objects'] = objects

    for k in range(level.ground_count):
        grd = level.ground[k]
        ground = {'x': grd.x, 'y': grd.y, 'id': grd.id, 'bid': grd.background_id}
        grounds.append(ground)
    dic['ground'] = grounds

    with open(f"../../datasets/smm2/jsons/level_{i}.json", 'w') as f:
        json.dump(dic, f, indent=4)


