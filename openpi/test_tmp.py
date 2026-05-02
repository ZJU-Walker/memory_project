import json, numpy as np
p = 'assets/pi05_yam_cup_lora/kewalk/cup_task_0428/norm_stats.json'                                                                                    
d = json.load(open(p))                                                                                                                                 
print('keys:', list(d['norm_stats'].keys()))
for k, v in d['norm_stats'].items():                                                                                                                   
    mean = np.array(v['mean']); std = np.array(v['std'])                                                                                               
    print(f'{k:8s} mean={np.round(mean,4)}  std={np.round(std,4)}')
    assert not np.any(np.isnan(mean)), f'NaN in {k} mean'                                                                                              
    assert not np.any(np.isnan(std)),  f'NaN in {k} std'
    assert np.all(std > 0),             f'Zero std in {k}'                                                                                             
print('all stats valid')