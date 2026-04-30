                                                                                                                                       
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset                                                                                     
ds = LeRobotDataset('kewalk/cup_task_0428')                                                                                                            
print('num_episodes:', ds.num_episodes)                                                                                                                
print('num_frames  :', ds.num_frames)                                                                                                                  
print('fps         :', ds.fps)                                                                                                                         
print('keys        :', list(ds[0].keys()))                                                                                                             
print('state shape :', ds[0]['observation.state'].shape if 'observation.state' in ds[0] else ds[0]['state'].shape)
print('action shape:', ds[0]['action'].shape if 'action' in ds[0] else ds[0]['actions'].shape)                                                         
print('task        :', ds.meta.tasks if hasattr(ds.meta, 'tasks') else 'n/a')                                                                          
