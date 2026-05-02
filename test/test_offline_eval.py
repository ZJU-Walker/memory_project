# import numpy as np                                                                                                                                     
# from i2rt.robots.kinematics import Kinematics
# from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml                                                                        
# from i2rt.robots.get_robot import get_yam_robot                                                                                            
                                                                                                                                                        
# kin = Kinematics(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER), 'grasp_site')                                                       
# print('FK(zeros) translation:', kin.fk(np.zeros(6))[:3, 3])                                                                                            
                                                                                                                                                        
# robot = get_yam_robot(arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310, sim=True)                                                            
# print('SimRobot type      :', type(robot).__name__)                                                                                                    
# print('SimRobot joint pos :', robot.get_joint_pos())                                                                                                   
# print('SimRobot DoF       :', len(robot.get_joint_pos()))      


import numpy as np
d = np.load('/home/kewalk/memory_project/eval/results_demo1.npz')                                                                                      
gt = d['gt_action'][:, 6]; pred = d['pred_first'][:, 6]; state = d['state'][:, 6]                                                                      
print(f'state    j7 range: [{state.min():.4f}, {state.max():.4f}]  span={state.max()-state.min():.4f}  mean={state.mean():.4f}')                       
print(f'gt_action j7 range: [{gt.min():.4f}, {gt.max():.4f}]  span={gt.max()-gt.min():.4f}  mean={gt.mean():.4f}')                                     
print(f'pred_first j7 range: [{pred.min():.4f}, {pred.max():.4f}]  span={pred.max()-pred.min():.4f}  mean={pred.mean():.4f}')                          
print()                                                                                                                                                
# When does the gripper close in the demo? Find times where j7 is below half of its range                                                              
mid = 0.5 * (state.min() + state.max())                                                                                                                
closing_steps = np.where(state < mid)[0]
print(f'midpoint = {mid:.4f}; steps where state j7 below midpoint: {len(closing_steps)}/{len(state)}')                                                 
if len(closing_steps) > 0:                                                                                                                             
    print(f'  first close-ish step: {closing_steps[0]}, last: {closing_steps[-1]}')                                                                    
    print(f'  state at that step: {state[closing_steps[0]]:.4f}; gt action: {gt[closing_steps[0]]:.4f}; pred: {pred[closing_steps[0]]:.4f}')  