import os
import re

def calculate_consistency_single(flow1,flow2,reliable1,reliable2):
    os.system('./consistencyChecker/consistencyChecker {} {} {}'.format(flow1,flow2,reliable1))
    os.system('./consistencyChecker/consistencyChecker {} {} {}'.format(flow2,flow1,reliable2)) 


# forward_path = '/home/xzy/video/PWC-Net/PyTorch/forward_flow_output'
# backward_path = '/home/xzy/video/PWC-Net/PyTorch/backward_flow_output'
# out_path = './outputs/consistency'
def calculate_consistency(forward_path,backward_path,forward_consistency_out_path,backward_consistency_out_path):
    forward_flow = os.listdir(forward_path)
    backward_flow = os.listdir(backward_path)
    forward_flow.sort(key=lambda x: int(re.split('\.|\_',x)[1]))
    backward_flow.sort(key=lambda x: int(re.split('\.|\_',x)[1]))
    i = 1
    for forward,backward in zip(forward_flow,backward_flow):
        print('Now, it is calcaulating consistency between flows {} and {}'.format(forward,backward))
        flow1 = os.path.join(forward_path,forward)
        flow2 = os.path.join(backward_path,backward)
        reliable1 = 'reliable_forward_{}.txt'.format(i)
        reliable2 = 'reliable_backward_{}.txt'.format(i)
        reliable1 = os.path.join(forward_consistency_out_path,reliable1)
        reliable2 = os.path.join(backward_consistency_out_path,reliable2)
        calculate_consistency_single(flow1,flow2,reliable1,reliable2)
        i +=1 


# data/video_picture_flow/11_ponte_WS_pan_Videv/
# forward_flow
# backward_flow
def calculate_consistency_multi(in_path,out_path):
    for i,dir_path in enumerate(os.listdir(in_path)):
        forward_path = os.path.join(in_path,dir_path,'forward_flow')
        backward_path = os.path.join(in_path,dir_path,'backward_flow')
        forward_consistency_out_path = os.path.join(in_path,dir_path,'forward_consistency')
        backward_consistency_out_path = os.path.join(in_path,dir_path,'backward_consistency')
        if not os.path.exists(forward_consistency_out_path):
            os.makedirs(forward_consistency_out_path) 
            os.makedirs(backward_consistency_out_path)
        print('It is calculating dir {} consistency and the forward_consistency is {} and backward_consistency {} !'.format(i,forward_consistency_out_path,backward_consistency_out_path))
        calculate_consistency(forward_path,backward_path,forward_consistency_out_path,backward_consistency_out_path)
        
        


if __name__ == "__main__":
    forward_path = '/home/xzy/video/data/video-picture-flow/seagul-H264/forward_flow'
    backward_path = '/home/xzy/video/data/video-picture-flow/seagul-H264/backward_flow'
    forward_consistency_out_path = '/home/xzy/video/data/video-picture-flow/seagul-H264/forward_consistency'
    backward_consistency_out_path = '/home/xzy/video/data/video-picture-flow/seagul-H264/backward_consistency'

    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    calculate_consistency(forward_path,backward_path,forward_consistency_out_path,backward_consistency_out_path)
    
    # in_path = '/home/xzy/video/data/video-picture-flow/'
    # out_path = '/home/xzy/video/data/video-picture-flow/'
    # calculate_consistency_multi(in_path,out_path)