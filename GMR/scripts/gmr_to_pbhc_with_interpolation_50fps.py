#!/usr/bin/env python3
"""
GMR数据格式转换并执行插值处理
将GMR输出的PKL文件转换为PBHC格式，并添加动作开头和结尾的插值
"""

import numpy as np
import joblib
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def convert_gmr_to_pbhc_format(gmr_data):
    """
    转换GMR格式到PBHC格式
    GMR: 'root_pos', 'root_rot', 'dof_pos', 'fps'
    PBHC: 'root_trans_offset', 'root_rot', 'dof', 'fps', 'pose_aa', 'smpl_joints', 'contact_mask'
    """
    root_pos = gmr_data['root_pos']
    root_rot = gmr_data['root_rot']
    dof_pos = gmr_data['dof_pos']
    fps = gmr_data.get('fps', 50)
    
    num_frames = root_pos.shape[0]
    num_dof = dof_pos.shape[1]
    
    # 生成contact_mask (默认双脚接地)
    contact_mask = np.ones((num_frames, 2)) * 0.5
    
    # 简化版pose_aa (不需要dof_axis文件)
    root_aa = R.from_quat(root_rot).as_rotvec()
    pose_aa = np.zeros((num_frames, 27, 3), dtype=np.float32)
    pose_aa[:, 0, :] = root_aa
    
    pbhc_data = {
        'root_trans_offset': root_pos.astype(np.float32),
        'root_rot': root_rot.astype(np.float32),
        'dof': dof_pos.astype(np.float32),
        'fps': fps,
        'pose_aa': pose_aa,
        'smpl_joints': np.zeros_like(pose_aa),
        'contact_mask': contact_mask.astype(np.float32)
    }
    
    return pbhc_data


def interpolate_motion(input_data, start_ext_frames, end_ext_frames, default_pose, contact_mask, fix_root_rot=False):
    """
    在动作开头和结尾添加插值
    """
    root_trans = input_data[:, :3]
    root_rot = input_data[:, 3:7]
    dof_pos = input_data[:, 7:]
    
    num_dof = dof_pos.shape[1]
    print(f"检测到DOF数量: {num_dof}")
    
    default_rt = default_pose[0:3]
    default_rr = default_pose[3:7]
    default_dof = default_pose[7:]
    
    start_rot_aa = R.from_quat(root_rot[0]).as_euler('ZYX')
    end_rot_aa = R.from_quat(root_rot[-1]).as_euler('ZYX')
    default_rr_aa = R.from_quat(default_rr).as_euler('ZYX')
    
    # ========== 起始处插值 ==========
    start_rr, start_dof = [], []
    if start_ext_frames > 0:
        # root trans (主要插值Z轴高度)
        start_z = np.linspace(default_rt[2], root_trans[0, 2], start_ext_frames)
        start_root_trans = np.zeros((start_ext_frames, 3))
        start_root_trans[:, 0] = root_trans[0, 0]
        start_root_trans[:, 1] = root_trans[0, 1]
        start_root_trans[:, 2] = start_z
        
        # dof pos (线性插值)
        start_dof = np.linspace(default_dof, dof_pos[0],
                                num=start_ext_frames + 1,
                                endpoint=False)[1:].reshape(-1, num_dof)
        
        # root rot (球面插值)
        if not fix_root_rot:
            rotations = R.from_euler('ZYX', [
                np.concatenate((start_rot_aa[0:1], default_rr_aa[1:])),
                np.concatenate((start_rot_aa[0:1], start_rot_aa[1:]))
            ])
            times = np.linspace(0, 1, start_ext_frames)
            slerp = Slerp([0, 1], rotations)
            interp_rots = slerp(times).as_euler('ZYX')
            start_rr = R.from_euler('ZYX', interp_rots).as_quat()
    
    # ========== 结束处插值 ==========
    end_rr, end_dof = [], []
    if end_ext_frames > 0:
        # root trans
        end_z = np.linspace(root_trans[-1, 2], default_rt[2], end_ext_frames)
        end_root_trans = np.zeros((end_ext_frames, 3))
        end_root_trans[:, 0] = root_trans[-1, 0]
        end_root_trans[:, 1] = root_trans[-1, 1]
        end_root_trans[:, 2] = end_z
        
        # dof pos
        end_dof = np.linspace(dof_pos[-1], default_dof,
                              num=end_ext_frames + 1)[1:].reshape(-1, num_dof)
        
        # root rot
        if not fix_root_rot:
            end_rotations = R.from_euler('ZYX', [
                np.concatenate((end_rot_aa[0:1], default_rr_aa[1:])),
                np.concatenate((end_rot_aa[0:1], end_rot_aa[1:]))
            ])
            times = np.linspace(1, 0, end_ext_frames)
            slerp = Slerp([0, 1], end_rotations)
            interp_rots = slerp(times).as_euler('ZYX')
            end_rr = R.from_euler('ZYX', interp_rots).as_quat()
    
    # ========== 合并数据 ==========
    new_root_trans = np.vstack([
        start_root_trans if start_ext_frames > 0 else np.empty((0, 3)),
        root_trans,
        end_root_trans if end_ext_frames > 0 else np.empty((0, 3))
    ])
    
    if not fix_root_rot:
        new_root_rot = np.vstack([
            start_rr if start_ext_frames > 0 else np.empty((0, 4)),
            root_rot,
            end_rr if end_ext_frames > 0 else np.empty((0, 4))
        ])
    else:
        total_frame = start_ext_frames + input_data.shape[0] + end_ext_frames
        new_root_rot = np.tile(default_rr, (total_frame, 1))
    
    new_dof_pos = np.vstack([
        start_dof if start_ext_frames > 0 else np.empty((0, num_dof)),
        dof_pos,
        end_dof if end_ext_frames > 0 else np.empty((0, num_dof))
    ])
    
    output_data = np.concatenate((new_root_trans, new_root_rot, new_dof_pos), axis=1)
    return output_data, contact_mask


def process_gmr_file(input_file, output_file, start_inter_frames, end_inter_frames, 
                     start_frame=0, end_frame=-1):
    """
    处理GMR文件：转换格式 + 添加插值
    """
    print(f"\n{'='*60}")
    print(f"处理文件: {input_file}")
    print(f"{'='*60}")
    
    # 1. 加载GMR数据
    gmr_data = joblib.load(input_file)
    print(f"GMR数据格式键: {list(gmr_data.keys())}")
    print(f"原始帧数: {gmr_data['root_pos'].shape[0]}")
    print(f"DOF数量: {gmr_data['dof_pos'].shape[1]}")
    
    # 2. 转换为PBHC格式
    pbhc_data = convert_gmr_to_pbhc_format(gmr_data)
    
    # 3. 裁剪数据
    if end_frame == -1:
        end_frame = pbhc_data['dof'].shape[0]
    
    print(f"裁剪范围: [{start_frame}:{end_frame}]")
    
    root_trans = pbhc_data['root_trans_offset'][start_frame:end_frame]
    root_rot = pbhc_data['root_rot'][start_frame:end_frame]
    dof = pbhc_data['dof'][start_frame:end_frame]
    contact_mask = pbhc_data['contact_mask'][start_frame:end_frame]
    
    # 4. 确定默认姿态
    num_dof = dof.shape[1]
    if num_dof == 23:
        default_pose = np.array([
            0.0, 0.0, 0.80,                         # root pos
            0.0, 0.0, 0.0, 1.0,                     # root rot
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,         # 左腿
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,         # 右腿
            0.0, 0.0, 0.0,                          # 腰部
            0.2, 0.2, 0.0, 0.9,                     # 左臂
            0.2, -0.2, 0.0, 0.9                     # 右臂
        ])
    elif num_dof == 29:
        default_pose = np.array([
            0.0, 0.0, 0.80,                         # root pos
            0.0, 0.0, 0.0, 1.0,                     # root rot
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,         # 左腿 (6)
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,         # 右腿 (6)
            0.0, 0.0, 0.0,                          # 腰部 (3)
            0.2, 0.2, 0.0, 0.9, 0.0, 0.0, 0.0,      # 左臂 (7)
            0.2, -0.2, 0.0, 0.9, 0.0, 0.0, 0.0      # 右臂 (7)
        ])
    else:
        raise ValueError(f"不支持的DOF数量: {num_dof}")
    
    # 5. 准备插值的contact_mask
    cit = [0.5, 0.5]
    contact_mask_start = np.tile([cit], (start_inter_frames, 1))
    contact_mask_end = np.tile([cit], (end_inter_frames, 1))
    contact_mask = np.concatenate((contact_mask_start, contact_mask, contact_mask_end), axis=0)
    
    # 6. 执行插值
    input_data = np.concatenate((root_trans, root_rot, dof), axis=1)
    output_data, contact_mask = interpolate_motion(
        input_data=input_data,
        start_ext_frames=start_inter_frames,
        end_ext_frames=end_inter_frames,
        default_pose=default_pose,
        contact_mask=contact_mask,
        fix_root_rot=False
    )
    
    # 7. 生成最终PBHC格式数据
    final_root_trans = output_data[:, :3]
    final_root_rot = output_data[:, 3:7]
    final_dof = output_data[:, 7:]
    final_root_aa = R.from_quat(final_root_rot).as_rotvec()
    
    final_pose_aa = np.zeros((output_data.shape[0], 27, 3), dtype=np.float32)
    final_pose_aa[:, 0, :] = final_root_aa
    
    final_data = {
        'root_trans_offset': final_root_trans.astype(np.float32),
        'pose_aa': final_pose_aa,
        'dof': final_dof.astype(np.float32),
        'root_rot': final_root_rot.astype(np.float32),
        'smpl_joints': np.zeros_like(final_pose_aa),
        'fps': 50,
        'contact_mask': contact_mask.astype(np.float32)
    }
    
    # 8. 保存为GMR兼容格式（用于CSV转换）
    # 使用原始GMR的键名，以便batch_gmr_pkl_to_csv.py可以处理
    gmr_compatible_data = {
        'root_pos': final_root_trans.astype(np.float32),
        'root_rot': final_root_rot.astype(np.float32),
        'dof_pos': final_dof.astype(np.float32),
        'fps': 50,
        'local_body_pos': None,  # 可视化不需要，但加载时需要
        'link_body_list': None   # 可视化不需要，但加载时需要
    }
    
    # 使用pickle保存（与原始GMR格式一致）
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(gmr_compatible_data, f)
    
    print(f"\n✅ 处理完成！")
    print(f"输出帧数: {final_dof.shape[0]} (原始{end_frame-start_frame} + 开头{start_inter_frames} + 结尾{end_inter_frames})")
    print(f"输出文件: {output_file}")
    print(f"文件大小: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"格式: GMR兼容格式 (可用batch_gmr_pkl_to_csv.py转换)")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GMR数据转换并添加插值')
    parser.add_argument('--input', type=str, required=True, help='输入的GMR PKL文件')
    parser.add_argument('--output', type=str, help='输出的PBHC PKL文件（默认自动生成）')
    parser.add_argument('--start_inter_frame', type=int, default=50, help='开头插值帧数')
    parser.add_argument('--end_inter_frame', type=int, default=50, help='结尾插值帧数')
    parser.add_argument('--start', type=int, default=0, help='起始帧')
    parser.add_argument('--end', type=int, default=-1, help='结束帧')
    
    args = parser.parse_args()
    
    # 自动生成输出文件名
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_interp_S{args.start_inter_frame}_E{args.end_inter_frame}.pkl")
    
    process_gmr_file(
        input_file=args.input,
        output_file=args.output,
        start_inter_frames=args.start_inter_frame,
        end_inter_frames=args.end_inter_frame,
        start_frame=args.start,
        end_frame=args.end
    )

