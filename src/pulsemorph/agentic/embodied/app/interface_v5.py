"""Exact public tracker tensor interface, independent of private phrase profiles."""
INPUTS=('current_anchor_rot','current_dof_pos','current_dof_vel','current_root_local_ang_vel','historical_processed_actions','mimic_future_anchor_rot','mimic_future_dof_pos','mimic_future_dof_vel')
OUTPUTS=('actions','joint_pos_targets','stiffness_targets','damping_targets')
MODEL_SHA='a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c'
BODY_SHA='e3926a6578cafaee38d0a7a3b7e7ebb0031f623c69e5cd34aa97d61e943b7e8e'
CONTROLLER_SHA='57d81b60dda9aeb30c57d3428a822f203a83bca399c5abe82ffea7a9aaf8b139'
def interface_guard(session):
    shapes=[['batch_size',4],['batch_size',29],['batch_size',29],['batch_size',3],['batch_size',1,29],['batch_size',4,4],['batch_size',4,29],['batch_size',4,29]]
    if [(n.name,n.shape,n.type)for n in session.get_inputs()]!=[(n,s,'tensor(float)')for n,s in zip(INPUTS,shapes)] or [(n.name,n.shape,n.type)for n in session.get_outputs()]!=[(n,['batch_size',29],'tensor(float)')for n in OUTPUTS]:raise ValueError('Unsupported exact tracker graph interface')
