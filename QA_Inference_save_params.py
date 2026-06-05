import numpy as np
from Ant_quant import *
env = gym.make('Ant-v5')
import torch as th
import zipfile
import imageio
import io
import mujoco
from scipy.spatial.transform import Rotation as R
from compression_utils import *


from Ant_quant import *
# model = PPO.load("/home/ritwik/MuJoCo_Quant/logs_ant/best_model_fixscale_1/best_model", env=env)

save_wt_scales_activations = True
NEURONS = 64
cfg = {
    "quant_act": True,
    "quant_weights": True,
    "activation": "relu",
}
env = gym.make("Ant-v5")
model_path = "/home/ritwik/MuJoCo_Quant/logs_ant_new/_fixscale_64_20250831_141022/Quant_ant_69.zip"

# Get geom IDs for the ant's feet
model = env.unwrapped.model
data = env.unwrapped.data
# sim = env.unwrapped.sim

foot_geom_names = ["aux_1_geom", "aux_2_geom", "aux_3_geom", "aux_4_geom"]
foot_geom_ids = [model.geom(name).id for name in foot_geom_names]

foot_geom_names = ["left_ankle_geom", "right_ankle_geom", "third_ankle_geom", "fourth_ankle_geom"]
foot_geom_ids = [model.geom(name).id for name in foot_geom_names]

def check_foot_contacts(data, foot_geom_ids, floor_id):
    contacts = np.zeros(len(foot_geom_ids), dtype=int)
    for i, fid in enumerate(foot_geom_ids):
        for c in range(data.ncon):
            contact = data.contact[c]
            if ((contact.geom1 == fid and contact.geom2 == floor_id) or
                (contact.geom2 == fid and contact.geom1 == floor_id)):
                contacts[i] = 1
                break
    return contacts

# floor ID
floor_id = model.geom("floor").id

def foot_contact_flags(env_model, env_data):
    # qpos: [x, y, z, qw, qx, qy, qz, joint_angles...]
    # The first 7 are torso pos+quat, after that are joint angles.
    # To approximate contact, use foot height from site_xpos or body_xpos.
    
    # foot_geom_names = ["left_ankle_geom", "right_ankle_geom", "third_ankle_geom", "fourth_ankle_geom"]
    # foot_geom_names = ["aux_1_geom", "aux_2_geom", "aux_3_geom", "aux_4_geom"]
    # foot_geom_names = ["right_leg_geom", "left_leg_geom", "back_leg_geom", "rightback_leg_geom"]
    # foot_geom_names = ["floor"]
    foot_geom_names = ["left_ankle_geom", "left_leg_geom", "aux_1_geom", "aux_2_geom", "aux_3_geom", "aux_4_geom"]
    # foot_geom_names = ["flfl"]
    foot_geom_ids = [model.geom(name).id for name in foot_geom_names]
    contacts = {}
    for id in foot_geom_ids:
        z = env_data.geom_xpos[id][2]  # z-coordinate of the foot geom
        contacts[f] = (z < 0.08)  # threshold (tune depending on env scale)
    return contacts

def log_contacts(data, model, step):
    """
    Logs all contacts for the current timestep.
    """
    ncon = data.ncon
    logs = []
    for c in range(ncon):
        contact = data.contact[c]
        g1 = model.geom_id2name(contact.geom1)
        g2 = model.geom_id2name(contact.geom2)
        logs.append({
            "step": step,
            "geom1": g1,
            "geom2": g2,
            "pos": contact.pos.copy(),   # contact position (x, y, z)
            "dist": contact.dist,        # distance (negative = penetration)
            "frame": contact.frame.copy(),  # contact normal + tangent basis
        })
    return logs


def forward_pass(obs):
    obs_quant = torch.round(torch.tensor(obs)*2**e0/m0).to(torch.float32)
    obs_quant = torch.clip(obs_quant, -127, 127)
    a1 = torch.round((torch.matmul(obs_quant,w1.T)+b1)*m1/2**e1)
    # a1 = torch.relu(a1)
    a1 = torch.clip(a1, 0, 255)
    a2 = torch.round((torch.matmul(a1,w2.T)+b2)*m2/2**e2)
    # a2 = torch.relu(a2)
    a2 = torch.clip(a2, 0, 255)
    a3 = torch.round((torch.matmul(a2,w3.T)+b3)*m3/2**e3)
    a3 = torch.clip(a3, -127, 127)

    return a3

def save_forward_pass(obs):
    obs_quant = torch.round(torch.tensor(obs)*2**e0/m0).to(torch.float32)
    obs_quant = torch.clip(obs_quant, -127, 127)
    a1 = torch.round((torch.matmul(obs_quant,w1.T)+b1)*m1/2**e1)
    # a1 = torch.relu(a1)
    a1 = torch.clip(a1, 0, 255)
    a2 = torch.round((torch.matmul(a1,w2.T)+b2)*m2/2**e2)
    # a2 = torch.relu(a2)
    a2 = torch.clip(a2, 0, 255)
    a3 = torch.round((torch.matmul(a2,w3.T)+b3)*m3/2**e3)
    a3 = torch.clip(a3, -127, 127)

    return obs_quant, a1, a2, a3, w1, w2, w3, b1, b2, b3, m0, m1, m2, m3, e0, e1, e2, e3




def batch_frexp_refactored(inputs):
    """
    Decompose the scaling factor into mantissa and twos exponent, ensuring:
    - Mantissa lies in [0, 2^16)
    - Exponent remains meaningful (adjusted to match the scaled mantissa)

    Parameters:
    ----------
    inputs: Tensor
        Scaling factor

    Returns:
    -------
    mantissa: Tensor
    exponent: Tensor
    """
    shape_of_input = inputs.size()

    # Flatten the input tensor to a 1D tensor
    inputs = inputs.view(-1)

    # Decompose into mantissa and exponent
    output_m, output_e = np.frexp(inputs.cpu().numpy())

    # Prepare adjusted mantissa and exponent lists
    tmp_m = []
    tmp_e = []

    # Define the bias to scale mantissa to [0, 2^16)
    scale_factor = 16  # 2^16

    for idx, (m, e) in enumerate(zip(output_m, output_e)):
        # Scale mantissa from [0.5, 1) to [0, 2^16)
        int_m_shifted = int(Decimal(m * (2 ** scale_factor)).quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP))

        # Adjust the exponent to account for the scaling of the mantissa
        e -= scale_factor
        e= e*-1
        e = max(0, min(255, e))
        # Ensure mantissa and exponent are within valid bounds
        if int_m_shifted >= 2 ** 16:  # Cap mantissa at maximum (65535)
            int_m_shifted = 2 ** 16 - 1

        elif inputs[idx] >= 65535:
            int_m_shifted = 65535
            e = 0

        tmp_m.append(int_m_shifted)
        tmp_e.append(e)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tensor = torch.tensor(0.).to(device)

    return torch.from_numpy(np.array(tmp_m)).to(device).view(shape_of_input), \
           torch.from_numpy(np.array(tmp_e)).to(device).view(shape_of_input)


# model_path = "/home/ritwik/MuJoCo_Quant/logs_ant_new/_fixscale_256_20250828_133853/Quant_ant_38.zip"
# Open the zip and load policy.pth
with zipfile.ZipFile(model_path, "r") as archive:
    with archive.open("policy.pth", "r") as f:
        state_dict = th.load(io.BytesIO(f.read()), map_location="cpu")

Sx = state_dict['mlp_extractor.policy_net.act1.act_scaling_factor']
print(Sx)
Sw = state_dict['mlp_extractor.policy_net.fc1.fc_scaling_factor']
Sx1 = state_dict['mlp_extractor.policy_net.act2.act_scaling_factor']
Sw1 = state_dict['mlp_extractor.policy_net.fc2.fc_scaling_factor']
# Sx2 = state_dict['mlp_extractor.policy_net.act3.act_scaling_factor']
Sw2 = state_dict['action_net.fc1.fc_scaling_factor']
Sx2 = state_dict['action_net.act1.act_scaling_factor']
Sw3= state_dict['action_net.fc1.fc_scaling_factor'] 
Sx3 = state_dict['action_net.act2.act_scaling_factor']


m0, e0 = batch_frexp_refactored(Sx)
m1, e1 = batch_frexp_refactored(Sw*Sx/Sx1)
m2, e2 = batch_frexp_refactored(Sw1*Sx1/Sx2)
m3, e3 = batch_frexp_refactored(Sx2*Sw2/Sx3)

w1 = state_dict['mlp_extractor.policy_net.fc1.weight_integer']
b1 = state_dict['mlp_extractor.policy_net.fc1.bias_integer']
w2 = state_dict['mlp_extractor.policy_net.fc2.weight_integer']
b2 = state_dict['mlp_extractor.policy_net.fc2.bias_integer']
w3 = state_dict['action_net.fc1.weight_integer']
b3 = state_dict['action_net.fc1.bias_integer']

env = gym.make('Ant-v5')
obs = env.reset()
if save_wt_scales_activations:
    obs_quant, a1, a2, a3, w1, w2, w3, b1, b2, b3, m0, m1, m2, m3, e0, e1, e2, e3 = save_forward_pass(obs[0])
    # Save these tensors to a .csv file 
    path = f"_{NEURONS}_Quant_activations_weights/"
    path_scales = path + "Scales/"
    path_weights = path + "Weights/"
    path_activations = path + "Activations/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_scales, exist_ok=True)
    os.makedirs(path_weights, exist_ok=True)
    os.makedirs(path_activations, exist_ok=True)
    np.savetxt(path_activations + "obs_quant.csv", obs_quant.cpu().numpy(), delimiter=",")
    np.savetxt(path_activations + "a1.csv", a1.cpu().numpy(), delimiter=",")
    np.savetxt(path_activations + "a2.csv", a2.cpu().numpy(), delimiter=",")        
    np.savetxt(path_activations + "a3.csv", a3.cpu().numpy(), delimiter=",")
    np.savetxt(path_weights + "w1.csv", w1.T.cpu().numpy(), delimiter=",")
    np.savetxt(path_weights + "w2.csv", w2.T.cpu().numpy(), delimiter=",")
    np.savetxt(path_weights + "w3.csv", w3.T.cpu().numpy(), delimiter=",")
    np.savetxt(path_weights + "b1.csv", b1.cpu().numpy(), delimiter=",")
    np.savetxt(path_weights + "b2.csv", b2.cpu().numpy(), delimiter=",")
    np.savetxt(path_weights + "b3.csv", b3.cpu().numpy(), delimiter=",")
    np.savetxt(path_scales + "m0.csv", m0.cpu().numpy().reshape(1,), delimiter=",")
    np.savetxt(path_scales + "m1.csv", m1.cpu().numpy().reshape(1,),   delimiter=",")
    np.savetxt(path_scales + "m2.csv", m2.cpu().numpy().reshape(1,),   delimiter=",")
    np.savetxt(path_scales + "m3.csv", m3.cpu().numpy().reshape(1,),   delimiter=",")
    np.savetxt(path_scales + "e0.csv", e0.cpu().numpy().reshape(1,),   delimiter=",")
    np.savetxt(path_scales + "e1.csv", e1.cpu().numpy().reshape(1,),   delimiter=",")
    np.savetxt(path_scales + "e2.csv", e2.cpu().numpy().reshape(1,),   delimiter=",")
    np.savetxt(path_scales + "e3.csv", e3.cpu().numpy().reshape(1,),   delimiter=",")
    exit()

env = gym.make('Ant-v5')
# obs = env.reset()

import imageio
action_space_high = env.action_space.high
action_space_low = env.action_space.low
frames = []
cd_list = []
for _ in range(10):
    timestep=0
    contact_counts = {f: 0 for f in ["front_left_foot","front_right_foot","back_left_foot","back_right_foot"]}
    contacts_over_time = []
    all_contact_logs = []
    obs = env.reset()
    model = env.unwrapped.model
    data = env.unwrapped.data
    total_reward= 0
    done = False
    while not done and timestep < 1000:
        if timestep==0:
            action = forward_pass(obs[0])*Sx3
        else:
            action = forward_pass(obs)*Sx3
        
        # action = torch.tanh(action) * torch.tensor(env.action_space.high)
        action = action.cpu().numpy()
        # action[0], action[1], action[2] = 0, 0, 0
        
        action = np.clip(action, action_space_low, action_space_high)
        # if timestep ==0:
        #     action, _ = model.policy.predict(obs[0], deterministic=True)
        # else:
        #     action, _ = model.policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        c = foot_contact_flags(model, data)
        print("rendered")
        obs_req = obs
        # break
        done = terminated or truncated
        
        # if done:
            # print(timestep)
        total_reward += reward
        timestep+=1
    print(total_reward)
    # proportions = {f: contact_counts[f]/total_steps for f in contact_counts}
    # print(proportions)
# video_path = f"env_trial_quant.mp4"
# imageio.mimsave(video_path, frames, fps=30)
# print(f"Saved video to {video_path}")
